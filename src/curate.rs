//! Dataset curation & teacher distillation engine.
//! Supports single and multi-teacher consensus across CLI engines (agy, claude, ollama)
//! to auto-label raw data, synthesize domain examples, and build counterfactual hard-negative twins.

use anyhow::{bail, Context, Result};
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::Command;

use crate::curatefile::{CurateMode, Curatefile, TargetDatasetFormat};

fn deserialize_optional_usize<'de, D>(deserializer: D) -> std::result::Result<Option<usize>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let val: Option<serde_json::Value> = Option::deserialize(deserializer)?;
    match val {
        Some(serde_json::Value::Number(n)) => Ok(n.as_u64().map(|v| v as usize)),
        Some(serde_json::Value::String(s)) => Ok(s.parse::<usize>().ok()),
        _ => Ok(None),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawInputItem {
    #[serde(default, deserialize_with = "deserialize_optional_usize")]
    pub id: Option<usize>,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub state: Option<serde_json::Value>,
    #[serde(default)]
    pub questions: Option<serde_json::Value>,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub utterance: Option<String>,
    #[serde(default)]
    pub premise: Option<String>,
    #[serde(default)]
    pub hypothesis: Option<String>,
    /// Optional gold label (e.g., from exported benchmark failures or existing datasets)
    #[serde(default, alias = "label")]
    pub gold_label: Option<String>,
    /// Optional model misprediction (from 'trainaya bench --export-failures')
    #[serde(default)]
    pub predicted_label: Option<String>,
    /// Optional failure severity tag (from 'trainaya bench --export-failures')
    #[serde(default)]
    pub failure_type: Option<String>,
}

impl RawInputItem {
    pub fn get_text(&self) -> String {
        self.text
            .clone()
            .or_else(|| self.prompt.clone())
            .or_else(|| self.utterance.clone())
            .or_else(|| match (&self.premise, &self.hypothesis) {
                (Some(p), Some(h)) => Some(format!("Premise: {} | Hypothesis: {}", p, h)),
                (Some(p), None) => Some(p.clone()),
                _ => None,
            })
            .or_else(|| self.state.as_ref().map(|s| s.to_string()))
            .unwrap_or_default()
    }

    /// Resolve per-item question and valid labels if embedded (e.g. from bench --export-failures), or fall back to global schema
    pub fn resolve_item_schema(
        &self,
        default_question: &str,
        default_labels: &[String],
    ) -> (String, Vec<String>) {
        if let Some(ref q_val) = self.questions {
            if let Ok(q_str) = serde_json::to_string(q_val) {
                let (q, labels) = extract_schema_metadata(&q_str);
                if labels.len() >= 2 {
                    return (q, labels);
                }
            }
        }
        (default_question.to_string(), default_labels.to_vec())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeacherAnnotation {
    pub id: usize,
    pub label: String,
    #[serde(default = "default_confidence")]
    pub confidence: f64,
    #[serde(default)]
    pub reasoning: Option<String>,
    /// Minimally perturbed counterfactual state where flipping 1 causal fact flips the decision
    #[serde(
        default,
        alias = "counterfactual_state",
        alias = "counterfactual",
        alias = "counterfactual_text"
    )]
    pub hard_negative: Option<String>,
    /// Target inverted label for the counterfactual twin
    #[serde(
        default,
        alias = "counterfactual_label",
        alias = "flipped_label",
        alias = "counterfactual_target"
    )]
    pub hard_negative_label: Option<String>,
}

fn default_confidence() -> f64 {
    0.95
}

/// Extract the decision question instructions and ordered valid criteria labels from a schema JSON string
pub fn extract_schema_metadata(schema_content: &str) -> (String, Vec<String>) {
    let default_question = "Classify the input according to policy criteria.".to_string();
    let parsed: serde_json::Value = match serde_json::from_str(schema_content) {
        Ok(v) => v,
        Err(_) => return (default_question, Vec::new()),
    };

    let root = parsed
        .get("decision")
        .or_else(|| {
            parsed
                .get("questions")
                .and_then(|q| q.get("decision"))
        })
        .unwrap_or(&parsed);

    let question = root
        .get("instructions")
        .or_else(|| root.get("question"))
        .and_then(|v| v.as_str())
        .unwrap_or("Classify the input according to policy criteria.")
        .to_string();

    let mut valid_labels = Vec::new();
    if let Some(criteria) = root.get("criteria").or_else(|| root.get("options")) {
        match criteria {
            serde_json::Value::Object(map) => {
                for k in map.keys() {
                    valid_labels.push(k.clone());
                }
            }
            serde_json::Value::Array(arr) => {
                for (idx, item) in arr.iter().enumerate() {
                    if let Some(s) = item.as_str() {
                        // If rubric levels like "Level 0: ...", use index string if > 2 items
                        if s.to_lowercase().starts_with("level ") {
                            valid_labels.push(idx.to_string());
                        } else {
                            valid_labels.push(s.to_string());
                        }
                    } else {
                        valid_labels.push(idx.to_string());
                    }
                }
            }
            _ => {}
        }
    }

    (question, valid_labels)
}

/// Resolve a guaranteed valid opposite/flipped label for a counterfactual twin example
pub fn resolve_counterfactual_label(
    primary_label: &str,
    explicit_cf_label: Option<&str>,
    valid_labels: &[String],
) -> Option<String> {
    let prim_trimmed = primary_label.trim();

    // 1. If the teacher explicitly provided a flipped label that differs from primary_label, prefer it
    if let Some(cf_lbl) = explicit_cf_label {
        let cf_trimmed = cf_lbl.trim();
        if !cf_trimmed.is_empty() && !cf_trimmed.eq_ignore_ascii_case(prim_trimmed) {
            // If valid_labels is non-empty, normalize case against valid_labels if possible
            if let Some(matched) = valid_labels
                .iter()
                .find(|v| v.eq_ignore_ascii_case(cf_trimmed))
            {
                return Some(matched.clone());
            }
            return Some(cf_trimmed.to_string());
        }
    }

    // 2. Otherwise pick the opposite label from schema's valid_labels
    if valid_labels.len() >= 2 {
        if let Some(opposite) = valid_labels
            .iter()
            .find(|v| !v.trim().eq_ignore_ascii_case(prim_trimmed))
        {
            return Some(opposite.clone());
        }
    }

    // 3. Fallback canonical inversions for boolean / NLI primitives
    match prim_trimmed.to_lowercase().as_str() {
        "true" => Some("false".to_string()),
        "false" => Some("true".to_string()),
        "yes" => Some("no".to_string()),
        "no" => Some("yes".to_string()),
        "entailment" => Some("contradiction".to_string()),
        "contradiction" => Some("entailment".to_string()),
        "0" => Some("1".to_string()),
        "1" => Some("0".to_string()),
        _ => None,
    }
}

/// Consolidate annotations from one or multiple teachers using strict consensus voting.
/// - For 1 teacher: returns annotations as-is.
/// - For 2 teachers: requires unanimous agreement; disagreement forces confidence = 0.0 (quarantine).
/// - For >= 3 teachers: requires strict majority (> 50%); disagreement forces confidence = 0.0 (quarantine).
pub fn consolidate_teacher_annotations(
    per_teacher: &[(String, Vec<TeacherAnnotation>)],
    batch_ids: &[usize],
) -> Vec<TeacherAnnotation> {
    if per_teacher.is_empty() {
        return Vec::new();
    }
    if per_teacher.len() == 1 {
        return per_teacher[0].1.clone();
    }

    let mut consolidated = Vec::with_capacity(batch_ids.len());

    for &item_id in batch_ids {
        let mut votes: Vec<(&str, &TeacherAnnotation)> = Vec::new();
        for (t_name, annos) in per_teacher {
            if let Some(found) = annos.iter().find(|a| a.id == item_id) {
                votes.push((t_name.as_str(), found));
            }
        }

        if votes.is_empty() {
            continue;
        }
        if votes.len() == 1 {
            let (t_name, single) = votes[0];
            let mut cloned = single.clone();
            // Discount confidence slightly when other teachers failed to return an item
            cloned.confidence *= 0.85;
            cloned.reasoning = Some(format!(
                "[Single response from {}]: {}",
                t_name,
                single.reasoning.as_deref().unwrap_or("")
            ));
            consolidated.push(cloned);
            continue;
        }

        // Group votes by lowercase label
        let mut label_groups: HashMap<String, Vec<(&str, &TeacherAnnotation)>> = HashMap::new();
        for (t_name, anno) in &votes {
            let norm_label = anno.label.trim().to_lowercase();
            label_groups
                .entry(norm_label)
                .or_default()
                .push((*t_name, *anno));
        }

        // Find winning label group
        let best_group = label_groups
            .values()
            .max_by(|a, b| {
                a.len().cmp(&b.len()).then_with(|| {
                    let sum_a: f64 = a.iter().map(|(_, x)| x.confidence).sum();
                    let sum_b: f64 = b.iter().map(|(_, x)| x.confidence).sum();
                    sum_a
                        .partial_cmp(&sum_b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
            })
            .unwrap();

        let total_votes = votes.len();
        let winning_votes = best_group.len();
        let has_consensus = winning_votes * 2 > total_votes;

        let vote_breakdown: Vec<String> = votes
            .iter()
            .map(|(t, a)| format!("{}='{}'({:.2})", t, a.label, a.confidence))
            .collect();

        if has_consensus {
            let canonical_label = best_group[0].1.label.clone();
            let mean_conf: f64 =
                best_group.iter().map(|(_, a)| a.confidence).sum::<f64>() / (winning_votes as f64);
            let agreement_ratio = (winning_votes as f64) / (total_votes as f64);
            let final_conf = (mean_conf * agreement_ratio).clamp(0.0, 1.0);

            let best_hn = best_group
                .iter()
                .find_map(|(_, a)| a.hard_negative.clone().filter(|s| !s.trim().is_empty()));
            let best_hn_label = best_group.iter().find_map(|(_, a)| {
                a.hard_negative_label
                    .clone()
                    .filter(|s| !s.trim().is_empty())
            });

            consolidated.push(TeacherAnnotation {
                id: item_id,
                label: canonical_label,
                confidence: final_conf,
                reasoning: Some(format!(
                    "Consensus ({}/{}): {} | {}",
                    winning_votes,
                    total_votes,
                    vote_breakdown.join(", "),
                    best_group[0].1.reasoning.as_deref().unwrap_or("")
                )),
                hard_negative: best_hn,
                hard_negative_label: best_hn_label,
            });
        } else {
            // Disagreement -> force confidence = 0.0 so item is automatically quarantined
            consolidated.push(TeacherAnnotation {
                id: item_id,
                label: best_group[0].1.label.clone(),
                confidence: 0.0,
                reasoning: Some(format!(
                    "QUARANTINED (Multi-Teacher Disagreement): {}",
                    vote_breakdown.join(" vs ")
                )),
                hard_negative: None,
                hard_negative_label: None,
            });
        }
    }

    consolidated
}

/// Execute teacher prompt using local CLI (agy, claude) or local Ollama daemon
pub fn query_teacher(engine: &str, system_prompt: &str, user_prompt: &str) -> Result<String> {
    let combined_prompt = format!(
        "{}\n\nStrict requirement: Output ONLY valid raw JSON without any markdown code fences, headers, or explanations.\n\n{}",
        system_prompt, user_prompt
    );

    let engine_lower = engine.to_lowercase();

    if engine_lower == "agy" || engine_lower.starts_with("agy:") {
        let output = Command::new("agy")
            .arg("-p")
            .arg(&combined_prompt)
            .arg("--output-format")
            .arg("text")
            .output()
            .with_context(|| "Failed to execute 'agy' CLI. Ensure 'agy' is installed in PATH.")?;

        if !output.status.success() {
            let err = String::from_utf8_lossy(&output.stderr);
            bail!("'agy' CLI exited with error: {}", err.trim());
        }

        let raw = String::from_utf8_lossy(&output.stdout).trim().to_string();
        Ok(clean_json_output(&raw))
    } else if engine_lower == "claude" || engine_lower.starts_with("claude:") {
        let output = Command::new("claude")
            .arg("-p")
            .arg(&combined_prompt)
            .output()
            .with_context(|| "Failed to execute 'claude' CLI. Ensure 'claude' is installed in PATH.")?;

        if !output.status.success() {
            let err = String::from_utf8_lossy(&output.stderr);
            bail!("'claude' CLI exited with error: {}", err.trim());
        }

        let raw = String::from_utf8_lossy(&output.stdout).trim().to_string();
        Ok(clean_json_output(&raw))
    } else if engine_lower.starts_with("ollama:") {
        let model = engine_lower.strip_prefix("ollama:").unwrap_or("llama3.3:70b");
        let payload = serde_json::json!({
            "model": model,
            "prompt": combined_prompt,
            "stream": false,
            "format": "json"
        });

        let resp = ureq::post("http://127.0.0.1:11434/api/generate")
            .set("Content-Type", "application/json")
            .send_json(&payload)
            .with_context(|| "Failed to reach local Ollama daemon at http://127.0.0.1:11434")?;

        let res_json: serde_json::Value = resp.into_json()?;
        let resp_text = res_json["response"]
            .as_str()
            .unwrap_or_default()
            .trim()
            .to_string();
        Ok(clean_json_output(&resp_text))
    } else {
        bail!(
            "Unsupported teacher engine '{}'. Supported engines: 'agy' (default), 'claude', 'ollama:<model>'",
            engine
        );
    }
}

fn parse_teacher_annotations(teacher_resp: &str) -> Vec<TeacherAnnotation> {
    match serde_json::from_str::<Vec<TeacherAnnotation>>(teacher_resp) {
        Ok(parsed) => parsed,
        Err(_) => {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(teacher_resp) {
                if let Some(arr) = val
                    .get("annotations")
                    .or_else(|| val.get("items"))
                    .or_else(|| val.get("results"))
                {
                    serde_json::from_value(arr.clone()).unwrap_or_default()
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            }
        }
    }
}

/// Query all configured teachers and consolidate their annotations via strict consensus
pub fn query_teachers_consensus(
    teachers: &[String],
    system_prompt: &str,
    user_prompt: &str,
    batch_ids: &[usize],
) -> Result<Vec<TeacherAnnotation>> {
    let mut per_teacher = Vec::with_capacity(teachers.len());
    for engine in teachers {
        let resp = query_teacher(engine, system_prompt, user_prompt)?;
        let parsed = parse_teacher_annotations(&resp);
        per_teacher.push((engine.clone(), parsed));
    }
    Ok(consolidate_teacher_annotations(&per_teacher, batch_ids))
}

fn clean_json_output(raw: &str) -> String {
    let mut s = raw.trim();
    if s.starts_with("```json") {
        s = s.strip_prefix("```json").unwrap_or(s);
    } else if s.starts_with("```") {
        s = s.strip_prefix("```").unwrap_or(s);
    }
    if s.ends_with("```") {
        s = s.strip_suffix("```").unwrap_or(s);
    }
    s.trim().to_string()
}

/// Load raw input rows from file (supports JSONL, exported benchmark failure records, or plain text)
pub fn load_raw_inputs(path: &Path) -> Result<Vec<RawInputItem>> {
    let file = File::open(path).with_context(|| format!("Opening raw inputs at {:?}", path))?;
    let reader = BufReader::new(file);
    let mut items = Vec::new();

    for (idx, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Support either JSONL object or plain text line
        if trimmed.starts_with('{') {
            if let Ok(mut item) = serde_json::from_str::<RawInputItem>(trimmed) {
                if item.id.is_none() {
                    item.id = Some(idx);
                }
                items.push(item);
                continue;
            }
        }

        // Plain text fallback
        items.push(RawInputItem {
            id: Some(idx),
            text: Some(trimmed.to_string()),
            state: None,
            questions: None,
            prompt: None,
            utterance: None,
            premise: None,
            hypothesis: None,
            gold_label: None,
            predicted_label: None,
            failure_type: None,
        });
    }

    Ok(items)
}

/// Execute curation pipeline
pub fn run_curate(cf: Curatefile) -> Result<()> {
    crate::utils::print_banner("Ollaya Dataset Curation & Distillation");

    let teachers = cf.teachers();
    println!("  {} Curation Configuration:", "🔬".cyan().bold());
    if teachers.len() > 1 {
        println!(
            "    • Teacher Consensus : {} ({} engines voting)",
            teachers.join(" + ").bold().green(),
            teachers.len()
        );
    } else {
        println!("    • Teacher Engine    : {}", cf.teacher.bold().green());
    }
    println!("    • Curation Mode     : {:?}", cf.mode);
    println!("    • Target Format     : {:?}", cf.format);
    if let Some(ref inp) = cf.input {
        println!("    • Raw Input Data    : {:?}", inp);
    } else {
        println!("    • Raw Input Data    : None (Pure Synthetic Generation)");
    }
    if let Some(ref sc) = cf.schema {
        println!("    • Criteria Schema   : {:?}", sc);
    }
    println!("    • Batch Size        : {}", cf.batch_size);
    println!("    • Min Confidence    : {:.0}%", cf.min_confidence * 100.0);
    println!("    • Counterfactuals   : {}", if cf.add_hard_negatives { "Enabled (Twin Pairs)".green() } else { "Disabled".yellow() });
    println!("    • Output Target     : {:?}", cf.output);

    // Read schema if provided
    let schema_content = if let Some(ref sc_path) = cf.schema {
        std::fs::read_to_string(sc_path)
            .with_context(|| format!("Reading schema at {:?}", sc_path))?
    } else {
        serde_json::json!({
            "decision": {
                "type": "choice",
                "instructions": "Classify the input into appropriate domain category",
                "criteria": {
                    "general": "General category",
                    "urgent": "Urgent attention required"
                }
            }
        })
        .to_string()
    };

    let (schema_question, valid_labels) = extract_schema_metadata(&schema_content);

    if let Some(parent) = cf.output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let mut out_file = File::create(&cf.output)
        .with_context(|| format!("Creating output dataset at {:?}", cf.output))?;

    let quarantine_path = cf.output.with_extension("quarantine.jsonl");
    let mut quar_file = File::create(&quarantine_path)
        .with_context(|| format!("Creating quarantine dataset at {:?}", quarantine_path))?;

    let mut total_accepted = 0;
    let mut total_counterfactuals = 0;
    let mut total_quarantined = 0;
    let mut class_distribution: HashMap<String, usize> = HashMap::new();

    match cf.mode {
        CurateMode::Label | CurateMode::Augment => {
            let is_augment = cf.mode == CurateMode::Augment;
            let input_path = cf.input.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "MODE '{:?}' requires an INPUT file path in Curatefile or --input",
                    cf.mode
                )
            })?;

            let raw_items = load_raw_inputs(input_path)?;
            if raw_items.is_empty() {
                bail!("Raw input dataset at {:?} is empty.", input_path);
            }

            println!(
                "\n  {} Loaded {} input rows for {}...",
                "📂".cyan(),
                raw_items.len(),
                if is_augment {
                    "counterfactual twin augmentation"
                } else {
                    "annotation"
                }
            );

            let pb = ProgressBar::new(raw_items.len() as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("    [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} rows processed: {msg}")
                    .unwrap()
                    .progress_chars("━╾─"),
            );

            let system_prompt = if is_augment {
                format!(
                    "You are a SOTA Decision Model Counterfactual Architect.\n\
                     Default policy criteria (unless overridden by per-item 'valid_labels' / 'question'):\n\
                     {}\n\n\
                     For each input item (which may include 'gold_label', 'predicted_label', 'question', and 'valid_labels'),\n\
                     verify the true 'label' (MUST be one of the item's 'valid_labels' if provided, and should match 'gold_label' if present)\n\
                     and generate a minimally perturbed Counterfactual Twin ('hard_negative') where\n\
                     changing ONLY ONE causal fact in the text genuinely flips the decision to a different class ('hard_negative_label') from 'valid_labels'.\n\
                     Output a JSON array of objects with keys:\n\
                     - 'id': integer matching input id\n\
                     - 'label': verified true label matching the item's valid_labels\n\
                     - 'confidence': float between 0.0 and 1.0\n\
                     - 'reasoning': short causal explanation\n\
                     - 'hard_negative': minimally modified text that flips the decision\n\
                     - 'hard_negative_label': the flipped target label (must be a valid label different from 'label')",
                    schema_content
                )
            } else {
                format!(
                    "You are an expert dataset annotator and classifier.\n\
                     Your task is to assign the single most accurate label to each input item based on these policy criteria:\n\
                     {}\n\n\
                     For each input, output a JSON array of objects with keys:\n\
                     - 'id': integer matching input id\n\
                     - 'label': string matching one of the criteria keys (or the item's 'valid_labels' if present)\n\
                     - 'confidence': float between 0.0 and 1.0\n\
                     - 'reasoning': short rationale\n\
                     - 'hard_negative': optional minimally perturbed counterfactual text that flips the label\n\
                     - 'hard_negative_label': the flipped criteria key for 'hard_negative'",
                    schema_content
                )
            };

            for chunk in raw_items.chunks(cf.batch_size.max(1)) {
                let mut batch_ids = Vec::with_capacity(chunk.len());
                let user_batch: Vec<serde_json::Value> = chunk
                    .iter()
                    .enumerate()
                    .map(|(i, it)| {
                        let id = it.id.unwrap_or(i);
                        batch_ids.push(id);
                        let (item_q, item_labels) =
                            it.resolve_item_schema(&schema_question, &valid_labels);
                        let mut obj = serde_json::json!({
                            "id": id,
                            "text": it.get_text(),
                            "question": item_q,
                            "valid_labels": item_labels,
                        });
                        if let Some(ref gl) = it.gold_label {
                            obj["gold_label"] = serde_json::json!(gl);
                        }
                        if let Some(ref pl) = it.predicted_label {
                            obj["predicted_label"] = serde_json::json!(pl);
                        }
                        if let Some(ref ft) = it.failure_type {
                            obj["failure_type"] = serde_json::json!(ft);
                        }
                        obj
                    })
                    .collect();

                let user_prompt = serde_json::to_string_pretty(&user_batch)?;
                let annotations =
                    query_teachers_consensus(&teachers, &system_prompt, &user_prompt, &batch_ids)?;

                for anno in annotations {
                    let source_item = chunk.iter().find(|c| c.id == Some(anno.id));
                    let text = source_item.map(|c| c.get_text()).unwrap_or_default();
                    let (item_question, item_valid_labels) = source_item
                        .map(|c| c.resolve_item_schema(&schema_question, &valid_labels))
                        .unwrap_or_else(|| (schema_question.clone(), valid_labels.clone()));

                    // If augmenting benchmark failures that already carry a verified gold_label, ensure primary_label belongs to item_valid_labels
                    let primary_label = if let Some(gl) = source_item.and_then(|c| c.gold_label.as_deref()) {
                        if !item_valid_labels.is_empty()
                            && !item_valid_labels
                                .iter()
                                .any(|v| v.eq_ignore_ascii_case(anno.label.trim()))
                        {
                            gl.to_string()
                        } else {
                            anno.label.clone()
                        }
                    } else {
                        anno.label.clone()
                    };

                    if anno.confidence >= cf.min_confidence {
                        write_formatted_row_with_schema(
                            &mut out_file,
                            cf.format,
                            &text,
                            &primary_label,
                            &item_question,
                            &item_valid_labels,
                        )?;
                        *class_distribution.entry(primary_label.clone()).or_insert(0) += 1;
                        total_accepted += 1;

                        // Emit verified Counterfactual Twin if enabled or in Augment mode
                        if cf.add_hard_negatives || is_augment {
                            if let Some(hn_text) = anno.hard_negative.as_deref() {
                                if !hn_text.trim().is_empty() {
                                    if let Some(flipped_label) = resolve_counterfactual_label(
                                        &primary_label,
                                        anno.hard_negative_label.as_deref(),
                                        &item_valid_labels,
                                    ) {
                                        write_formatted_row_with_schema(
                                            &mut out_file,
                                            cf.format,
                                            hn_text.trim(),
                                            &flipped_label,
                                            &item_question,
                                            &item_valid_labels,
                                        )?;
                                        *class_distribution
                                            .entry(flipped_label)
                                            .or_insert(0) += 1;
                                        total_accepted += 1;
                                        total_counterfactuals += 1;
                                    }
                                }
                            }
                        }
                    } else {
                        // Quarantine low-confidence or teacher-disagreed rows for review
                        let quar_item = serde_json::json!({
                            "text": text,
                            "predicted_label": anno.label,
                            "confidence": anno.confidence,
                            "reasoning": anno.reasoning,
                        });
                        writeln!(quar_file, "{}", serde_json::to_string(&quar_item)?)?;
                        total_quarantined += 1;
                    }
                    pb.inc(1);
                }
            }
            pb.finish_with_message("Curation batch processing complete.");
        }
        CurateMode::Synth => {
            println!(
                "\n  {} Synthesizing {} training records from schema criteria...",
                "✨".cyan().bold(),
                cf.count
            );

            let pb = ProgressBar::new(cf.count as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("    [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} samples generated: {msg}")
                    .unwrap()
                    .progress_chars("━╾─"),
            );

            let system_prompt = format!(
                "You are an expert synthetic data generator for SOTA System-1 decision models.\n\
                 Based on the policy criteria:\n\
                 {}\n\n\
                 Generate balanced, highly realistic, nuanced training examples covering each category.\n\
                 Include boundary cases, negations, domain jargon, and optionally a minimally perturbed counterfactual twin.\n\
                 Return a JSON array of objects with keys:\n\
                 - 'text': realistic input state\n\
                 - 'label': string matching one of the criteria keys\n\
                 - 'confidence': float between 0.0 and 1.0\n\
                 - 'hard_negative': optional minimally perturbed counterfactual state that flips the label\n\
                 - 'hard_negative_label': optional flipped criteria key",
                schema_content
            );

            let user_prompt = format!("Generate {} diverse samples now.", cf.count);
            let teacher_resp = query_teacher(&teachers[0], &system_prompt, &user_prompt)?;

            #[derive(Deserialize)]
            struct SynthSample {
                text: String,
                label: String,
                #[serde(default = "default_confidence")]
                confidence: f64,
                #[serde(default, alias = "counterfactual_state")]
                hard_negative: Option<String>,
                #[serde(default, alias = "counterfactual_label")]
                hard_negative_label: Option<String>,
            }

            let samples: Vec<SynthSample> = serde_json::from_str(&teacher_resp).unwrap_or_default();
            for s in samples {
                if s.confidence >= cf.min_confidence {
                    write_formatted_row_with_schema(
                        &mut out_file,
                        cf.format,
                        &s.text,
                        &s.label,
                        &schema_question,
                        &valid_labels,
                    )?;
                    *class_distribution.entry(s.label.clone()).or_insert(0) += 1;
                    total_accepted += 1;

                    if cf.add_hard_negatives {
                        if let Some(hn_text) = s.hard_negative.as_deref() {
                            if !hn_text.trim().is_empty() {
                                if let Some(flipped_label) = resolve_counterfactual_label(
                                    &s.label,
                                    s.hard_negative_label.as_deref(),
                                    &valid_labels,
                                ) {
                                    write_formatted_row_with_schema(
                                        &mut out_file,
                                        cf.format,
                                        hn_text.trim(),
                                        &flipped_label,
                                        &schema_question,
                                        &valid_labels,
                                    )?;
                                    *class_distribution.entry(flipped_label).or_insert(0) += 1;
                                    total_accepted += 1;
                                    total_counterfactuals += 1;
                                }
                            }
                        }
                    }
                } else {
                    total_quarantined += 1;
                }
                pb.inc(1);
            }
            pb.finish_with_message("Synthesis complete.");
        }
    }

    // Print summary
    println!("\n  {} Curation Results Summary:", "📊".purple().bold());
    println!(
        "    • Accepted Samples      : {} (including {} counterfactual twins)",
        total_accepted.to_string().green().bold(),
        total_counterfactuals.to_string().cyan()
    );
    if total_quarantined > 0 {
        println!(
            "    • Quarantined Rows      : {} (saved to {:?})",
            total_quarantined.to_string().yellow(),
            quarantine_path
        );
    } else {
        println!("    • Quarantined Rows      : 0 (100% passed confidence & consensus gate)");
        let _ = std::fs::remove_file(&quarantine_path);
    }

    println!("\n  {} Class Distribution:", "🏷️".cyan().bold());
    for (label, count) in &class_distribution {
        println!("    • {:<24} : {:>4} samples", label.bold(), count);
    }

    println!(
        "\n  {} Curated dataset written to: {:?}",
        "✓".green().bold(),
        cf.output
    );
    println!("    You can now directly train on this dataset with 'trainaya train' or your .Trainfile!\n");

    Ok(())
}

fn write_formatted_row_with_schema(
    writer: &mut File,
    format: TargetDatasetFormat,
    text: &str,
    label: &str,
    schema_question: &str,
    valid_labels: &[String],
) -> Result<()> {
    match format {
        TargetDatasetFormat::Decision => {
            if valid_labels.len() >= 2 {
                let target_idx = valid_labels
                    .iter()
                    .position(|v| v.eq_ignore_ascii_case(label.trim()))
                    .unwrap_or(0);
                let row = serde_json::json!({
                    "state": text,
                    "label": label,
                    "question": schema_question,
                    "options": valid_labels,
                    "target_index": target_idx,
                });
                writeln!(writer, "{}", serde_json::to_string(&row)?)?;
            } else {
                let row = serde_json::json!({
                    "state": text,
                    "label": label,
                });
                writeln!(writer, "{}", serde_json::to_string(&row)?)?;
            }
        }
        TargetDatasetFormat::Classification => {
            let row = serde_json::json!({
                "text": text,
                "label": label
            });
            writeln!(writer, "{}", serde_json::to_string(&row)?)?;
        }
        TargetDatasetFormat::Nli => {
            let row = serde_json::json!({
                "premise": text,
                "hypothesis": format!("This action relates to {}", label),
                "label": label
            });
            writeln!(writer, "{}", serde_json::to_string(&row)?)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_counterfactual_label() {
        let valid = vec!["general".to_string(), "urgent".to_string()];
        // Auto-invert from schema when explicit is None
        assert_eq!(
            resolve_counterfactual_label("general", None, &valid),
            Some("urgent".to_string())
        );
        // Reject explicit if it is identical to primary_label and fallback to opposite
        assert_eq!(
            resolve_counterfactual_label("urgent", Some("urgent"), &valid),
            Some("general".to_string())
        );
        // Honor explicit when valid and different
        let valid3 = vec![
            "entailment".to_string(),
            "neutral".to_string(),
            "contradiction".to_string(),
        ];
        assert_eq!(
            resolve_counterfactual_label("entailment", Some("contradiction"), &valid3),
            Some("contradiction".to_string())
        );
    }

    #[test]
    fn test_multi_teacher_consensus_agreement_and_disagreement() {
        let t1 = vec![
            TeacherAnnotation {
                id: 0,
                label: "urgent".to_string(),
                confidence: 0.94,
                reasoning: Some("Clearly urgent".to_string()),
                hard_negative: Some("Non-urgent minor update".to_string()),
                hard_negative_label: Some("general".to_string()),
            },
            TeacherAnnotation {
                id: 1,
                label: "urgent".to_string(),
                confidence: 0.91,
                reasoning: Some("Seems urgent".to_string()),
                hard_negative: None,
                hard_negative_label: None,
            },
        ];

        let t2 = vec![
            TeacherAnnotation {
                id: 0,
                label: "urgent".to_string(),
                confidence: 0.96,
                reasoning: Some("Critical SLA".to_string()),
                hard_negative: None,
                hard_negative_label: None,
            },
            TeacherAnnotation {
                id: 1,
                label: "general".to_string(),
                confidence: 0.89,
                reasoning: Some("Routine inquiry".to_string()),
                hard_negative: None,
                hard_negative_label: None,
            },
        ];

        let per_teacher = vec![("agy".to_string(), t1), ("claude".to_string(), t2)];
        let consolidated = consolidate_teacher_annotations(&per_teacher, &[0, 1]);
        assert_eq!(consolidated.len(), 2);

        // Item 0: both agreed on "urgent" -> high confidence ~0.95, preserves counterfactual
        assert_eq!(consolidated[0].id, 0);
        assert_eq!(consolidated[0].label, "urgent");
        assert!((consolidated[0].confidence - 0.95).abs() < 1e-6);
        assert_eq!(
            consolidated[0].hard_negative.as_deref(),
            Some("Non-urgent minor update")
        );

        // Item 1: teachers disagreed ("urgent" vs "general") -> confidence forced to 0.0 (quarantined!)
        assert_eq!(consolidated[1].id, 1);
        assert_eq!(consolidated[1].confidence, 0.0);
        assert!(consolidated[1]
            .reasoning
            .as_deref()
            .unwrap_or("")
            .contains("Disagreement"));
    }

    #[test]
    fn test_load_exported_bench_failure_record() {
        let json_line = r#"{"id":"paws-1","subset":"paws","kind":"noul","text":"sentence_1: A | sentence_2: B","state":{},"questions":{},"gold_label":"true","predicted_label":"false","confidence":0.92,"failure_type":"misclassified_overconfident"}"#;
        let item: RawInputItem = serde_json::from_str(json_line).unwrap();
        assert_eq!(item.id, None); // String ID "paws-1" gracefully yields None so loader assigns sequential index
        assert_eq!(item.get_text(), "sentence_1: A | sentence_2: B");
        assert_eq!(item.gold_label.as_deref(), Some("true"));
        assert_eq!(item.predicted_label.as_deref(), Some("false"));
        assert_eq!(
            item.failure_type.as_deref(),
            Some("misclassified_overconfident")
        );
    }
}
