//! Interactive & Declarative Use-Case Dataset Curation, Raw Dataset Reformatting,
//! and Teacher Agent Distillation Engine.
//!
//! Supports single and multi-teacher consensus across CLI agents (`agy`, `claude`, `codex`, `gemini`)
//! and local daemons (`ollama:<model>`, `ollaya:<model>`) to:
//! 1. Synthesize tailored decision schemas and datasets directly from a free-text Use-Case description.
//! 2. Reformat arbitrary user datasets (`.jsonl`, `.csv`, `.json`, `.txt`) into canonical
//!    Ollaya/Trainaya training records with calibrated soft target distributions and counterfactual twins.

use anyhow::{bail, Context, Result};
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::curatefile::{CurateMode, Curatefile, TargetDatasetFormat};

fn deserialize_optional_usize<'de, D>(
    deserializer: D,
) -> std::result::Result<Option<usize>, D::Error>
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
    /// Optional existing label in the raw dataset (used as a hint or verified by the teacher)
    #[serde(default, alias = "label", alias = "category", alias = "target")]
    pub gold_label: Option<String>,
    /// Capture any arbitrary extra fields from user JSON/CSV rows so nothing is lost during reformatting
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl RawInputItem {
    /// Return a structured or textual representation of the input state for the teacher & training record
    pub fn get_state_value(&self) -> serde_json::Value {
        if let Some(ref st) = self.state {
            if !st.is_null() {
                return st.clone();
            }
        }
        if let (Some(p), Some(h)) = (&self.premise, &self.hypothesis) {
            return serde_json::json!({
                "premise": p,
                "hypothesis": h
            });
        }
        if let Some(ref t) = self.text {
            return serde_json::Value::String(t.clone());
        }
        if let Some(ref p) = self.prompt {
            return serde_json::Value::String(p.clone());
        }
        if let Some(ref u) = self.utterance {
            return serde_json::Value::String(u.clone());
        }
        if !self.extra.is_empty() {
            let mut map = serde_json::Map::new();
            for (k, v) in &self.extra {
                if !matches!(
                    k.as_str(),
                    "id" | "questions" | "gold_label" | "label" | "category"
                ) {
                    map.insert(k.clone(), v.clone());
                }
            }
            if !map.is_empty() {
                return serde_json::Value::Object(map);
            }
        }
        serde_json::Value::String(String::new())
    }

    pub fn get_text(&self) -> String {
        match self.get_state_value() {
            serde_json::Value::String(s) => s,
            other => serde_json::to_string(&other).unwrap_or_default(),
        }
    }

    /// Resolve per-item question and valid labels if embedded, or fall back to global schema
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
    /// Optional cleaned/reformatted state produced by the teacher when reformatting messy raw data
    #[serde(default, alias = "state", alias = "clean_state")]
    pub reformatted_state: Option<serde_json::Value>,
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
    let (_, question, valid_labels, _) = extract_full_schema_metadata(schema_content);
    (question, valid_labels)
}

/// Extract `(kind, question, valid_labels, criteria_value)` from a schema JSON string
pub fn extract_full_schema_metadata(
    schema_content: &str,
) -> (String, String, Vec<String>, serde_json::Value) {
    let default_question = "Classify the input according to policy criteria.".to_string();
    let parsed: serde_json::Value = match serde_json::from_str(schema_content) {
        Ok(v) => v,
        Err(_) => {
            return (
                "choice".to_string(),
                default_question,
                Vec::new(),
                serde_json::Value::Null,
            )
        }
    };

    let root = parsed
        .get("decision")
        .or_else(|| parsed.get("questions").and_then(|q| q.get("decision")))
        .unwrap_or(&parsed);

    let kind = root
        .get("type")
        .or_else(|| root.get("kind"))
        .and_then(|v| v.as_str())
        .unwrap_or("choice")
        .to_lowercase();

    let question = root
        .get("instructions")
        .or_else(|| root.get("question"))
        .or_else(|| root.get("statement"))
        .and_then(|v| v.as_str())
        .unwrap_or("Classify the input according to policy criteria.")
        .to_string();

    let mut valid_labels = Vec::new();
    let mut criteria_val = serde_json::Value::Null;

    if kind == "noul" {
        valid_labels = vec!["false".to_string(), "true".to_string()];
    } else if let Some(criteria) = root.get("criteria").or_else(|| root.get("options")) {
        criteria_val = criteria.clone();
        match criteria {
            serde_json::Value::Object(map) => {
                for k in map.keys() {
                    valid_labels.push(k.clone());
                }
            }
            serde_json::Value::Array(arr) => {
                for (idx, item) in arr.iter().enumerate() {
                    if let Some(s) = item.as_str() {
                        if kind == "score" || s.to_lowercase().starts_with("level ") {
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

    (kind, question, valid_labels, criteria_val)
}

/// Resolve a guaranteed valid opposite/flipped label for a counterfactual twin example
pub fn resolve_counterfactual_label(
    primary_label: &str,
    explicit_cf_label: Option<&str>,
    valid_labels: &[String],
) -> Option<String> {
    let prim_trimmed = primary_label.trim();

    if let Some(cf_lbl) = explicit_cf_label {
        let cf_trimmed = cf_lbl.trim();
        if !cf_trimmed.is_empty() && !cf_trimmed.eq_ignore_ascii_case(prim_trimmed) {
            if let Some(matched) = valid_labels
                .iter()
                .find(|v| v.eq_ignore_ascii_case(cf_trimmed))
            {
                return Some(matched.clone());
            }
            return Some(cf_trimmed.to_string());
        }
    }

    if valid_labels.len() >= 2 {
        if let Some(opposite) = valid_labels
            .iter()
            .find(|v| !v.trim().eq_ignore_ascii_case(prim_trimmed))
        {
            return Some(opposite.clone());
        }
    }

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
            cloned.confidence *= 0.85;
            cloned.reasoning = Some(format!(
                "[Single response from {}]: {}",
                t_name,
                single.reasoning.as_deref().unwrap_or("")
            ));
            consolidated.push(cloned);
            continue;
        }

        let mut label_groups: HashMap<String, Vec<(&str, &TeacherAnnotation)>> = HashMap::new();
        for (t_name, anno) in &votes {
            let norm_label = anno.label.trim().to_lowercase();
            label_groups
                .entry(norm_label)
                .or_default()
                .push((*t_name, *anno));
        }

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

            let best_state = best_group
                .iter()
                .find_map(|(_, a)| a.reformatted_state.clone());
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
                reformatted_state: best_state,
                hard_negative: best_hn,
                hard_negative_label: best_hn_label,
            });
        } else {
            consolidated.push(TeacherAnnotation {
                id: item_id,
                label: best_group[0].1.label.clone(),
                confidence: 0.0,
                reasoning: Some(format!(
                    "QUARANTINED (Multi-Teacher Disagreement): {}",
                    vote_breakdown.join(" vs ")
                )),
                reformatted_state: None,
                hard_negative: None,
                hard_negative_label: None,
            });
        }
    }

    consolidated
}

/// Detect installed CLI agents and running local model daemons to assist interactive teacher selection
pub fn detect_available_teachers() -> Vec<String> {
    let mut found = Vec::new();
    for cli_bin in &["agy", "claude", "codex", "gemini"] {
        if Command::new("which")
            .arg(cli_bin)
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
        {
            found.push((*cli_bin).to_string());
        }
    }
    if ureq::get("http://127.0.0.1:11434/api/tags")
        .timeout(std::time::Duration::from_millis(250))
        .call()
        .is_ok()
    {
        found.push("ollama:<model>".to_string());
    }
    if ureq::get("http://127.0.0.1:11435/api/tags")
        .timeout(std::time::Duration::from_millis(250))
        .call()
        .is_ok()
    {
        found.push("ollaya:<model>".to_string());
    }
    if found.is_empty() {
        found.push("agy".to_string());
    }
    found
}

/// Execute teacher prompt using local CLI (`agy`, `claude`, `codex`, `gemini`) or local daemon (`ollama:<model>`, `ollaya:<model>`)
pub fn query_teacher(engine: &str, system_prompt: &str, user_prompt: &str) -> Result<String> {
    let combined_prompt = format!(
        "{}\n\nStrict requirement: Output ONLY valid raw JSON without any markdown code fences, headers, or explanations.\n\n{}",
        system_prompt, user_prompt
    );

    let engine_lower = engine.trim().to_lowercase();

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
            .with_context(|| {
                "Failed to execute 'claude' CLI. Ensure 'claude' is installed in PATH."
            })?;

        if !output.status.success() {
            let err = String::from_utf8_lossy(&output.stderr);
            bail!("'claude' CLI exited with error: {}", err.trim());
        }

        let raw = String::from_utf8_lossy(&output.stdout).trim().to_string();
        Ok(clean_json_output(&raw))
    } else if engine_lower == "codex" || engine_lower.starts_with("codex:") {
        let output = Command::new("codex")
            .arg("exec")
            .arg(&combined_prompt)
            .output()
            .with_context(|| {
                "Failed to execute 'codex' CLI. Ensure 'codex' is installed in PATH."
            })?;

        if !output.status.success() {
            let err = String::from_utf8_lossy(&output.stderr);
            bail!("'codex' CLI exited with error: {}", err.trim());
        }

        let raw = String::from_utf8_lossy(&output.stdout).trim().to_string();
        Ok(clean_json_output(&raw))
    } else if engine_lower == "gemini" || engine_lower.starts_with("gemini:") {
        let output = Command::new("gemini")
            .arg("-p")
            .arg(&combined_prompt)
            .output()
            .with_context(|| {
                "Failed to execute 'gemini' CLI. Ensure 'gemini' is installed in PATH."
            })?;

        if !output.status.success() {
            let err = String::from_utf8_lossy(&output.stderr);
            bail!("'gemini' CLI exited with error: {}", err.trim());
        }

        let raw = String::from_utf8_lossy(&output.stdout).trim().to_string();
        Ok(clean_json_output(&raw))
    } else if engine_lower.starts_with("ollama:") || engine_lower.starts_with("ollaya:") {
        let is_ollaya = engine_lower.starts_with("ollaya:");
        let model = if is_ollaya {
            engine.trim()[7..].trim()
        } else {
            engine.trim()[7..].trim()
        };
        let port = if is_ollaya { 11435 } else { 11434 };
        let endpoint = format!("http://127.0.0.1:{}/api/generate", port);
        let payload = serde_json::json!({
            "model": if model.is_empty() { "llama3.3:70b" } else { model },
            "prompt": combined_prompt,
            "stream": false,
            "format": "json"
        });

        let resp = ureq::post(&endpoint)
            .set("Content-Type", "application/json")
            .send_json(&payload)
            .with_context(|| format!("Failed to reach local daemon at {}", endpoint))?;

        let res_json: serde_json::Value = resp.into_json()?;
        let resp_text = res_json["response"]
            .as_str()
            .unwrap_or_default()
            .trim()
            .to_string();
        Ok(clean_json_output(&resp_text))
    } else {
        bail!(
            "Unsupported teacher engine '{}'. Supported engines: 'agy', 'claude', 'codex', 'gemini', 'ollama:<model>', 'ollaya:<model>'",
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
    let s = s.trim();
    // If teacher emitted leading commentary before '[' or '{', slice from the first JSON delimiter
    if !s.starts_with('{') && !s.starts_with('[') {
        if let Some(pos) = s.find(['{', '[']) {
            return s[pos..].trim().to_string();
        }
    }
    s.to_string()
}

/// Automatically synthesize a tailored Ollaya/Trainaya decision schema from a free-text Use-Case description
pub fn synthesize_schema_from_use_case(
    teacher: &str,
    use_case: &str,
    kind: &str,
    sample_preview: Option<&str>,
) -> Result<String> {
    let preview_block = if let Some(prev) = sample_preview {
        format!(
            "\nSample raw input rows from the user's dataset:\n{}\n",
            prev
        )
    } else {
        String::new()
    };

    let system_prompt = format!(
        "You are an expert Decision Schema Architect for Ollaya and Trainaya decision models.\n\
         Design a crisp, non-overlapping decision schema tailored to the user's Use-Case.\n\
         Requested decision primitive type: '{}'.\n\
         - If type is 'choice': 'criteria' MUST be a JSON object mapping 2 to 8 snake_case category keys to clear 1-sentence boundary descriptions.\n\
         - If type is 'noul': 'criteria' MUST be null, and 'instructions' MUST be a direct Yes/No verification question.\n\
         - If type is 'score': 'criteria' MUST be a JSON array of 5 rubric strings ordered from index 0 (lowest) to index 4 (highest).\n\
         Output strictly a JSON object matching:\n\
         {{\"decision\": {{\"type\": \"{}\", \"instructions\": \"...\", \"criteria\": ...}}}}",
        kind, kind
    );

    let user_prompt = format!(
        "Target Use-Case Description:\n\"{}\"{}\nGenerate the JSON decision schema now.",
        use_case, preview_block
    );

    let raw_schema = query_teacher(teacher, &system_prompt, &user_prompt)?;
    let parsed: serde_json::Value = serde_json::from_str(&raw_schema)
        .with_context(|| "Teacher returned invalid JSON while synthesizing decision schema")?;

    let normalized = if parsed.get("decision").is_some() {
        parsed
    } else {
        serde_json::json!({ "decision": parsed })
    };

    Ok(serde_json::to_string_pretty(&normalized)?)
}

/// Load raw input rows from `.jsonl`, `.json` array, `.csv` / `.tsv`, or `.txt`
pub fn load_raw_inputs(path: &Path) -> Result<Vec<RawInputItem>> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    // 1. CSV / TSV support: parse header columns into structured `state` JSON objects
    if ext == "csv" || ext == "tsv" {
        let delimiter = if ext == "tsv" { b'\t' } else { b',' };
        let mut rdr = csv::ReaderBuilder::new()
            .delimiter(delimiter)
            .flexible(true)
            .from_path(path)
            .with_context(|| format!("Opening CSV input at {:?}", path))?;

        let headers: Vec<String> = rdr
            .headers()?
            .iter()
            .map(|h| h.trim().to_string())
            .collect();

        let mut items = Vec::new();
        for (idx, result) in rdr.records().enumerate() {
            let record = result?;
            let mut state_map = serde_json::Map::new();
            let mut gold_label = None;
            let mut text_val = None;

            for (col_idx, val) in record.iter().enumerate() {
                let col_name = headers
                    .get(col_idx)
                    .cloned()
                    .unwrap_or_else(|| format!("col_{}", col_idx));
                let trimmed_val = val.trim();
                if trimmed_val.is_empty() {
                    continue;
                }
                let lower_col = col_name.to_lowercase();
                if matches!(
                    lower_col.as_str(),
                    "label" | "gold_label" | "target" | "category" | "class"
                ) {
                    gold_label = Some(trimmed_val.to_string());
                } else {
                    if matches!(
                        lower_col.as_str(),
                        "text" | "prompt" | "utterance" | "input" | "message" | "body"
                    ) && text_val.is_none()
                    {
                        text_val = Some(trimmed_val.to_string());
                    }
                    state_map.insert(
                        col_name,
                        serde_json::Value::String(trimmed_val.to_string()),
                    );
                }
            }

            let state = if state_map.len() == 1 && text_val.is_some() {
                serde_json::Value::String(text_val.clone().unwrap())
            } else {
                serde_json::Value::Object(state_map)
            };

            items.push(RawInputItem {
                id: Some(idx),
                text: text_val,
                state: Some(state),
                questions: None,
                prompt: None,
                utterance: None,
                premise: None,
                hypothesis: None,
                gold_label,
                extra: HashMap::new(),
            });
        }
        return Ok(items);
    }

    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Opening raw inputs at {:?}", path))?;
    let trimmed_all = content.trim();

    // 2. JSON array (`[...]`) support
    if trimmed_all.starts_with('[') {
        if let Ok(mut arr) = serde_json::from_str::<Vec<RawInputItem>>(trimmed_all) {
            for (idx, item) in arr.iter_mut().enumerate() {
                if item.id.is_none() {
                    item.id = Some(idx);
                }
            }
            return Ok(arr);
        }
    }

    // 3. JSONL or plain text lines
    let file = File::open(path).with_context(|| format!("Opening raw inputs at {:?}", path))?;
    let reader = BufReader::new(file);
    let mut items = Vec::new();

    for (idx, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        if trimmed.starts_with('{') {
            if let Ok(mut item) = serde_json::from_str::<RawInputItem>(trimmed) {
                if item.id.is_none() {
                    item.id = Some(idx);
                }
                items.push(item);
                continue;
            }
        }

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
            extra: HashMap::new(),
        });
    }

    Ok(items)
}

fn prompt_line(prompt_label: &str, default_val: Option<&str>) -> Result<String> {
    let mut stdout = io::stdout();
    if let Some(def) = default_val {
        if !def.is_empty() {
            print!("  {} [{}]: ", prompt_label, def.cyan());
        } else {
            print!("  {}: ", prompt_label);
        }
    } else {
        print!("  {}: ", prompt_label);
    }
    stdout.flush()?;

    let mut buf = String::new();
    io::stdin().read_line(&mut buf)?;
    let trimmed = buf.trim();
    if trimmed.is_empty() {
        Ok(default_val.unwrap_or("").to_string())
    } else {
        Ok(trimmed.to_string())
    }
}

/// Interactive terminal wizard guiding the user through Use-Case definition, teacher selection, and dataset curation/reformatting
pub fn run_interactive_wizard(cf: &mut Curatefile) -> Result<()> {
    println!(
        "\n  {} {}",
        "🧭".cyan().bold(),
        "Interactive Use-Case Curation & Teacher Distillation Wizard"
            .bold()
            .white()
    );
    println!(
        "  {}",
        "Define your target Use-Case, pick a Teacher Agent (agy, claude, ollama...), and synthesize or reformat a training dataset."
            .dimmed()
    );

    // 1. Free-text Use-Case description
    println!(
        "\n  {} {}",
        "1.".cyan().bold(),
        "Describe your final Use-Case (domain context, decision rules, what the model must decide):"
            .bold()
    );
    let current_uc = cf.use_case.clone().unwrap_or_else(|| {
        "Classify incoming user requests or documents into actionable domain decisions".to_string()
    });
    let uc_input = prompt_line("Use-Case Description", Some(&current_uc))?;
    if !uc_input.is_empty() {
        cf.use_case = Some(uc_input);
    }

    // 2. Workflow Mode
    println!(
        "\n  {} {}",
        "2.".cyan().bold(),
        "Select Curation Workflow:".bold()
    );
    println!(
        "     {} {} — Generate a tailored dataset & schema from your Use-Case",
        "[1]".green().bold(),
        "synth".bold()
    );
    println!(
        "     {} {} — Reformat & distill an existing dataset (.jsonl, .csv, .json, .txt) for this Use-Case",
        "[2]".cyan().bold(),
        "reformat".bold()
    );
    println!(
        "     {} {} — Reformat dataset + synthesize counterfactual hard-negative twins",
        "[3]".purple().bold(),
        "augment".bold()
    );
    let default_mode_choice = match cf.mode {
        CurateMode::Synth => "1",
        CurateMode::Label => "2",
        CurateMode::Augment => "3",
    };
    let mode_choice = prompt_line("Choice (1/2/3)", Some(default_mode_choice))?;
    cf.mode = match mode_choice.trim().to_lowercase().as_str() {
        "1" | "synth" | "generate" => CurateMode::Synth,
        "2" | "reformat" | "label" | "distill" => CurateMode::Label,
        "3" | "augment" | "contrastive" => CurateMode::Augment,
        _ => cf.mode,
    };

    // 3. Input Dataset (if reformat or augment)
    if matches!(cf.mode, CurateMode::Label | CurateMode::Augment) {
        println!(
            "\n  {} {}",
            "3.".cyan().bold(),
            "Path to raw dataset to reformat (.jsonl, .csv, .tsv, .json, .txt):".bold()
        );
        let default_inp = cf
            .input
            .as_ref()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| "./data/raw.jsonl".to_string());
        let inp_str = prompt_line("Input dataset path", Some(&default_inp))?;
        if !inp_str.is_empty() {
            cf.input = Some(PathBuf::from(inp_str));
        }
    }

    // 4. Teacher Agent Selection
    let detected = detect_available_teachers();
    println!(
        "\n  {} {} (detected: {})",
        "4.".cyan().bold(),
        "Select Teacher Agent(s) to distill (comma-separated for multi-teacher consensus):".bold(),
        detected.join(", ").green()
    );
    let teacher_input = prompt_line("Teacher Agent", Some(&cf.teacher))?;
    if !teacher_input.is_empty() {
        cf.teacher = teacher_input;
    }

    // 5. Decision Primitive Type & Optional Schema
    println!(
        "\n  {} {}",
        "5.".cyan().bold(),
        "Decision Primitive Kind ('choice' = multi-option, 'noul' = yes/no verification, 'score' = 0..4 rubric):"
            .bold()
    );
    let kind_input = prompt_line("Decision Kind", Some(&cf.kind))?;
    if matches!(
        kind_input.to_lowercase().as_str(),
        "choice" | "noul" | "score"
    ) {
        cf.kind = kind_input.to_lowercase();
    }

    let default_schema = cf
        .schema
        .as_ref()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default();
    let schema_input = prompt_line(
        "Optional explicit Schema JSON path (leave empty to auto-generate from Use-Case)",
        Some(&default_schema),
    )?;
    if schema_input.trim().is_empty() {
        cf.schema = None;
    } else {
        cf.schema = Some(PathBuf::from(schema_input));
    }

    // 6. Sample Count (for synth) & Output Path
    if cf.mode == CurateMode::Synth {
        let count_str = prompt_line("Number of samples to synthesize", Some(&cf.count.to_string()))?;
        if let Ok(c) = count_str.parse::<usize>() {
            cf.count = c.max(1);
        }
    }

    let out_default = cf.output.to_string_lossy().to_string();
    let out_input = prompt_line("Output dataset path (.jsonl)", Some(&out_default))?;
    if !out_input.is_empty() {
        cf.output = PathBuf::from(out_input);
    }

    let save_cf = prompt_line("Save this recipe to .Curatefile? (y/n)", Some("y"))?;
    if save_cf.to_lowercase().starts_with('y') {
        let cf_path = Path::new(".Curatefile");
        std::fs::write(cf_path, cf.to_curatefile_string())?;
        println!(
            "  {} Saved curation recipe to {}",
            "✓".green().bold(),
            cf_path.display()
        );
    }

    Ok(())
}

/// Execute curation pipeline
pub fn run_curate(mut cf: Curatefile) -> Result<()> {
    crate::utils::print_banner("Use-Case Dataset Curation & Teacher Distillation");

    if cf.interactive && io::stdin().is_terminal() {
        run_interactive_wizard(&mut cf)?;
    }

    let teachers = cf.teachers();
    println!("\n  {} Curation Configuration:", "🔬".cyan().bold());
    if let Some(ref uc) = cf.use_case {
        println!("    • Use-Case          : {}", uc.bold().white());
    }
    if teachers.len() > 1 {
        println!(
            "    • Teacher Consensus : {} ({} agents voting)",
            teachers.join(" + ").bold().green(),
            teachers.len()
        );
    } else {
        println!("    • Teacher Agent     : {}", cf.teacher.bold().green());
    }
    println!(
        "    • Workflow Mode     : {} ({:?})",
        cf.mode.as_str().bold().cyan(),
        cf.mode
    );
    println!("    • Target Format     : {:?} ({})", cf.format, cf.kind);
    if let Some(ref inp) = cf.input {
        println!("    • Raw Input Dataset : {:?}", inp);
    } else {
        println!("    • Raw Input Dataset : None (Direct Use-Case Synthesis)");
    }
    println!("    • Batch Size        : {}", cf.batch_size);
    println!("    • Min Confidence    : {:.0}%", cf.min_confidence * 100.0);
    println!(
        "    • Counterfactuals   : {}",
        if cf.add_hard_negatives {
            "Enabled (Contrastive Twin Pairs)".green()
        } else {
            "Disabled".yellow()
        }
    );
    println!("    • Output Target     : {:?}", cf.output);

    // Pre-load raw items if in Reformat/Label or Augment mode so we can show a preview to the schema generator if needed
    let preloaded_items = if matches!(cf.mode, CurateMode::Label | CurateMode::Augment) {
        let input_path = cf.input.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "MODE '{}' requires an INPUT dataset file path (via --input, Curatefile, or interactive wizard)",
                cf.mode.as_str()
            )
        })?;
        let items = load_raw_inputs(input_path)?;
        if items.is_empty() {
            bail!("Raw input dataset at {:?} is empty.", input_path);
        }
        Some(items)
    } else {
        None
    };

    // Resolve or synthesize decision schema from Use-Case
    let schema_content = if let Some(ref sc_path) = cf.schema {
        println!("    • Decision Schema   : {:?}", sc_path);
        std::fs::read_to_string(sc_path)
            .with_context(|| format!("Reading schema at {:?}", sc_path))?
    } else if let Some(ref uc) = cf.use_case {
        println!(
            "    • Decision Schema   : {}",
            "Auto-synthesizing from Use-Case via Teacher Agent...".cyan()
        );
        let sample_preview = preloaded_items.as_ref().map(|items| {
            let preview_vec: Vec<serde_json::Value> = items
                .iter()
                .take(3)
                .map(|it| it.get_state_value())
                .collect();
            serde_json::to_string_pretty(&preview_vec).unwrap_or_default()
        });
        let generated_schema = synthesize_schema_from_use_case(
            &teachers[0],
            uc,
            &cf.kind,
            sample_preview.as_deref(),
        )?;
        let schema_out_path = cf.output.with_extension("schema.json");
        if let Some(parent) = schema_out_path.parent() {
            if !parent.as_os_str().is_empty() {
                let _ = std::fs::create_dir_all(parent);
            }
        }
        let _ = std::fs::write(&schema_out_path, &generated_schema);
        println!(
            "    • Generated Schema  : Saved to {:?}",
            schema_out_path
        );
        generated_schema
    } else {
        serde_json::json!({
            "decision": {
                "type": cf.kind,
                "instructions": "Classify the input into the appropriate domain category.",
                "criteria": {
                    "general": "Standard or routine domain request",
                    "priority": "Requires priority attention or specialized handling"
                }
            }
        })
        .to_string()
    };

    let (schema_kind, schema_question, valid_labels, schema_criteria) =
        extract_full_schema_metadata(&schema_content);
    let effective_kind = if !schema_kind.is_empty() {
        schema_kind
    } else {
        cf.kind.clone()
    };

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

    let use_case_context = cf
        .use_case
        .as_deref()
        .map(|uc| format!("Target Use-Case: \"{}\"\n", uc))
        .unwrap_or_default();

    match cf.mode {
        CurateMode::Label | CurateMode::Augment => {
            let is_augment = cf.mode == CurateMode::Augment;
            let raw_items = preloaded_items.unwrap();

            println!(
                "\n  {} Loaded {} raw input records for {}...",
                "📂".cyan(),
                raw_items.len(),
                if is_augment {
                    "Use-Case reformatting + counterfactual twin distillation"
                } else {
                    "Use-Case reformatting & teacher distillation"
                }
            );

            let pb = ProgressBar::new(raw_items.len() as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("    [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} rows processed: {msg}")
                    .unwrap()
                    .progress_chars("━╾─"),
            );

            let system_prompt = format!(
                "You are a SOTA Decision Dataset Reformatter and Teacher Distiller.\n\
                 {}\
                 Decision Schema:\n\
                 {}\n\n\
                 For each raw input item:\n\
                 1. Assign the single most accurate 'label' from the schema's valid_labels ({:?}). If 'existing_label' is present, map or verify it against valid_labels.\n\
                 2. Estimate a calibrated probability 'confidence' in [0.50, 0.99] reflecting how unambiguous the decision is.\n\
                 3. Optionally provide 'reformatted_state' if the raw input state needs cleaning/structuring for training (otherwise omit or keep null).\n\
                 4. {} provide a minimally perturbed Counterfactual Twin ('hard_negative') where changing ONLY ONE causal fact in the input genuinely flips the decision to 'hard_negative_label'.\n\
                 Output ONLY a JSON array of objects with keys:\n\
                 - 'id': integer matching input id\n\
                 - 'label': string matching one of valid_labels\n\
                 - 'confidence': float between 0.0 and 1.0\n\
                 - 'reasoning': concise rationale\n\
                 - 'reformatted_state': optional cleaned string or JSON object\n\
                 - 'hard_negative': minimally modified state text that flips the decision\n\
                 - 'hard_negative_label': the flipped target label from valid_labels",
                use_case_context,
                schema_content,
                valid_labels,
                if is_augment || cf.add_hard_negatives {
                    "Always"
                } else {
                    "Optionally"
                }
            );

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
                            "state": it.get_state_value(),
                            "question": item_q,
                            "valid_labels": item_labels,
                        });
                        if let Some(ref gl) = it.gold_label {
                            obj["existing_label"] = serde_json::json!(gl);
                        }
                        obj
                    })
                    .collect();

                let user_prompt = serde_json::to_string_pretty(&user_batch)?;
                let annotations =
                    query_teachers_consensus(&teachers, &system_prompt, &user_prompt, &batch_ids)?;

                for anno in annotations {
                    let source_item = chunk.iter().find(|c| c.id == Some(anno.id));
                    let final_state = anno
                        .reformatted_state
                        .clone()
                        .filter(|v| !v.is_null())
                        .or_else(|| source_item.map(|c| c.get_state_value()))
                        .unwrap_or_else(|| serde_json::Value::String(String::new()));

                    let (item_question, item_valid_labels) = source_item
                        .map(|c| c.resolve_item_schema(&schema_question, &valid_labels))
                        .unwrap_or_else(|| (schema_question.clone(), valid_labels.clone()));

                    let primary_label =
                        if let Some(gl) = source_item.and_then(|c| c.gold_label.as_deref()) {
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
                        write_curated_record(
                            &mut out_file,
                            cf.format,
                            &effective_kind,
                            &final_state,
                            &primary_label,
                            anno.confidence,
                            &item_question,
                            &item_valid_labels,
                            &schema_criteria,
                            cf.use_case.as_deref(),
                        )?;
                        *class_distribution.entry(primary_label.clone()).or_insert(0) += 1;
                        total_accepted += 1;

                        if cf.add_hard_negatives || is_augment {
                            if let Some(hn_text) = anno.hard_negative.as_deref() {
                                if !hn_text.trim().is_empty() {
                                    if let Some(flipped_label) = resolve_counterfactual_label(
                                        &primary_label,
                                        anno.hard_negative_label.as_deref(),
                                        &item_valid_labels,
                                    ) {
                                        let hn_state =
                                            serde_json::Value::String(hn_text.trim().to_string());
                                        write_curated_record(
                                            &mut out_file,
                                            cf.format,
                                            &effective_kind,
                                            &hn_state,
                                            &flipped_label,
                                            anno.confidence.clamp(0.85, 0.96),
                                            &item_question,
                                            &item_valid_labels,
                                            &schema_criteria,
                                            cf.use_case.as_deref(),
                                        )?;
                                        *class_distribution.entry(flipped_label).or_insert(0) += 1;
                                        total_accepted += 1;
                                        total_counterfactuals += 1;
                                    }
                                }
                            }
                        }
                    } else {
                        let quar_item = serde_json::json!({
                            "state": final_state,
                            "candidate_label": anno.label,
                            "confidence": anno.confidence,
                            "reasoning": anno.reasoning,
                        });
                        writeln!(quar_file, "{}", serde_json::to_string(&quar_item)?)?;
                        total_quarantined += 1;
                    }
                    pb.inc(1);
                }
            }
            pb.finish_with_message("Dataset reformatting & teacher distillation complete.");
        }
        CurateMode::Synth => {
            println!(
                "\n  {} Synthesizing {} Use-Case training records via {}...",
                "✨".cyan().bold(),
                cf.count,
                teachers[0].bold().green()
            );

            let pb = ProgressBar::new(cf.count as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("    [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} samples generated: {msg}")
                    .unwrap()
                    .progress_chars("━╾─"),
            );

            let system_prompt = format!(
                "You are an expert synthetic dataset architect for SOTA System-1 decision models.\n\
                 {}\
                 Based on the decision schema:\n\
                 {}\n\n\
                 Generate balanced, realistic, domain-authentic training examples covering all valid options ({:?}).\n\
                 Include subtle boundary cases, negations, realistic formatting, and a minimally perturbed counterfactual twin ('hard_negative' + 'hard_negative_label') where flipping one causal detail flips the decision.\n\
                 Return ONLY a JSON array of objects with keys:\n\
                 - 'text': realistic input state (string or structured JSON object)\n\
                 - 'label': string matching one of the schema criteria keys ({:?})\n\
                 - 'confidence': calibrated float between 0.80 and 0.99\n\
                 - 'hard_negative': minimally perturbed counterfactual state that flips the label\n\
                 - 'hard_negative_label': flipped criteria key from {:?}",
                use_case_context, schema_content, valid_labels, valid_labels, valid_labels
            );

            #[derive(Deserialize)]
            struct SynthSample {
                #[serde(alias = "state")]
                text: serde_json::Value,
                label: String,
                #[serde(default = "default_confidence")]
                confidence: f64,
                #[serde(default, alias = "counterfactual_state")]
                hard_negative: Option<String>,
                #[serde(default, alias = "counterfactual_label")]
                hard_negative_label: Option<String>,
            }

            let batch_step = cf.batch_size.clamp(5, 25);
            let mut remaining = cf.count;

            while remaining > 0 {
                let req_count = remaining.min(batch_step);
                let user_prompt = format!(
                    "Generate {} diverse, non-repetitive training samples for this Use-Case now.",
                    req_count
                );
                let teacher_resp = query_teacher(&teachers[0], &system_prompt, &user_prompt)?;
                let samples: Vec<SynthSample> =
                    serde_json::from_str(&teacher_resp).unwrap_or_default();

                if samples.is_empty() {
                    break;
                }

                for s in samples {
                    if s.confidence >= cf.min_confidence {
                        write_curated_record(
                            &mut out_file,
                            cf.format,
                            &effective_kind,
                            &s.text,
                            &s.label,
                            s.confidence,
                            &schema_question,
                            &valid_labels,
                            &schema_criteria,
                            cf.use_case.as_deref(),
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
                                        let hn_state =
                                            serde_json::Value::String(hn_text.trim().to_string());
                                        write_curated_record(
                                            &mut out_file,
                                            cf.format,
                                            &effective_kind,
                                            &hn_state,
                                            &flipped_label,
                                            s.confidence.clamp(0.85, 0.96),
                                            &schema_question,
                                            &valid_labels,
                                            &schema_criteria,
                                            cf.use_case.as_deref(),
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

                remaining = remaining.saturating_sub(req_count);
            }
            pb.finish_with_message("Use-Case synthesis complete.");
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
        "\n  {} Curated dataset ready at: {:?}",
        "✓".green().bold(),
        cf.output
    );
    println!(
        "    • Adapt a GGUF/HF model : {}",
        format!(
            "trainaya adapt --from <model> -t <tag> --data {}",
            cf.output.display()
        )
        .cyan()
    );
    println!(
        "    • Fine-tune end-to-end  : {}\n",
        format!("trainaya train --data {} -t <tag>", cf.output.display()).cyan()
    );

    Ok(())
}

/// Build a calibrated soft probability distribution across `valid_labels` from the teacher's label and confidence
pub fn build_soft_target_distribution(
    kind: &str,
    target_idx: usize,
    num_options: usize,
    confidence: f64,
) -> Vec<f64> {
    if num_options < 2 {
        return vec![1.0];
    }
    let conf = confidence.clamp(1.0 / (num_options as f64), 0.99);
    let remainder = 1.0 - conf;

    if kind == "score" && num_options >= 3 {
        // Ordinal unimodal smoothing around `target_idx` for score rubrics
        let mut weights = vec![0.0; num_options];
        let mut off_sum = 0.0;
        for (i, w) in weights.iter_mut().enumerate() {
            if i != target_idx {
                let dist = (i as isize - target_idx as isize).unsigned_abs() as f64;
                *w = (-1.35 * dist).exp();
                off_sum += *w;
            }
        }
        let mut probs = vec![0.0; num_options];
        for i in 0..num_options {
            if i == target_idx {
                probs[i] = (conf * 10000.0).round() / 10000.0;
            } else if off_sum > 0.0 {
                probs[i] = ((remainder * (weights[i] / off_sum)) * 10000.0).round() / 10000.0;
            }
        }
        return probs;
    }

    let uniform_other = remainder / ((num_options - 1) as f64);
    (0..num_options)
        .map(|i| {
            let p = if i == target_idx { conf } else { uniform_other };
            (p * 10000.0).round() / 10000.0
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn write_curated_record(
    writer: &mut File,
    format: TargetDatasetFormat,
    kind: &str,
    state: &serde_json::Value,
    label: &str,
    confidence: f64,
    schema_question: &str,
    valid_labels: &[String],
    schema_criteria: &serde_json::Value,
    use_case: Option<&str>,
) -> Result<()> {
    let state_text = match state {
        serde_json::Value::String(s) => s.clone(),
        other => serde_json::to_string(other).unwrap_or_default(),
    };

    match format {
        TargetDatasetFormat::Decision => {
            if valid_labels.len() >= 2 {
                let target_idx = valid_labels
                    .iter()
                    .position(|v| v.eq_ignore_ascii_case(label.trim()))
                    .unwrap_or(0);
                let target_probs = build_soft_target_distribution(
                    kind,
                    target_idx,
                    valid_labels.len(),
                    confidence,
                );
                let canonical_label = &valid_labels[target_idx];

                let mut row = serde_json::json!({
                    "kind": kind,
                    "state": state,
                    "question": schema_question,
                    "options": valid_labels,
                    "target": target_probs,
                    "target_index": target_idx,
                    "label": canonical_label,
                    "confidence": confidence,
                });
                if !schema_criteria.is_null() {
                    row["criteria"] = schema_criteria.clone();
                }
                if let Some(uc) = use_case {
                    row["use_case"] = serde_json::json!(uc);
                }
                writeln!(writer, "{}", serde_json::to_string(&row)?)?;
            } else {
                let row = serde_json::json!({
                    "state": state,
                    "label": label,
                    "confidence": confidence,
                });
                writeln!(writer, "{}", serde_json::to_string(&row)?)?;
            }
        }
        TargetDatasetFormat::Classification => {
            let row = serde_json::json!({
                "text": state_text,
                "label": label
            });
            writeln!(writer, "{}", serde_json::to_string(&row)?)?;
        }
        TargetDatasetFormat::Nli => {
            let row = serde_json::json!({
                "premise": state_text,
                "hypothesis": format!("This input belongs to {}", label),
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
        assert_eq!(
            resolve_counterfactual_label("general", None, &valid),
            Some("urgent".to_string())
        );
        assert_eq!(
            resolve_counterfactual_label("urgent", Some("urgent"), &valid),
            Some("general".to_string())
        );
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
                reformatted_state: None,
                hard_negative: Some("Non-urgent minor update".to_string()),
                hard_negative_label: Some("general".to_string()),
            },
            TeacherAnnotation {
                id: 1,
                label: "urgent".to_string(),
                confidence: 0.91,
                reasoning: Some("Seems urgent".to_string()),
                reformatted_state: None,
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
                reformatted_state: None,
                hard_negative: None,
                hard_negative_label: None,
            },
            TeacherAnnotation {
                id: 1,
                label: "general".to_string(),
                confidence: 0.89,
                reasoning: Some("Routine inquiry".to_string()),
                reformatted_state: None,
                hard_negative: None,
                hard_negative_label: None,
            },
        ];

        let per_teacher = vec![("agy".to_string(), t1), ("claude".to_string(), t2)];
        let consolidated = consolidate_teacher_annotations(&per_teacher, &[0, 1]);
        assert_eq!(consolidated.len(), 2);

        assert_eq!(consolidated[0].id, 0);
        assert_eq!(consolidated[0].label, "urgent");
        assert!((consolidated[0].confidence - 0.95).abs() < 1e-6);
        assert_eq!(
            consolidated[0].hard_negative.as_deref(),
            Some("Non-urgent minor update")
        );

        assert_eq!(consolidated[1].id, 1);
        assert_eq!(consolidated[1].confidence, 0.0);
        assert!(consolidated[1]
            .reasoning
            .as_deref()
            .unwrap_or("")
            .contains("Disagreement"));
    }

    #[test]
    fn test_soft_target_distribution_choice_and_score() {
        let probs_choice = build_soft_target_distribution("choice", 1, 3, 0.90);
        assert_eq!(probs_choice.len(), 3);
        assert!((probs_choice[1] - 0.90).abs() < 1e-4);
        assert!((probs_choice[0] - 0.05).abs() < 1e-4);
        assert!((probs_choice[2] - 0.05).abs() < 1e-4);

        let probs_score = build_soft_target_distribution("score", 4, 5, 0.85);
        assert_eq!(probs_score.len(), 5);
        // Ordinal unimodal decay: level 3 > level 2 > level 1 > level 0
        assert!(probs_score[4] > probs_score[3]);
        assert!(probs_score[3] > probs_score[2]);
        assert!(probs_score[2] > probs_score[1]);
        assert!(probs_score[1] > probs_score[0]);
    }
}
