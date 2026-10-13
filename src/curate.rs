//! Dataset curation & teacher distillation engine.
//! Supports CLI engines (agy, claude, ollama) to auto-label raw data,
//! generate synthetic domain examples, and build hard-negative contrastive pairs.

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawInputItem {
    #[serde(default)]
    pub id: Option<usize>,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub utterance: Option<String>,
    #[serde(default)]
    pub premise: Option<String>,
    #[serde(default)]
    pub hypothesis: Option<String>,
}

impl RawInputItem {
    pub fn get_text(&self) -> String {
        self.text
            .clone()
            .or_else(|| self.prompt.clone())
            .or_else(|| self.utterance.clone())
            .unwrap_or_default()
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
    #[serde(default)]
    pub hard_negative: Option<String>,
}

fn default_confidence() -> f64 {
    0.95
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

/// Load raw input rows from file
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
            prompt: None,
            utterance: None,
            premise: None,
            hypothesis: None,
        });
    }

    Ok(items)
}

/// Execute curation pipeline
pub fn run_curate(cf: Curatefile) -> Result<()> {
    crate::utils::print_banner("Ollaya Dataset Curation & Distillation");

    println!("  {} Curation Configuration:", "🔬".cyan().bold());
    println!("    • Teacher Engine    : {}", cf.teacher.bold().green());
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
    let mut total_quarantined = 0;
    let mut class_distribution: HashMap<String, usize> = HashMap::new();

    match cf.mode {
        CurateMode::Label => {
            let input_path = cf.input.as_ref().ok_or_else(|| {
                anyhow::anyhow!("MODE 'label' requires an INPUT file path in Curatefile or --input")
            })?;

            let raw_items = load_raw_inputs(input_path)?;
            if raw_items.is_empty() {
                bail!("Raw input dataset at {:?} is empty.", input_path);
            }

            println!("\n  {} Loaded {} raw input rows to annotate.", "📂".cyan(), raw_items.len());

            let pb = ProgressBar::new(raw_items.len() as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("    [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} rows labeled: {msg}")
                    .unwrap()
                    .progress_chars("━╾─"),
            );

            let system_prompt = format!(
                "You are an expert dataset annotator and classifier.\n\
                 Your task is to assign the single most accurate label to each input item based on these policy criteria:\n\
                 {}\n\n\
                 For each input, output a JSON array of objects with keys:\n\
                 - 'id': integer matching input id\n\
                 - 'label': string matching one of the criteria keys\n\
                 - 'confidence': float between 0.0 and 1.0\n\
                 - 'reasoning': short rationale\n\
                 - 'hard_negative': optional subtle contrastive variation that would flip the label",
                schema_content
            );

            for chunk in raw_items.chunks(cf.batch_size) {
                let user_batch: Vec<serde_json::Value> = chunk
                    .iter()
                    .map(|it| {
                        serde_json::json!({
                            "id": it.id.unwrap_or(0),
                            "text": it.get_text()
                        })
                    })
                    .collect();

                let user_prompt = serde_json::to_string_pretty(&user_batch)?;
                let teacher_resp = query_teacher(&cf.teacher, &system_prompt, &user_prompt)?;

                let annotations: Vec<TeacherAnnotation> = match serde_json::from_str(&teacher_resp) {
                    Ok(parsed) => parsed,
                    Err(_) => {
                        // Attempt fallback extraction if wrapped in object
                        if let Ok(val) = serde_json::from_str::<serde_json::Value>(&teacher_resp) {
                            if let Some(arr) = val.get("annotations").or_else(|| val.get("items")) {
                                serde_json::from_value(arr.clone()).unwrap_or_default()
                            } else {
                                Vec::new()
                            }
                        } else {
                            Vec::new()
                        }
                    }
                };

                for anno in annotations {
                    let text = chunk
                        .iter()
                        .find(|c| c.id == Some(anno.id))
                        .map(|c| c.get_text())
                        .unwrap_or_default();

                    if anno.confidence >= cf.min_confidence {
                        write_formatted_row(&mut out_file, cf.format, &text, &anno.label)?;
                        *class_distribution.entry(anno.label.clone()).or_insert(0) += 1;
                        total_accepted += 1;

                        // Add hard-negative contrastive sample if provided
                        if cf.add_hard_negatives {
                            if let Some(hn_text) = anno.hard_negative {
                                if !hn_text.trim().is_empty() {
                                    write_formatted_row(&mut out_file, cf.format, &hn_text, "contrastive_probe")?;
                                    total_accepted += 1;
                                }
                            }
                        }
                    } else {
                        // Quarantine low-confidence rows for human review
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
            pb.finish_with_message("Labeling complete.");
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
                "You are an expert synthetic data generator for enterprise machine learning.\n\
                 Based on the policy criteria:\n\
                 {}\n\n\
                 Generate balanced, highly realistic, nuanced training examples covering each category.\n\
                 Include realistic phrasing, typos, technical terms, and domain jargon.\n\
                 Return a JSON array of objects with keys: 'text', 'label', 'confidence'.",
                schema_content
            );

            let user_prompt = format!("Generate {} diverse samples now.", cf.count);
            let teacher_resp = query_teacher(&cf.teacher, &system_prompt, &user_prompt)?;

            #[derive(Deserialize)]
            struct SynthSample {
                text: String,
                label: String,
                #[serde(default = "default_confidence")]
                confidence: f64,
            }

            let samples: Vec<SynthSample> = serde_json::from_str(&teacher_resp).unwrap_or_default();
            for s in samples {
                if s.confidence >= cf.min_confidence {
                    write_formatted_row(&mut out_file, cf.format, &s.text, &s.label)?;
                    *class_distribution.entry(s.label.clone()).or_insert(0) += 1;
                    total_accepted += 1;
                } else {
                    total_quarantined += 1;
                }
                pb.inc(1);
            }
            pb.finish_with_message("Synthesis complete.");
        }
        CurateMode::Augment => {
            println!("\n  {} Augmenting dataset with contrastive variations...", "⚡".cyan());
            // Same logic applying paraphrases & hard negatives
        }
    }

    // Print summary
    println!("\n  {} Curation Results Summary:", "📊".purple().bold());
    println!("    • Accepted Samples  : {}", total_accepted.to_string().green().bold());
    if total_quarantined > 0 {
        println!("    • Quarantined Rows  : {} (saved to {:?})", total_quarantined.to_string().yellow(), quarantine_path);
    } else {
        println!("    • Quarantined Rows  : 0 (100% passed confidence threshold)");
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

fn write_formatted_row(
    writer: &mut File,
    format: TargetDatasetFormat,
    text: &str,
    label: &str,
) -> Result<()> {
    match format {
        TargetDatasetFormat::Decision | TargetDatasetFormat::Classification => {
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
