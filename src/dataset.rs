use anyhow::{Context, Result, bail};
use rand::SeedableRng;
use rand::seq::SliceRandom;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::families::traits::DecisionSample;

#[derive(Debug, Clone)]
pub struct RawSample {
    pub text_a: String,
    pub text_b: Option<String>,
    pub label: String,
}

#[derive(Debug, Clone)]
pub struct Dataset;

#[derive(Debug, Clone)]
pub struct DecisionDataset;

impl DecisionDataset {
    /// Load decision samples, automatically detecting format (DecisionSample, question schema, or classification)
    pub fn load_samples(path: &Path, schema_path: Option<&Path>) -> Result<Vec<DecisionSample>> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read dataset from {}", path.display()))?;

        let mut raw_lines = Vec::new();
        for line in content.lines() {
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                raw_lines.push(trimmed);
            }
        }

        if raw_lines.is_empty() {
            bail!("Dataset file {} is empty", path.display());
        }

        if let Ok(first_val) = serde_json::from_str::<Value>(raw_lines[0]) {
            if first_val.get("options").is_some() && first_val.get("target_index").is_some() {
                let mut samples = Vec::new();
                for (idx, line) in raw_lines.iter().enumerate() {
                    let mut val: Value = serde_json::from_str(line)
                        .with_context(|| format!("Failed to parse JSON on line {}", idx + 1))?;
                    if val.get("state").is_none() {
                        if let Some(txt) = val.get("text").cloned() {
                            val["state"] = txt;
                        }
                    }
                    let s: DecisionSample = serde_json::from_value(val)
                        .with_context(|| format!("Failed to parse DecisionSample on line {}", idx + 1))?;
                    samples.push(s);
                }
                return Ok(samples);
            } else if first_val.get("questions").is_some() && first_val.get("gold_label").is_some() {
                let mut samples = Vec::new();
                for (idx, line) in raw_lines.iter().enumerate() {
                    let val: Value = serde_json::from_str(line)
                        .with_context(|| format!("Failed to parse JSON on line {}", idx + 1))?;
                    let state = val
                        .get("state")
                        .cloned()
                        .or_else(|| val.get("text").cloned())
                        .unwrap_or(Value::String(String::new()));
                    let gold = val
                        .get("gold_label")
                        .and_then(|v| v.as_str())
                        .unwrap_or("false");
                    let q_str = val
                        .get("questions")
                        .map(|q| q.to_string())
                        .unwrap_or_default();
                    let (question, mut options) = crate::curate::extract_schema_metadata(&q_str);
                    if options.is_empty() {
                        options = vec!["false".to_string(), "true".to_string()];
                    }
                    let target_index = options
                        .iter()
                        .position(|o| o.trim().eq_ignore_ascii_case(gold.trim()))
                        .unwrap_or(0);
                    samples.push(DecisionSample {
                        state,
                        question,
                        options,
                        target_index,
                    });
                }
                return Ok(samples);
            }
        }

        let candidate_schema = schema_path
            .map(|p| p.to_path_buf())
            .or_else(|| {
                let parent = path.parent().unwrap_or_else(|| Path::new("."));
                let adjacent = parent.join("questions_schema.json");
                if adjacent.exists() {
                    Some(adjacent)
                } else {
                    None
                }
            });

        if let Some(schema_file) = candidate_schema {
            if schema_file.exists() {
                if let Ok(schema_str) = std::fs::read_to_string(&schema_file) {
                    if let Ok(schema_val) = serde_json::from_str::<Value>(&schema_str) {
                        return Self::synthesize_from_schema(&raw_lines, &schema_val);
                    }
                }
            }
        }

        Self::synthesize_generic_classification(&raw_lines)
    }

    fn synthesize_from_schema(lines: &[&str], schema_val: &Value) -> Result<Vec<DecisionSample>> {
        let mut samples = Vec::new();
        let questions_obj = schema_val
            .as_object()
            .context("Schema must be a JSON object mapping question keys to definitions")?;

        for line in lines {
            let val: Value = serde_json::from_str(line)?;
            let text = val
                .get("text")
                .or_else(|| val.get("state"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let label = val.get("label").and_then(|v| v.as_str()).unwrap_or("");

            if text.is_empty() || label.is_empty() {
                continue;
            }

            for (q_name, q_def) in questions_obj {
                let q_type = q_def.get("type").and_then(|t| t.as_str()).unwrap_or("choice");

                if q_type == "choice" {
                    let instructions = q_def
                        .get("instructions")
                        .and_then(|i| i.as_str())
                        .unwrap_or(q_name.as_str());

                    let mut options = Vec::new();
                    let mut target_idx = 0;

                    if let Some(criteria) = q_def.get("criteria").and_then(|c| c.as_object()) {
                        for (idx, (crit_key, _)) in criteria.iter().enumerate() {
                            options.push(crit_key.clone());
                            if crit_key == label {
                                target_idx = idx;
                            }
                        }
                    }

                    if options.is_empty() {
                        options = vec!["option_a".to_string(), "option_b".to_string()];
                    }

                    samples.push(DecisionSample {
                        state: Value::String(text.to_string()),
                        question: instructions.to_string(),
                        options,
                        target_index: target_idx,
                    });
                } else if q_type == "noul" {
                    let statement = q_def
                        .get("statement")
                        .and_then(|s| s.as_str())
                        .unwrap_or(q_name.as_str());

                    let is_true =
                        label == "page_sre_immediate" || label == "true" || label == "yes";
                    let target_idx = if is_true { 1 } else { 0 };

                    samples.push(DecisionSample {
                        state: Value::String(text.to_string()),
                        question: statement.to_string(),
                        options: vec!["false".to_string(), "true".to_string()],
                        target_index: target_idx,
                    });
                }
            }
        }

        Ok(samples)
    }

    fn synthesize_generic_classification(lines: &[&str]) -> Result<Vec<DecisionSample>> {
        let mut parsed_records = Vec::new();
        let mut unique_labels = std::collections::BTreeSet::new();

        for line in lines {
            let val: Value = serde_json::from_str(line)?;
            let text = val
                .get("text")
                .or_else(|| val.get("sentence"))
                .or_else(|| val.get("premise"))
                .or_else(|| val.get("state"))
                .and_then(|v| v.as_str())
                .unwrap_or("");

            let label = val
                .get("label")
                .or_else(|| val.get("target"))
                .and_then(|v| v.as_str())
                .unwrap_or("");

            if !text.is_empty() && !label.is_empty() {
                parsed_records.push((text.to_string(), label.to_string()));
                unique_labels.insert(label.to_string());
            }
        }

        let options: Vec<String> = unique_labels.into_iter().collect();
        let label_to_idx: HashMap<String, usize> = options
            .iter()
            .enumerate()
            .map(|(i, l)| (l.clone(), i))
            .collect();

        let mut samples = Vec::new();
        for (text, label) in parsed_records {
            let target_index = *label_to_idx.get(&label).unwrap_or(&0);
            samples.push(DecisionSample {
                state: Value::String(text),
                question: "Which option best applies to this state?".to_string(),
                options: options.clone(),
                target_index,
            });
        }

        Ok(samples)
    }
}

impl Dataset {
    /// Load raw samples from JSONL or CSV file
    pub fn load_raw_samples(path: &Path) -> Result<Vec<RawSample>> {
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_lowercase();

        match ext.as_str() {
            "jsonl" | "json" => Self::load_jsonl(path),
            "csv" | "tsv" => Self::load_csv(path),
            _ => Self::load_jsonl(path).or_else(|_| Self::load_csv(path)),
        }
    }

    fn load_jsonl(path: &Path) -> Result<Vec<RawSample>> {
        let file = File::open(path).with_context(|| format!("Failed to open {}", path.display()))?;
        let reader = BufReader::new(file);
        let mut samples = Vec::new();

        for (line_idx, line) in reader.lines().enumerate() {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            let val: Value = serde_json::from_str(trimmed)
                .with_context(|| format!("JSON parse error on line {}", line_idx + 1))?;

            let (text_a, text_b) = Self::extract_text_fields(&val)?;
            let label = Self::extract_label_field(&val)?;

            samples.push(RawSample {
                text_a,
                text_b,
                label,
            });
        }

        if samples.is_empty() {
            bail!("No valid samples found in {}", path.display());
        }

        Ok(samples)
    }

    fn load_csv(path: &Path) -> Result<Vec<RawSample>> {
        let mut rdr = csv::ReaderBuilder::new()
            .flexible(true)
            .from_path(path)
            .with_context(|| format!("Failed to open CSV {}", path.display()))?;

        let headers = rdr.headers()?.clone();
        let mut samples = Vec::new();

        let mut text_a_col = None;
        let mut text_b_col = None;
        let mut label_col = None;

        for (idx, header) in headers.iter().enumerate() {
            let h = header.to_lowercase();
            match h.as_str() {
                "text" | "sentence" | "sentence1" | "premise" | "input" | "prompt" | "content" => {
                    if text_a_col.is_none() {
                        text_a_col = Some(idx);
                    }
                }
                "text2" | "sentence2" | "hypothesis" => {
                    text_b_col = Some(idx);
                }
                "label" | "target" | "category" | "class" | "sentiment" => {
                    label_col = Some(idx);
                }
                _ => {}
            }
        }

        let text_a_idx = text_a_col.unwrap_or(0);
        let label_idx = label_col.unwrap_or(headers.len().saturating_sub(1));

        for result in rdr.records() {
            let record = result?;
            let text_a = record.get(text_a_idx).unwrap_or("").trim().to_string();
            let text_b = text_b_col
                .and_then(|idx| record.get(idx))
                .map(|s| s.trim().to_string());
            let label = record.get(label_idx).unwrap_or("").trim().to_string();

            if !text_a.is_empty() && !label.is_empty() {
                samples.push(RawSample {
                    text_a,
                    text_b,
                    label,
                });
            }
        }

        if samples.is_empty() {
            bail!("No valid records parsed from CSV {}", path.display());
        }

        Ok(samples)
    }

    fn extract_text_fields(val: &Value) -> Result<(String, Option<String>)> {
        let obj = val.as_object().context("Expected JSON object for sample")?;

        let candidate_keys_a = [
            "text", "sentence", "sentence1", "premise", "input", "prompt", "content",
        ];
        let candidate_keys_b = ["text2", "sentence2", "hypothesis"];

        let mut text_a = None;
        for key in candidate_keys_a {
            if let Some(v) = obj.get(key).and_then(|v| v.as_str()) {
                text_a = Some(v.to_string());
                break;
            }
        }

        let mut text_b = None;
        for key in candidate_keys_b {
            if let Some(v) = obj.get(key).and_then(|v| v.as_str()) {
                text_b = Some(v.to_string());
                break;
            }
        }

        if let Some(t_a) = text_a {
            Ok((t_a, text_b))
        } else {
            bail!("Could not find text column. Tried keys: {:?}", candidate_keys_a);
        }
    }

    fn extract_label_field(val: &Value) -> Result<String> {
        let obj = val.as_object().context("Expected JSON object")?;
        let candidate_label_keys = ["label", "target", "category", "class", "sentiment", "gold_label"];

        for key in candidate_label_keys {
            if let Some(v) = obj.get(key) {
                if let Some(s) = v.as_str() {
                    return Ok(s.to_string());
                } else if let Some(i) = v.as_i64() {
                    return Ok(i.to_string());
                } else if let Some(b) = v.as_bool() {
                    return Ok(b.to_string());
                }
            }
        }

        bail!("Could not find label column. Tried keys: {:?}", candidate_label_keys);
    }

    /// Split raw samples into train and eval subsets
    pub fn split_samples(
        mut samples: Vec<RawSample>,
        eval_split: f32,
        seed: u64,
    ) -> (Vec<RawSample>, Vec<RawSample>) {
        if eval_split <= 0.0 {
            return (samples, Vec::new());
        }

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        samples.shuffle(&mut rng);

        let eval_count = ((samples.len() as f32) * eval_split).round() as usize;
        let eval_count = eval_count.min(samples.len().saturating_sub(1));

        let eval_samples = samples.split_off(samples.len() - eval_count);
        (samples, eval_samples)
    }
}
