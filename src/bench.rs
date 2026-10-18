use anyhow::{bail, Context, Result};
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::{Duration, Instant};

/// The 13 official subsets from Bespoke Labs' Public Decision Benchmark (3,880 records total)
pub const PUBLIC_SUBSETS: &[(&str, &str, &str)] = &[
    ("vitaminc-dev", "VitaminC dev", "Choice, 3"),
    ("massive-en-US", "MASSIVE 1.1 en-US", "Choice, 18"),
    ("massive-de-DE", "MASSIVE 1.1 de-DE", "Choice, 18"),
    ("boolq", "BoolQ validation", "Noul"),
    ("squad2", "SQuAD 2.0 validation", "Noul"),
    ("paws", "PAWS labeled_final test", "Noul"),
    ("multinli", "MultiNLI dev matched", "Choice, 3"),
    ("civil_comments", "Civil Comments test", "Noul"),
    ("aegis2", "Aegis AI Content Safety 2.0 test", "Noul"),
    ("helpsteer2", "HelpSteer2 validation", "Score, 5"),
    ("summeval-relevance", "SummEval relevance test", "Score, 5"),
    ("summeval-consistency", "SummEval consistency test", "Score, 5"),
    ("pubmedqa", "PubMedQA pqa_labeled train", "Choice, 3"),
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchRecord {
    #[serde(default)]
    pub id: String,
    #[serde(rename = "type", default = "default_type")]
    pub kind: String,
    pub input: BenchInput,
    pub reference: BenchReference,
}

fn default_type() -> String {
    "choice".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchInput {
    pub state: serde_json::Value,
    pub questions: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchReference {
    pub label: serde_json::Value,
    #[serde(default)]
    pub distribution: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct SystemOneRequest<'a> {
    model: &'a str,
    state: &'a serde_json::Value,
    questions: &'a serde_json::Value,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct SystemOneResponse {
    #[serde(default)]
    model: String,
    #[serde(default)]
    answers: HashMap<String, SystemOneAnswer>,
    #[serde(default)]
    usage: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct SystemOneAnswer {
    #[serde(rename = "type")]
    kind: Option<String>,
    choice: Option<String>,
    noul: Option<f64>,
    score: Option<f64>,
    confidence: Option<f64>,
    probabilities: Option<HashMap<String, f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationRow {
    pub id: String,
    pub correct: bool,
    pub top_probability: f64,
    pub elapsed_ms: f64,
    pub predicted: String,
    pub reference: String,
    #[serde(default)]
    pub brier: f64,
    #[serde(default = "default_chance")]
    pub chance: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

fn default_chance() -> f64 {
    0.5
}

/// Compute random-guess chance probability (1/K) for a benchmark record following bench.ollaya.dev
pub fn record_chance(record: &BenchRecord) -> f64 {
    let decision = record
        .input
        .questions
        .get("decision")
        .or_else(|| record.input.questions.as_object().and_then(|m| m.values().next()));

    let q_type = decision
        .and_then(|d| d.get("type"))
        .and_then(|v| v.as_str())
        .unwrap_or(record.kind.as_str());

    if q_type == "noul" {
        return 0.5;
    }

    if let Some(dec) = decision {
        if let Some(crit) = dec.get("criteria") {
            match crit {
                serde_json::Value::Object(map) if !map.is_empty() => {
                    return 1.0 / (map.len().max(2) as f64);
                }
                serde_json::Value::Array(arr) if !arr.is_empty() => {
                    return 1.0 / (arr.len().max(2) as f64);
                }
                _ => {}
            }
        }
    }
    0.5
}

/// Extract ordered candidate label keys for multiclass Brier computation
fn record_candidate_keys(record: &BenchRecord, kind: &str) -> Vec<String> {
    if kind == "noul" {
        return vec!["false".to_string(), "true".to_string()];
    }
    let decision = record
        .input
        .questions
        .get("decision")
        .or_else(|| record.input.questions.as_object().and_then(|m| m.values().next()));
    if let Some(dec) = decision {
        if let Some(crit) = dec.get("criteria") {
            match crit {
                serde_json::Value::Object(map) => {
                    return map.keys().cloned().collect();
                }
                serde_json::Value::Array(arr) => {
                    return (0..arr.len()).map(|i| i.to_string()).collect();
                }
                _ => {}
            }
        }
    }
    Vec::new()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureExportRecord {
    pub id: String,
    pub subset: String,
    pub kind: String,
    /// Flattened context + question prompt ready for direct consumption by 'trainaya curate'
    pub text: String,
    pub state: serde_json::Value,
    pub questions: serde_json::Value,
    pub gold_label: String,
    pub predicted_label: String,
    pub confidence: f64,
    /// Failure severity class: "misclassified_overconfident", "misclassified", or "boundary_low_confidence"
    pub failure_type: String,
}

/// Format a BenchRecord's state and question into a clean, self-contained text representation
pub fn format_bench_record_as_text(record: &BenchRecord) -> String {
    let mut parts = Vec::new();
    match &record.input.state {
        serde_json::Value::Object(map) => {
            for (k, v) in map {
                let val_str = match v {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                parts.push(format!("{}: {}", k, val_str));
            }
        }
        serde_json::Value::String(s) => parts.push(s.clone()),
        other => parts.push(other.to_string()),
    }

    if let Some(decision) = record
        .input
        .questions
        .get("decision")
        .or_else(|| record.input.questions.as_object().and_then(|m| m.values().next()))
    {
        if let Some(instr) = decision.get("instructions").and_then(|v| v.as_str()) {
            parts.push(format!("Question: {}", instr));
        }
    }

    parts.join(" | ")
}

/// Determine if an EvaluationRow should be mined for active learning and return its failure_type
pub fn classify_failure_row(row: &EvaluationRow) -> Option<&'static str> {
    if row.error.is_some() {
        return None;
    }
    if !row.correct {
        if row.top_probability >= 0.80 {
            Some("misclassified_overconfident")
        } else {
            Some("misclassified")
        }
    } else if row.top_probability < 0.55 {
        Some("boundary_low_confidence")
    } else {
        None
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubsetSummary {
    pub subset: String,
    pub title: String,
    #[serde(default = "default_type")]
    pub kind: String,
    pub count: usize,
    pub errors: usize,
    pub accuracy: f64,
    #[serde(default = "default_chance")]
    pub chance: f64,
    /// Chance-corrected skill score: 100 * (accuracy - chance) / (1 - chance)
    #[serde(default)]
    pub skill: f64,
    pub ece10: f64,
    #[serde(default)]
    pub brier: f64,
    pub median_ms: f64,
    pub p95_ms: f64,
    /// Per-probe (top_probability, correct, brier, chance) for exact pooled ECE and bootstrap CIs
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub probe_outcomes: Vec<(f64, bool, f64, f64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkRunResult {
    pub model: String,
    pub server_url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manifest_digest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub binary_sha256: Option<String>,
    pub total_records: usize,
    pub total_errors: usize,
    /// Chance-corrected Public Skill Score (0..100) matching bench.ollaya.dev
    #[serde(default)]
    pub skill_score: f64,
    /// 95% bootstrap confidence interval for skill_score (1,000 draws, seed 20261014)
    #[serde(default)]
    pub skill_score_ci: [f64; 2],
    pub macro_accuracy: f64,
    /// 95% bootstrap confidence interval for macro_accuracy (0.0..1.0)
    #[serde(default)]
    pub macro_accuracy_ci: [f64; 2],
    pub micro_accuracy: f64,
    pub macro_ece: f64,
    pub pooled_ece: f64,
    /// Pooled multiclass Brier score matching bench.ollaya.dev
    #[serde(default)]
    pub brier: f64,
    /// Macro accuracy breakdown by task type ("choice", "noul", "score")
    #[serde(default)]
    pub by_type: HashMap<String, f64>,
    pub median_ms: f64,
    pub p95_ms: f64,
    pub per_subset: HashMap<String, SubsetSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullBenchmarkReport {
    pub timestamp: String,
    pub evaluated_model: BenchmarkRunResult,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub baseline_model: Option<BenchmarkRunResult>,
}

pub struct BenchRunner {
    pub url: String,
    pub limit: Option<usize>,
}

impl BenchRunner {
    pub fn new(url: String, limit: Option<usize>) -> Self {
        Self { url, limit }
    }

    /// Post one request to /v1/systemone with retry on transient network errors
    pub fn evaluate_record(
        &self,
        model: &str,
        record: &BenchRecord,
    ) -> Result<EvaluationRow> {
        let payload = SystemOneRequest {
            model,
            state: &record.input.state,
            questions: &record.input.questions,
        };

        let chance = record_chance(record);
        let start = Instant::now();
        let api_url = format!("{}/v1/systemone", self.url.trim_end_matches('/'));

        let mut last_err = None;
        let mut resp_opt = None;

        for attempt in 0..3 {
            match ureq::post(&api_url)
                .set("Authorization", "Bearer local")
                .set("Content-Type", "application/json")
                .timeout(Duration::from_secs(120))
                .send_json(&payload)
            {
                Ok(resp) => {
                    match resp.into_json::<SystemOneResponse>() {
                        Ok(parsed) => {
                            resp_opt = Some(parsed);
                            break;
                        }
                        Err(e) => {
                            last_err = Some(format!("Failed to parse response: {}", e));
                        }
                    }
                }
                Err(ureq::Error::Status(code, resp)) => {
                    let err_body = resp.into_string().unwrap_or_default();
                    last_err = Some(format!("HTTP {}: {}", code, err_body));
                    if code == 429 || code >= 500 {
                        std::thread::sleep(Duration::from_millis(500 * (attempt + 1)));
                        continue;
                    }
                    break;
                }
                Err(ureq::Error::Transport(e)) => {
                    last_err = Some(format!("Transport error: {}", e));
                    std::thread::sleep(Duration::from_millis(500 * (attempt + 1)));
                    continue;
                }
            }
        }

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        let resp = match resp_opt {
            Some(r) => r,
            None => {
                let err_msg = last_err.unwrap_or_else(|| "Unknown error".to_string());
                return Ok(EvaluationRow {
                    id: record.id.clone(),
                    correct: false,
                    top_probability: 0.0,
                    elapsed_ms,
                    predicted: "ERROR".to_string(),
                    reference: format!("{:?}", record.reference.label),
                    brier: 1.0,
                    chance,
                    error: Some(err_msg),
                });
            }
        };

        // Get the decision answer
        let answer = resp
            .answers
            .get("decision")
            .or_else(|| resp.answers.values().next());

        let answer = match answer {
            Some(a) => a,
            None => {
                return Ok(EvaluationRow {
                    id: record.id.clone(),
                    correct: false,
                    top_probability: 0.0,
                    elapsed_ms,
                    predicted: "NO_ANSWER".to_string(),
                    reference: format!("{:?}", record.reference.label),
                    brier: 1.0,
                    chance,
                    error: Some("Server returned empty answers map".to_string()),
                });
            }
        };

        // Determine question kind
        let kind = answer
            .kind
            .as_deref()
            .unwrap_or_else(|| record.kind.as_str());

        let (predicted, top_prob, correct, brier) = match kind {
            "noul" => {
                let gold_bool = match &record.reference.label {
                    serde_json::Value::Bool(b) => *b,
                    serde_json::Value::String(s) => s.to_lowercase() == "true" || s.to_lowercase() == "yes",
                    serde_json::Value::Number(n) => n.as_i64().unwrap_or(0) == 1,
                    _ => false,
                };

                let p = answer.noul.unwrap_or(0.5).clamp(0.0, 1.0);
                let pred_bool = p >= 0.5;
                let top_p = if pred_bool { p } else { 1.0 - p };
                let is_correct = pred_bool == gold_bool;
                let gold_t = if gold_bool { 1.0 } else { 0.0 };
                // Multiclass Brier over {"false": 1 - p, "true": p}
                let brier_val = 2.0 * (p - gold_t).powi(2);
                (pred_bool.to_string(), top_p, is_correct, brier_val)
            }
            "score" => {
                let gold_str = match &record.reference.label {
                    serde_json::Value::Number(n) => n.to_string(),
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                };

                let best_key = if let Some(probs) = &answer.probabilities {
                    probs
                        .iter()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(k, _)| k.clone())
                        .unwrap_or_else(|| "0".to_string())
                } else if let Some(c) = &answer.choice {
                    c.clone()
                } else if let Some(s) = answer.score {
                    s.round().to_string()
                } else {
                    "0".to_string()
                };

                let top_p = answer
                    .probabilities
                    .as_ref()
                    .and_then(|p| p.get(&best_key).copied())
                    .or(answer.confidence)
                    .unwrap_or(0.5)
                    .clamp(0.0, 1.0);

                let is_correct = best_key.trim() == gold_str.trim();
                let cand_keys = record_candidate_keys(record, "score");
                let brier_val = if let Some(probs) = &answer.probabilities {
                    let keys: Vec<String> = if cand_keys.is_empty() {
                        probs.keys().cloned().collect()
                    } else {
                        cand_keys
                    };
                    keys.iter()
                        .map(|k| {
                            let pk = probs.get(k).copied().unwrap_or(0.0);
                            let yk = if k.trim() == gold_str.trim() { 1.0 } else { 0.0 };
                            (pk - yk).powi(2)
                        })
                        .sum()
                } else {
                    let y = if is_correct { 1.0 } else { 0.0 };
                    (top_p - y).powi(2) + ((1.0 - top_p) - (1.0 - y)).powi(2)
                };
                (best_key, top_p, is_correct, brier_val)
            }
            _ => {
                // choice
                let gold_str = match &record.reference.label {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string().trim_matches('"').to_string(),
                };

                let pred_choice = if let Some(c) = &answer.choice {
                    c.clone()
                } else if let Some(probs) = &answer.probabilities {
                    probs
                        .iter()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(k, _)| k.clone())
                        .unwrap_or_default()
                } else {
                    "".to_string()
                };

                let top_p = answer
                    .probabilities
                    .as_ref()
                    .and_then(|p| p.get(&pred_choice).copied())
                    .or(answer.confidence)
                    .unwrap_or(0.5)
                    .clamp(0.0, 1.0);

                let is_correct = pred_choice.trim().eq_ignore_ascii_case(gold_str.trim());
                let cand_keys = record_candidate_keys(record, "choice");
                let brier_val = if let Some(probs) = &answer.probabilities {
                    let keys: Vec<String> = if cand_keys.is_empty() {
                        probs.keys().cloned().collect()
                    } else {
                        cand_keys
                    };
                    keys.iter()
                        .map(|k| {
                            let pk = probs.get(k).copied().unwrap_or(0.0);
                            let yk = if k.trim().eq_ignore_ascii_case(gold_str.trim()) {
                                1.0
                            } else {
                                0.0
                            };
                            (pk - yk).powi(2)
                        })
                        .sum()
                } else {
                    let y = if is_correct { 1.0 } else { 0.0 };
                    (top_p - y).powi(2) + ((1.0 - top_p) - (1.0 - y)).powi(2)
                };
                (pred_choice, top_p, is_correct, brier_val)
            }
        };

        let clean_ref = match &record.reference.label {
            serde_json::Value::String(s) => s.clone(),
            other => other.to_string().trim_matches('"').to_string(),
        };

        Ok(EvaluationRow {
            id: record.id.clone(),
            correct,
            top_probability: top_prob.clamp(0.0, 1.0),
            elapsed_ms,
            predicted,
            reference: clean_ref,
            brier,
            chance,
            error: None,
        })
    }

    /// Run evaluation for one model on a set of records
    pub fn run_subset(
        &self,
        model: &str,
        subset_name: &str,
        subset_title: &str,
        records: &[BenchRecord],
    ) -> Result<SubsetSummary> {
        let (summary, _) = self.run_subset_with_failures_progress(model, subset_name, subset_title, records, None)?;
        Ok(summary)
    }

    /// Run evaluation for one model on a set of records and extract failure/boundary probes
    pub fn run_subset_with_failures(
        &self,
        model: &str,
        subset_name: &str,
        subset_title: &str,
        records: &[BenchRecord],
    ) -> Result<(SubsetSummary, Vec<FailureExportRecord>)> {
        self.run_subset_with_failures_progress(model, subset_name, subset_title, records, None)
    }

    /// Run evaluation with optional live per-probe ProgressBar updates
    pub fn run_subset_with_failures_progress(
        &self,
        model: &str,
        subset_name: &str,
        subset_title: &str,
        records: &[BenchRecord],
        pb: Option<&ProgressBar>,
    ) -> Result<(SubsetSummary, Vec<FailureExportRecord>)> {
        let count_to_eval = if let Some(limit) = self.limit {
            records.len().min(limit)
        } else {
            records.len()
        };

        let records = &records[..count_to_eval];
        if records.is_empty() {
            return Ok((
                SubsetSummary {
                    subset: subset_name.to_string(),
                    title: subset_title.to_string(),
                    kind: "choice".to_string(),
                    count: 0,
                    errors: 0,
                    accuracy: 0.0,
                    chance: 0.5,
                    skill: 0.0,
                    ece10: 0.0,
                    brier: 0.0,
                    median_ms: 0.0,
                    p95_ms: 0.0,
                    probe_outcomes: Vec::new(),
                },
                Vec::new(),
            ));
        }

        let subset_kind = records
            .first()
            .map(|r| r.kind.clone())
            .unwrap_or_else(|| "choice".to_string());
        let total_sub = records.len();
        let mut rows = Vec::with_capacity(total_sub);
        let mut failures = Vec::new();
        let mut latencies = Vec::with_capacity(total_sub);
        let mut ece_pairs = Vec::with_capacity(total_sub);
        let mut probe_outcomes = Vec::with_capacity(total_sub);
        let mut errors = 0;
        let mut correct_count = 0;
        let mut brier_sum = 0.0;
        let mut chance_sum = 0.0;

        for (idx, record) in records.iter().enumerate() {
            let row = self.evaluate_record(model, record)?;
            if row.error.is_some() {
                errors += 1;
            } else {
                if row.correct {
                    correct_count += 1;
                }
                if let Some(failure_type) = classify_failure_row(&row) {
                    failures.push(FailureExportRecord {
                        id: record.id.clone(),
                        subset: subset_name.to_string(),
                        kind: record.kind.clone(),
                        text: format_bench_record_as_text(record),
                        state: record.input.state.clone(),
                        questions: record.input.questions.clone(),
                        gold_label: row.reference.clone(),
                        predicted_label: row.predicted.clone(),
                        confidence: row.top_probability,
                        failure_type: failure_type.to_string(),
                    });
                }
                latencies.push(row.elapsed_ms);
                ece_pairs.push((row.top_probability, row.correct));
                probe_outcomes.push((row.top_probability, row.correct, row.brier, row.chance));
                brier_sum += row.brier;
                chance_sum += row.chance;
            }
            rows.push(row);

            if let Some(bar) = pb {
                let done = idx + 1;
                let live_acc = (correct_count as f64 / done as f64) * 100.0;
                bar.set_message(format!(
                    "{} [{}/{} | acc: {:.1}%]",
                    subset_name, done, total_sub, live_acc
                ));
                bar.inc(1);
            }
        }

        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_ms = if latencies.is_empty() {
            0.0
        } else {
            latencies[latencies.len() / 2]
        };

        let p95_ms = if latencies.is_empty() {
            0.0
        } else {
            let idx = ((latencies.len() as f64) * 0.95).ceil() as usize;
            latencies[idx.saturating_sub(1).min(latencies.len() - 1)]
        };

        let valid_n = probe_outcomes.len();
        let accuracy = if valid_n == 0 {
            0.0
        } else {
            correct_count as f64 / valid_n as f64
        };
        let chance = if valid_n == 0 {
            record_chance(&records[0])
        } else {
            chance_sum / valid_n as f64
        };
        let skill = if (1.0 - chance).abs() < 1e-9 {
            0.0
        } else {
            100.0 * (accuracy - chance) / (1.0 - chance)
        };
        let brier = if valid_n == 0 {
            0.0
        } else {
            brier_sum / valid_n as f64
        };

        let ece10 = compute_ece(&ece_pairs, 10);

        Ok((
            SubsetSummary {
                subset: subset_name.to_string(),
                title: subset_title.to_string(),
                kind: subset_kind,
                count: rows.len(),
                errors,
                accuracy,
                chance,
                skill,
                ece10,
                brier,
                median_ms,
                p95_ms,
                probe_outcomes,
            },
            failures,
        ))
    }
}

/// Compute 10-bin Expected Calibration Error
pub fn compute_ece(pairs: &[(f64, bool)], bins: usize) -> f64 {
    if pairs.is_empty() {
        return 0.0;
    }
    let total = pairs.len() as f64;
    let mut err = 0.0;
    let b_count = bins as f64;

    for b in 0..bins {
        let low = b as f64 / b_count;
        let high = (b + 1) as f64 / b_count;
        let sel: Vec<_> = pairs
            .iter()
            .filter(|(p, _)| {
                if b == 0 && *p == 0.0 {
                    true
                } else {
                    *p > low && *p <= high
                }
            })
            .collect();

        if !sel.is_empty() {
            let n = sel.len() as f64;
            let sum_c: f64 = sel.iter().map(|(_, c)| if *c { 1.0 } else { 0.0 }).sum();
            let sum_p: f64 = sel.iter().map(|(p, _)| *p).sum();
            let acc = sum_c / n;
            let conf = sum_p / n;
            err += (n / total) * (acc - conf).abs();
        }
    }
    err
}

/// Load records from a JSONL file
pub fn load_jsonl(path: &Path) -> Result<Vec<BenchRecord>> {
    let file = File::open(path).with_context(|| format!("Opening {:?}", path))?;
    let reader = BufReader::new(file);
    let mut records = Vec::new();

    for (idx, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let rec: BenchRecord = serde_json::from_str(&line)
            .with_context(|| format!("Parsing JSONL line {} in {:?}", idx + 1, path))?;
        records.push(rec);
    }
    Ok(records)
}

/// Return the global benchmark cache directory (~/.cache/trainaya/benchmarks/public)
pub fn get_benchmark_cache_dir() -> std::path::PathBuf {
    if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
        if !xdg.trim().is_empty() {
            return std::path::PathBuf::from(xdg)
                .join("trainaya")
                .join("benchmarks")
                .join("public");
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        if !home.trim().is_empty() {
            return std::path::PathBuf::from(home)
                .join(".cache")
                .join("trainaya")
                .join("benchmarks")
                .join("public");
        }
    }
    std::path::PathBuf::from(".cache/trainaya/benchmarks/public")
}

/// Fetch raw JSON rows from Hugging Face Datasets Server REST API in chunks of 100 with automatic retry on rate limits
fn fetch_hf_rows(dataset: &str, config: &str, split: &str, target_count: usize) -> Result<Vec<serde_json::Value>> {
    let mut rows = Vec::with_capacity(target_count);
    let mut offset = 0usize;

    while rows.len() < target_count {
        let length = (target_count - rows.len()).min(100);
        let url = format!(
            "https://datasets-server.huggingface.co/rows?dataset={}&config={}&split={}&offset={}&length={}",
            dataset, config, split, offset, length
        );

        let mut last_err = None;
        let mut resp_opt = None;
        for attempt in 1..=5 {
            match ureq::get(&url)
                .timeout(Duration::from_secs(20))
                .call()
            {
                Ok(r) => {
                    resp_opt = Some(r);
                    break;
                }
                Err(ureq::Error::Status(429, _))
                | Err(ureq::Error::Status(502, _))
                | Err(ureq::Error::Status(503, _))
                | Err(ureq::Error::Status(504, _)) => {
                    last_err = Some(format!("HTTP rate-limit/busy on {}", dataset));
                    std::thread::sleep(Duration::from_millis(1800 * attempt));
                }
                Err(e) => {
                    last_err = Some(e.to_string());
                    if attempt < 3 {
                        std::thread::sleep(Duration::from_millis(800 * attempt));
                    } else {
                        break;
                    }
                }
            }
        }

        let resp = match resp_opt {
            Some(r) => r,
            None => bail!(
                "Fetching HF dataset {} ({}/{}): {}",
                dataset,
                config,
                split,
                last_err.unwrap_or_else(|| "unknown error".to_string())
            ),
        };

        let body: serde_json::Value = resp.into_json()?;
        let arr = match body.get("rows").and_then(|v| v.as_array()) {
            Some(a) if !a.is_empty() => a,
            _ => break,
        };

        for item in arr {
            if let Some(r) = item.get("row") {
                rows.push(r.clone());
                if rows.len() >= target_count {
                    break;
                }
            }
        }

        if arr.len() < length {
            break;
        }
        offset += arr.len();
        std::thread::sleep(Duration::from_millis(120));
    }

    Ok(rows)
}

/// Return the canonical bespokelabsai/nimble question definition for each of the 13 public subsets
pub fn nimble_question_template(subset_name: &str) -> Option<serde_json::Value> {
    match subset_name {
        "vitaminc-dev" => Some(serde_json::json!({
            "decision": {
                "type": "choice",
                "instructions": "Given the evidence and the claim, decide whether the evidence supports the claim, refutes it, or does not provide enough information.",
                "criteria": {
                    "SUPPORTS": "The evidence directly supports the claim.",
                    "REFUTES": "The evidence directly contradicts or refutes the claim.",
                    "NOT ENOUGH INFO": "The evidence does not contain enough information to verify or refute the claim."
                }
            }
        })),
        "massive-en-US" | "massive-de-DE" => Some(serde_json::json!({
            "decision": {
                "type": "choice",
                "instructions": "Classify the user utterance into one of the 18 domain categories.",
                "criteria": {
                    "alarm": "Setting, modifying, querying, or removing alarms.",
                    "audio": "Adjusting volume, muting, or audio output settings.",
                    "calendar": "Creating, querying, or deleting calendar events and reminders.",
                    "cooking": "Recipes, cooking instructions, food preparation questions.",
                    "datetime": "Asking for the current time, date, or timezone conversions.",
                    "email": "Reading, sending, replying to, or searching emails and contacts.",
                    "general": "Chitchat, greetings, jokes, or control commands directed at the assistant.",
                    "iot": "Controlling smart home devices: lights, plugs, thermostat, vacuum, coffee maker.",
                    "lists": "Creating, viewing, or removing items on shopping or to-do lists.",
                    "music": "Querying music metadata, preferences, or playback state.",
                    "news": "Requesting news headlines or updates on topics.",
                    "play": "Starting playback of music, radio, podcasts, audiobooks, or games.",
                    "qa": "Factoid, math, definition, currency, or stock queries.",
                    "recommendation": "Asking for recommendations of movies, restaurants, events, or local places.",
                    "social": "Posting to or querying social media accounts.",
                    "takeaway": "Ordering food delivery or inquiring about takeaway orders.",
                    "transport": "Traffic, train/bus schedules, directions, or booking a taxi/ride.",
                    "weather": "Weather forecasts and current conditions."
                }
            }
        })),
        "boolq" => Some(serde_json::json!({
            "decision": {
                "type": "noul",
                "instructions": "Given the passage, answer the yes/no question. Return true for yes and false for no.",
                "criteria": {
                    "true": "The passage supports answering yes to the question.",
                    "false": "The passage supports answering no to the question."
                }
            }
        })),
        "squad2" => Some(serde_json::json!({
            "decision": {
                "type": "noul",
                "instructions": "Given the paragraph and the question, decide whether the paragraph contains enough information to answer the question.",
                "criteria": {
                    "true": "The question can be answered directly from the paragraph.",
                    "false": "The paragraph does not contain the answer (unanswerable)."
                }
            }
        })),
        "paws" => Some(serde_json::json!({
            "decision": {
                "type": "noul",
                "instructions": "Decide whether sentence_1 and sentence_2 are paraphrases with the same meaning.",
                "criteria": {
                    "true": "The two sentences have the same meaning (paraphrases).",
                    "false": "The two sentences have different meanings."
                }
            }
        })),
        "multinli" => Some(serde_json::json!({
            "decision": {
                "type": "choice",
                "instructions": "Given the premise and the hypothesis, classify the natural language inference relationship.",
                "criteria": {
                    "entailment": "If the premise is true, the hypothesis must also be true.",
                    "neutral": "The hypothesis might be true or false given the premise.",
                    "contradiction": "If the premise is true, the hypothesis is definitely false."
                }
            }
        })),
        "civil_comments" => Some(serde_json::json!({
            "decision": {
                "type": "noul",
                "instructions": "Decide whether the comment is toxic (rude, disrespectful, or unreasonable enough to make someone leave a discussion).",
                "criteria": {
                    "true": "The comment is toxic.",
                    "false": "The comment is non-toxic."
                }
            }
        })),
        "aegis2" => Some(serde_json::json!({
            "decision": {
                "type": "noul",
                "instructions": "Decide whether the user message is unsafe (requests or contains harmful, illegal, or policy-violating content).",
                "criteria": {
                    "true": "The user message is unsafe.",
                    "false": "The user message is safe."
                }
            }
        })),
        "helpsteer2" => Some(serde_json::json!({
            "decision": {
                "type": "score",
                "instructions": "Rate the helpfulness of the assistant response to the user prompt on the 5-level scale from 0 (not helpful) to 4 (completely helpful).",
                "criteria": [
                    "The response is not useful or helpful at all.",
                    "The response is borderline unhelpful and mostly misses what the user asked.",
                    "The response is partially helpful but misses notable aspects of the prompt.",
                    "The response is helpful and addresses the prompt well with minor room for improvement.",
                    "The response is extremely helpful and completely fulfills the user prompt."
                ]
            }
        })),
        "summeval-relevance" => Some(serde_json::json!({
            "decision": {
                "type": "score",
                "instructions": "Rate the relevance of the summary to the source article on the 5-level scale from 0 to 4.",
                "criteria": [
                    "Irrelevant: the summary contains mostly redundant or off-topic information.",
                    "Low relevance: the summary misses most of the important points of the article.",
                    "Moderate relevance: the summary captures some key points along with excess detail.",
                    "Good relevance: the summary captures most main points with little excess.",
                    "High relevance: the summary concisely captures only the most important information."
                ]
            }
        })),
        "summeval-consistency" => Some(serde_json::json!({
            "decision": {
                "type": "score",
                "instructions": "Rate the factual consistency of the summary with the source article on the 5-level scale from 0 to 4.",
                "criteria": [
                    "Severely inconsistent: multiple fabricated facts or direct contradictions of the article.",
                    "Mostly inconsistent: notable unsupported claims or distortions.",
                    "Partially consistent: mixes supported facts with a minor unsupported detail.",
                    "Mostly consistent: faithful to the article with at most trivial wording drift.",
                    "Fully consistent: every statement in the summary is entailed by the article."
                ]
            }
        })),
        "pubmedqa" => Some(serde_json::json!({
            "decision": {
                "type": "choice",
                "instructions": "Given the biomedical research question and the abstract context (without the concluding summary), answer the question with yes, no, or maybe.",
                "criteria": {
                    "yes": "The abstract findings support an affirmative answer to the question.",
                    "no": "The abstract findings support a negative answer to the question.",
                    "maybe": "The abstract findings are mixed, conditional, or inconclusive."
                }
            }
        })),
        _ => None,
    }
}

/// Normalize a BenchRecord to match bespokelabsai/nimble's exact state key ordering and criteria definitions
pub fn normalize_nimble_record(subset_name: &str, rec: &mut BenchRecord) {
    if let Some(q_tmpl) = nimble_question_template(subset_name) {
        rec.input.questions = q_tmpl;
    }

    match subset_name {
        "vitaminc-dev" => {
            if let Some(obj) = rec.input.state.as_object() {
                let ev = obj.get("evidence").cloned().unwrap_or(serde_json::Value::Null);
                let cl = obj.get("claim").cloned().unwrap_or(serde_json::Value::Null);
                rec.input.state = serde_json::json!({ "evidence": ev, "claim": cl });
            }
        }
        "massive-en-US" | "massive-de-DE" => {
            if let Some(obj) = rec.input.state.as_object() {
                let utt = obj.get("utterance").cloned().unwrap_or(serde_json::Value::Null);
                let loc = obj
                    .get("locale")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!(if subset_name == "massive-de-DE" { "de-DE" } else { "en-US" }));
                rec.input.state = serde_json::json!({ "utterance": utt, "locale": loc });
            }
        }
        "boolq" => {
            if let Some(obj) = rec.input.state.as_object() {
                let p = obj.get("passage").cloned().unwrap_or(serde_json::Value::Null);
                let q = obj.get("question").cloned().unwrap_or(serde_json::Value::Null);
                rec.input.state = serde_json::json!({ "passage": p, "question": q });
            }
        }
        "squad2" => {
            if let Some(obj) = rec.input.state.as_object() {
                let p = obj.get("paragraph").cloned().unwrap_or(serde_json::Value::Null);
                let q = obj.get("question").cloned().unwrap_or(serde_json::Value::Null);
                rec.input.state = serde_json::json!({ "paragraph": p, "question": q });
            }
        }
        "paws" => {
            if let Some(obj) = rec.input.state.as_object() {
                let s1 = obj.get("sentence_1").cloned().unwrap_or(serde_json::Value::Null);
                let s2 = obj.get("sentence_2").cloned().unwrap_or(serde_json::Value::Null);
                rec.input.state = serde_json::json!({ "sentence_1": s1, "sentence_2": s2 });
            }
        }
        "multinli" => {
            if let Some(obj) = rec.input.state.as_object() {
                let p = obj.get("premise").cloned().unwrap_or(serde_json::Value::Null);
                let h = obj.get("hypothesis").cloned().unwrap_or(serde_json::Value::Null);
                rec.input.state = serde_json::json!({ "premise": p, "hypothesis": h });
            }
        }
        "civil_comments" => {
            if let Some(obj) = rec.input.state.as_object() {
                if let Some(c) = obj.get("comment").and_then(|v| v.as_str()) {
                    rec.input.state = serde_json::Value::String(c.to_string());
                }
            }
        }
        "aegis2" => {
            if let Some(obj) = rec.input.state.as_object() {
                let m = obj.get("user_message").cloned().unwrap_or(serde_json::Value::Null);
                rec.input.state = serde_json::json!({ "user_message": m });
            }
        }
        "helpsteer2" => {
            if let Some(obj) = rec.input.state.as_object() {
                let p = obj.get("prompt").cloned().unwrap_or(serde_json::Value::Null);
                let r = obj.get("response").cloned().unwrap_or(serde_json::Value::Null);
                rec.input.state = serde_json::json!({ "prompt": p, "response": r });
            }
        }
        "summeval-relevance" | "summeval-consistency" => {
            if let Some(obj) = rec.input.state.as_object() {
                let a = obj.get("article").cloned().unwrap_or(serde_json::Value::Null);
                let s = obj.get("summary").cloned().unwrap_or(serde_json::Value::Null);
                rec.input.state = serde_json::json!({ "article": a, "summary": s });
            }
        }
        "pubmedqa" => {
            if let Some(obj) = rec.input.state.as_object() {
                let q = obj.get("question").cloned().unwrap_or(serde_json::Value::Null);
                let c = obj.get("abstract_context").cloned().unwrap_or(serde_json::Value::Null);
                rec.input.state = serde_json::json!({ "question": q, "abstract_context": c });
            }
        }
        _ => {}
    }
}

/// Download and build one of the 13 public benchmark subsets from Hugging Face into ~/.cache/trainaya/benchmarks/public/<subset>/all.jsonl
pub fn fetch_and_cache_public_subset(
    subset_name: &str,
    cache_root: &Path,
    question_template: &serde_json::Value,
) -> Result<Vec<BenchRecord>> {
    let mut records = Vec::new();
    let q_tmpl = nimble_question_template(subset_name).unwrap_or_else(|| question_template.clone());

    match subset_name {
        "vitaminc-dev" => {
            let raw = fetch_hf_rows("tals/vitaminc", "default", "validation", 599)?;
            for (idx, r) in raw.into_iter().enumerate() {
                let claim = r.get("claim").and_then(|v| v.as_str()).unwrap_or_default();
                let evidence = r.get("evidence").and_then(|v| v.as_str()).unwrap_or_default();
                let label = r.get("label").and_then(|v| v.as_str()).unwrap_or("NOT ENOUGH INFO");
                records.push(BenchRecord {
                    id: format!("vitaminc-dev-{}", idx + 1),
                    kind: "choice".to_string(),
                    input: BenchInput {
                        state: serde_json::json!({ "evidence": evidence, "claim": claim }),
                        questions: q_tmpl.clone(),
                    },
                    reference: BenchReference {
                        label: serde_json::json!(label),
                        distribution: None,
                    },
                });
            }
        }
        "massive-en-US" => {
            let raw = fetch_hf_rows("mteb/amazon_massive_intent", "en", "test", 350)?;
            for (idx, r) in raw.into_iter().enumerate() {
                let text = r.get("text").and_then(|v| v.as_str()).unwrap_or_default();
                let full_label = r.get("label").and_then(|v| v.as_str()).unwrap_or("general");
                let domain = full_label.split('_').next().unwrap_or("general");
                records.push(BenchRecord {
                    id: format!("massive-en-{}", idx + 1),
                    kind: "choice".to_string(),
                    input: BenchInput {
                        state: serde_json::json!({ "utterance": text, "locale": "en-US" }),
                        questions: q_tmpl.clone(),
                    },
                    reference: BenchReference {
                        label: serde_json::json!(domain),
                        distribution: None,
                    },
                });
            }
        }
        "massive-de-DE" => {
            let raw = fetch_hf_rows("mteb/amazon_massive_intent", "de", "test", 350)?;
            for (idx, r) in raw.into_iter().enumerate() {
                let text = r.get("text").and_then(|v| v.as_str()).unwrap_or_default();
                let full_label = r.get("label").and_then(|v| v.as_str()).unwrap_or("general");
                let domain = full_label.split('_').next().unwrap_or("general");
                records.push(BenchRecord {
                    id: format!("massive-de-{}", idx + 1),
                    kind: "choice".to_string(),
                    input: BenchInput {
                        state: serde_json::json!({ "utterance": text, "locale": "de-DE" }),
                        questions: q_tmpl.clone(),
                    },
                    reference: BenchReference {
                        label: serde_json::json!(domain),
                        distribution: None,
                    },
                });
            }
        }
        "boolq" => {
            let raw = fetch_hf_rows("google/boolq", "default", "validation", 300)?;
            for (idx, r) in raw.into_iter().enumerate() {
                let passage = r.get("passage").and_then(|v| v.as_str()).unwrap_or_default();
                let question = r.get("question").and_then(|v| v.as_str()).unwrap_or_default();
                let ans = r.get("answer").and_then(|v| v.as_bool()).unwrap_or(false);
                records.push(BenchRecord {
                    id: format!("boolq-{}", idx + 1),
                    kind: "noul".to_string(),
                    input: BenchInput {
                        state: serde_json::json!({ "passage": passage, "question": question }),
                        questions: q_tmpl.clone(),
                    },
                    reference: BenchReference {
                        label: serde_json::json!(ans),
                        distribution: None,
                    },
                });
            }
        }
        "squad2" => {
            let raw = fetch_hf_rows("rajpurkar/squad_v2", "squad_v2", "validation", 299)?;
            for (idx, r) in raw.into_iter().enumerate() {
                let paragraph = r.get("context").and_then(|v| v.as_str()).unwrap_or_default();
                let question = r.get("question").and_then(|v| v.as_str()).unwrap_or_default();
                let answerable = r
                    .get("answers")
                    .and_then(|a| a.get("text"))
                    .and_then(|t| t.as_array())
                    .map(|arr| !arr.is_empty())
                    .unwrap_or(false);
                records.push(BenchRecord {
                    id: format!("squad2-{}", idx + 1),
                    kind: "noul".to_string(),
                    input: BenchInput {
                        state: serde_json::json!({ "paragraph": paragraph, "question": question }),
                        questions: q_tmpl.clone(),
                    },
                    reference: BenchReference {
                        label: serde_json::json!(answerable),
                        distribution: None,
                    },
                });
            }
        }
        "paws" => {
            let raw = fetch_hf_rows("google-research-datasets/paws", "labeled_final", "test", 250)?;
            for (idx, r) in raw.into_iter().enumerate() {
                let s1 = r.get("sentence1").and_then(|v| v.as_str()).unwrap_or_default();
                let s2 = r.get("sentence2").and_then(|v| v.as_str()).unwrap_or_default();
                let is_para = r.get("label").and_then(|v| v.as_i64()).unwrap_or(0) == 1;
                records.push(BenchRecord {
                    id: format!("paws-{}", idx + 1),
                    kind: "noul".to_string(),
                    input: BenchInput {
                        state: serde_json::json!({ "sentence_1": s1, "sentence_2": s2 }),
                        questions: q_tmpl.clone(),
                    },
                    reference: BenchReference {
                        label: serde_json::json!(is_para),
                        distribution: None,
                    },
                });
            }
        }
        "multinli" => {
            let raw = fetch_hf_rows("nyu-mll/multi_nli", "default", "validation_matched", 299)?;
            for (idx, r) in raw.into_iter().enumerate() {
                let premise = r.get("premise").and_then(|v| v.as_str()).unwrap_or_default();
                let hypothesis = r.get("hypothesis").and_then(|v| v.as_str()).unwrap_or_default();
                let lbl_int = r.get("label").and_then(|v| v.as_i64()).unwrap_or(1);
                let lbl_str = match lbl_int {
                    0 => "entailment",
                    2 => "contradiction",
                    _ => "neutral",
                };
                records.push(BenchRecord {
                    id: format!("multinli-{}", idx + 1),
                    kind: "choice".to_string(),
                    input: BenchInput {
                        state: serde_json::json!({ "premise": premise, "hypothesis": hypothesis }),
                        questions: q_tmpl.clone(),
                    },
                    reference: BenchReference {
                        label: serde_json::json!(lbl_str),
                        distribution: None,
                    },
                });
            }
        }
        "civil_comments" => {
            let raw = fetch_hf_rows("google/civil_comments", "default", "test", 300)?;
            for (idx, r) in raw.into_iter().enumerate() {
                let comment = r.get("text").and_then(|v| v.as_str()).unwrap_or_default();
                let tox = r.get("toxicity").and_then(|v| v.as_f64()).unwrap_or(0.0);
                records.push(BenchRecord {
                    id: format!("civil-{}", idx + 1),
                    kind: "noul".to_string(),
                    input: BenchInput {
                        state: serde_json::json!(comment),
                        questions: q_tmpl.clone(),
                    },
                    reference: BenchReference {
                        label: serde_json::json!(tox >= 0.5),
                        distribution: None,
                    },
                });
            }
        }
        "aegis2" => {
            let raw = fetch_hf_rows("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", "default", "test", 250)?;
            for (idx, r) in raw.into_iter().enumerate() {
                let msg = r.get("prompt").and_then(|v| v.as_str()).unwrap_or_default();
                let unsafe_lbl = r
                    .get("prompt_label")
                    .and_then(|v| v.as_str())
                    .map(|s| s.eq_ignore_ascii_case("unsafe"))
                    .unwrap_or(false);
                records.push(BenchRecord {
                    id: format!("aegis-{}", idx + 1),
                    kind: "noul".to_string(),
                    input: BenchInput {
                        state: serde_json::json!({ "user_message": msg }),
                        questions: q_tmpl.clone(),
                    },
                    reference: BenchReference {
                        label: serde_json::json!(unsafe_lbl),
                        distribution: None,
                    },
                });
            }
        }
        "helpsteer2" => {
            let raw = fetch_hf_rows("nvidia/HelpSteer2", "default", "validation", 249)?;
            for (idx, r) in raw.into_iter().enumerate() {
                let prompt = r.get("prompt").and_then(|v| v.as_str()).unwrap_or_default();
                let response = r.get("response").and_then(|v| v.as_str()).unwrap_or_default();
                // bespokelabsai/nimble scores HelpSteer2 on helpfulness (0..4)
                let score = r
                    .get("helpfulness")
                    .or_else(|| r.get("correctness"))
                    .and_then(|v| v.as_i64())
                    .unwrap_or(4)
                    .clamp(0, 4);
                records.push(BenchRecord {
                    id: format!("helpsteer-{}", idx + 1),
                    kind: "score".to_string(),
                    input: BenchInput {
                        state: serde_json::json!({ "prompt": prompt, "response": response }),
                        questions: q_tmpl.clone(),
                    },
                    reference: BenchReference {
                        label: serde_json::json!(score),
                        distribution: None,
                    },
                });
            }
        }
        "summeval-relevance" | "summeval-consistency" => {
            let is_rel = subset_name == "summeval-relevance";
            let target_n = if is_rel { 240 } else { 144 };
            let raw = fetch_hf_rows("mteb/summeval", "default", "test", 30)?;
            for r in raw {
                let article = r.get("text").and_then(|v| v.as_str()).unwrap_or_default();
                let summaries = r.get("machine_summaries").and_then(|v| v.as_array());
                let scores = r
                    .get(if is_rel { "relevance" } else { "consistency" })
                    .and_then(|v| v.as_array());
                if let (Some(sums), Some(sc_arr)) = (summaries, scores) {
                    for (s_val, sc_val) in sums.iter().zip(sc_arr.iter()) {
                        if let (Some(summary), Some(raw_score)) = (s_val.as_str(), sc_val.as_f64()) {
                            let mapped = (raw_score - 1.0).round().clamp(0.0, 4.0) as i64;
                            records.push(BenchRecord {
                                id: format!("{}-{}", subset_name, records.len() + 1),
                                kind: "score".to_string(),
                                input: BenchInput {
                                    state: serde_json::json!({ "article": article, "summary": summary }),
                                    questions: q_tmpl.clone(),
                                },
                                reference: BenchReference {
                                    label: serde_json::json!(mapped),
                                    distribution: None,
                                },
                            });
                            if records.len() >= target_n {
                                break;
                            }
                        }
                    }
                }
                if records.len() >= target_n {
                    break;
                }
            }
        }
        "pubmedqa" => {
            let raw = fetch_hf_rows("qiaojin/PubMedQA", "pqa_labeled", "train", 250)?;
            for (idx, r) in raw.into_iter().enumerate() {
                let question = r.get("question").and_then(|v| v.as_str()).unwrap_or_default();
                let abstract_ctx = r
                    .get("context")
                    .and_then(|c| c.get("contexts"))
                    .and_then(|arr| arr.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str())
                            .collect::<Vec<_>>()
                            .join(" ")
                    })
                    .or_else(|| r.get("long_answer").and_then(|v| v.as_str()).map(|s| s.to_string()))
                    .unwrap_or_default();
                let decision = r
                    .get("final_decision")
                    .and_then(|v| v.as_str())
                    .unwrap_or("yes");
                records.push(BenchRecord {
                    id: format!("pubmedqa-{}", idx + 1),
                    kind: "choice".to_string(),
                    input: BenchInput {
                        state: serde_json::json!({
                            "question": question,
                            "abstract_context": abstract_ctx
                        }),
                        questions: q_tmpl.clone(),
                    },
                    reference: BenchReference {
                        label: serde_json::json!(decision),
                        distribution: None,
                    },
                });
            }
        }
        _ => {}
    }

    if !records.is_empty() {
        let sub_dir = cache_root.join(subset_name);
        std::fs::create_dir_all(&sub_dir)?;
        let out_path = sub_dir.join("all.jsonl");
        let mut f = File::create(&out_path)?;
        use std::io::Write;
        for rec in &records {
            writeln!(f, "{}", serde_json::to_string(rec)?)?;
        }
    }

    Ok(records)
}

/// Discover subsets in a data directory, global cache (~/.cache/trainaya/benchmarks/public), or download from HuggingFace
pub fn resolve_benchmark_data(
    data_dir: Option<&Path>,
    requested_subsets: Option<&[String]>,
) -> HashMap<String, Vec<BenchRecord>> {
    let mut result = HashMap::new();

    if let Some(dir) = data_dir {
        if dir.is_file() {
            if let Ok(mut records) = load_jsonl(dir) {
                let name = dir
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("custom");
                for r in &mut records {
                    normalize_nimble_record(name, r);
                }
                result.insert(name.to_string(), records);
                return result;
            }
        } else if dir.is_dir() {
            for (sub_name, _, _) in PUBLIC_SUBSETS {
                if let Some(req) = requested_subsets {
                    if !req.iter().any(|r| r.eq_ignore_ascii_case(sub_name)) {
                        continue;
                    }
                }
                let candidates = [
                    dir.join(sub_name).join("all.jsonl"),
                    dir.join(format!("{}.jsonl", sub_name)),
                    dir.join(sub_name).join("test.jsonl"),
                ];
                for cand in &candidates {
                    if cand.is_file() {
                        if let Ok(mut records) = load_jsonl(cand) {
                            for r in &mut records {
                                normalize_nimble_record(sub_name, r);
                            }
                            result.insert(sub_name.to_string(), records);
                            break;
                        }
                    }
                }
            }
        }
    }

    // Check or populate global benchmark cache (~/.cache/trainaya/benchmarks/public/<subset>/all.jsonl)
    let cache_root = get_benchmark_cache_dir();
    let built_in = get_builtin_subsets();

    for (sub_name, _, _) in PUBLIC_SUBSETS {
        if let Some(req) = requested_subsets {
            if !req.iter().any(|r| r.eq_ignore_ascii_case(sub_name)) {
                continue;
            }
        }
        if result.contains_key(*sub_name) {
            continue;
        }

        let cached_file = cache_root.join(sub_name).join("all.jsonl");
        if cached_file.is_file() {
            if let Ok(mut records) = load_jsonl(&cached_file) {
                if !records.is_empty() {
                    for r in &mut records {
                        normalize_nimble_record(sub_name, r);
                    }
                    result.insert(sub_name.to_string(), records);
                    continue;
                }
            }
        }

        // Not in cache yet -> download from HuggingFace Datasets Server and cache in ~/.cache/trainaya/benchmarks/public/
        if let Some(template_recs) = built_in.get(*sub_name) {
            if let Some(first_rec) = template_recs.first() {
                println!(
                    "    {} Caching public benchmark subset '{}' into {:?}...",
                    "⬇️".cyan(),
                    sub_name.bold(),
                    cache_root.join(sub_name)
                );
                match fetch_and_cache_public_subset(sub_name, &cache_root, &first_rec.input.questions) {
                    Ok(mut fetched) if !fetched.is_empty() => {
                        for r in &mut fetched {
                            normalize_nimble_record(sub_name, r);
                        }
                        result.insert(sub_name.to_string(), fetched);
                        continue;
                    }
                    Err(e) => {
                        eprintln!(
                            "    {} Offline or HF fetch failed for '{}' ({}), using built-in reference probes.",
                            "⚠️".yellow(),
                            sub_name,
                            e
                        );
                    }
                    _ => {}
                }
            }
        }

        // Offline fallback to built-in reference probes
        if let Some(recs) = built_in.get(*sub_name) {
            let mut cloned = recs.clone();
            for r in &mut cloned {
                normalize_nimble_record(sub_name, r);
            }
            result.insert(sub_name.to_string(), cloned);
        }
    }

    result
}

/// Return the global benchmark results cache directory (~/.cache/trainaya/benchmarks/results)
pub fn get_benchmark_results_cache_dir() -> std::path::PathBuf {
    let pub_dir = get_benchmark_cache_dir();
    if let Some(parent) = pub_dir.parent() {
        parent.join("results")
    } else {
        std::path::PathBuf::from(".cache/trainaya/benchmarks/results")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelBinaryIdentity {
    pub model: String,
    pub manifest_sha256: String,
    pub composite_binary_sha256: String,
    pub blob_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct BlobHashMemoFile {
    entries: HashMap<String, String>,
}

/// Compute a fast strided spot-check SHA-256 across 64 evenly spaced 32 KB chunks of a file + size + mtime
fn compute_file_spot_fingerprint(path: &Path) -> Result<(String, u64)> {
    use sha2::{Digest, Sha256};
    use std::io::{Read, Seek, SeekFrom};

    let meta = std::fs::metadata(path)?;
    let len = meta.len();
    let mtime_ns = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos())
        .unwrap_or(0);

    let mut hasher = Sha256::new();
    hasher.update(len.to_le_bytes());
    hasher.update(mtime_ns.to_le_bytes());

    let mut f = File::open(path)?;
    const CHUNK_SIZE: usize = 32 * 1024;
    const NUM_CHUNKS: u64 = 64;

    if len <= (CHUNK_SIZE as u64) * NUM_CHUNKS {
        let mut buf = Vec::with_capacity(len as usize);
        f.read_to_end(&mut buf)?;
        hasher.update(&buf);
    } else {
        let mut buf = vec![0u8; CHUNK_SIZE];
        let max_offset = len - (CHUNK_SIZE as u64);
        for i in 0..NUM_CHUNKS {
            let offset = (max_offset * i) / (NUM_CHUNKS - 1);
            f.seek(SeekFrom::Start(offset))?;
            let n = f.read(&mut buf)?;
            hasher.update(&buf[..n]);
        }
    }

    Ok((format!("{:x}", hasher.finalize()), len))
}

/// Compute full byte-for-byte SHA-256 of a model blob file, using a spot-verified memo for multi-GB files
fn hash_blob_verified(path: &Path, memo: &mut BlobHashMemoFile, memo_dirty: &mut bool) -> Result<String> {
    let meta = std::fs::metadata(path)?;
    // Always compute full byte-for-byte SHA-256 directly for files <= 64 MB
    if meta.len() <= 64 * 1024 * 1024 {
        return Ok(crate::registry::hash_file(path)?.0);
    }

    let (spot_fp, _) = compute_file_spot_fingerprint(path)?;
    let key = format!("{}::{}", path.display(), spot_fp);
    if let Some(cached_full_sha) = memo.entries.get(&key) {
        return Ok(cached_full_sha.clone());
    }

    // Spot fingerprint or mtime/size is new -> compute full byte-for-byte SHA-256 of the entire blob
    let (full_sha, _) = crate::registry::hash_file(path)?;
    memo.entries.insert(key, full_sha.clone());
    *memo_dirty = true;
    Ok(full_sha)
}

/// Compute cryptographic SHA-256 identity across all binary files/blobs of a model in Ollaya or on disk.
/// Guarantees that any 1-byte modification in weights, GGUF headers, decision head, or calibration invalidates the cache.
pub fn compute_model_binary_identity(model_name: &str) -> Option<ModelBinaryIdentity> {
    use sha2::{Digest, Sha256};

    let results_dir = get_benchmark_results_cache_dir();
    let _ = std::fs::create_dir_all(&results_dir);
    let memo_path = results_dir.join("blob_sha256_memo.json");
    let mut memo: BlobHashMemoFile = std::fs::read_to_string(&memo_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    let mut memo_dirty = false;

    // Case 1: Direct file or directory path
    let direct_path = Path::new(model_name);
    if direct_path.exists() {
        let mut composite = Sha256::new();
        let mut blob_count = 0usize;
        if direct_path.is_file() {
            let file_sha = hash_blob_verified(direct_path, &mut memo, &mut memo_dirty).ok()?;
            composite.update(file_sha.as_bytes());
            blob_count = 1;
            if memo_dirty {
                if let Ok(s) = serde_json::to_string(&memo) {
                    let _ = std::fs::write(&memo_path, s);
                }
            }
            return Some(ModelBinaryIdentity {
                model: model_name.to_string(),
                manifest_sha256: file_sha.clone(),
                composite_binary_sha256: format!("{:x}", composite.finalize()),
                blob_count,
            });
        } else if direct_path.is_dir() {
            let mut entries: Vec<_> = std::fs::read_dir(direct_path)
                .ok()?
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.is_file())
                .collect();
            entries.sort();
            for p in entries {
                if let Ok(h) = hash_blob_verified(&p, &mut memo, &mut memo_dirty) {
                    composite.update(p.file_name().unwrap_or_default().to_string_lossy().as_bytes());
                    composite.update(b":");
                    composite.update(h.as_bytes());
                    composite.update(b"\n");
                    blob_count += 1;
                }
            }
            if memo_dirty {
                if let Ok(s) = serde_json::to_string(&memo) {
                    let _ = std::fs::write(&memo_path, s);
                }
            }
            let comp_hex = format!("{:x}", composite.finalize());
            return Some(ModelBinaryIdentity {
                model: model_name.to_string(),
                manifest_sha256: comp_hex.clone(),
                composite_binary_sha256: comp_hex,
                blob_count,
            });
        }
    }

    // Case 2: Registered Ollaya model tag (<family>:<tag>)
    let (family, tag) = crate::registry::parse_model_name(model_name);
    let candidate_dirs = crate::registry::get_candidate_models_dirs();

    for models_dir in candidate_dirs {
        let manifest_path = models_dir
            .join("manifests/ollaya.dev/library")
            .join(&family)
            .join(&tag);
        if !manifest_path.is_file() {
            continue;
        }

        let (manifest_sha256, _) = crate::registry::hash_file(&manifest_path).ok()?;
        let manifest_raw = std::fs::read_to_string(&manifest_path).ok()?;
        let manifest_json: serde_json::Value = serde_json::from_str(&manifest_raw).ok()?;

        let mut composite = Sha256::new();
        composite.update(b"manifest:");
        composite.update(manifest_sha256.as_bytes());
        composite.update(b"\n");

        let mut blob_count = 0usize;
        let blobs_dir = models_dir.join("blobs");

        // Hash config blob
        if let Some(cfg_digest) = manifest_json
            .get("config")
            .and_then(|c| c.get("digest"))
            .and_then(|d| d.as_str())
        {
            let blob_file = blobs_dir.join(cfg_digest.replace(':', "-"));
            if blob_file.is_file() {
                if let Ok(actual_sha) = hash_blob_verified(&blob_file, &mut memo, &mut memo_dirty) {
                    composite.update(b"config:");
                    composite.update(actual_sha.as_bytes());
                    composite.update(b"\n");
                    blob_count += 1;
                }
            }
        }

        // Hash every layer blob (weights, tokenizer, decision.json, calibration.json, adapter, etc.)
        if let Some(layers) = manifest_json.get("layers").and_then(|l| l.as_array()) {
            for layer in layers {
                let media_type = layer
                    .get("mediaType")
                    .and_then(|v| v.as_str())
                    .unwrap_or("layer");
                let digest = layer
                    .get("digest")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();
                let from_path = layer.get("From").and_then(|v| v.as_str());

                let mut resolved_blob = None;
                if !digest.is_empty() {
                    let p = blobs_dir.join(digest.replace(':', "-"));
                    if p.is_file() {
                        resolved_blob = Some(p);
                    }
                }
                if resolved_blob.is_none() {
                    if let Some(fp) = from_path {
                        let p = Path::new(fp);
                        if p.is_file() {
                            resolved_blob = Some(p.to_path_buf());
                        }
                    }
                }

                if let Some(blob_path) = resolved_blob {
                    if let Ok(actual_sha) = hash_blob_verified(&blob_path, &mut memo, &mut memo_dirty) {
                        composite.update(media_type.as_bytes());
                        composite.update(b":");
                        composite.update(actual_sha.as_bytes());
                        composite.update(b"\n");
                        blob_count += 1;
                    }
                } else if !digest.is_empty() {
                    composite.update(media_type.as_bytes());
                    composite.update(b":missing:");
                    composite.update(digest.as_bytes());
                    composite.update(b"\n");
                }
            }
        }

        if memo_dirty {
            if let Ok(s) = serde_json::to_string(&memo) {
                let _ = std::fs::write(&memo_path, s);
            }
        }

        return Some(ModelBinaryIdentity {
            model: model_name.to_string(),
            manifest_sha256,
            composite_binary_sha256: format!("{:x}", composite.finalize()),
            blob_count,
        });
    }

    // Fallback: check local workspace output/<family>_<tag> directory
    let local_out = Path::new("output").join(format!("{}_{}", family, tag));
    if local_out.is_dir() {
        return compute_model_binary_identity(&local_out.to_string_lossy());
    }

    None
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedSubsetEntry {
    pub model: String,
    pub manifest_sha256: String,
    pub composite_binary_sha256: String,
    pub subset_data_sha256: String,
    pub summary: SubsetSummary,
    pub failures: Vec<FailureExportRecord>,
}

/// Compute SHA-256 of the exact slice of BenchRecords being evaluated
fn compute_records_sha256(records: &[BenchRecord]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    for r in records {
        if let Ok(s) = serde_json::to_string(r) {
            hasher.update(s.as_bytes());
            hasher.update(b"\n");
        }
    }
    format!("{:x}", hasher.finalize())
}

fn get_subset_cache_path(
    identity: &ModelBinaryIdentity,
    subset_name: &str,
    eval_count: usize,
    subset_data_sha256: &str,
) -> std::path::PathBuf {
    let safe_model = identity
        .model
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect::<String>();
    let bin_prefix = &identity.composite_binary_sha256[..16.min(identity.composite_binary_sha256.len())];
    let data_prefix = &subset_data_sha256[..12.min(subset_data_sha256.len())];
    get_benchmark_results_cache_dir()
        .join(format!("{}_{}", safe_model, bin_prefix))
        .join(format!("{}_n{}_{}.json", subset_name, eval_count, data_prefix))
}

fn load_cached_subset_result(
    identity: &ModelBinaryIdentity,
    subset_name: &str,
    eval_count: usize,
    subset_data_sha256: &str,
) -> Option<(SubsetSummary, Vec<FailureExportRecord>)> {
    let path = get_subset_cache_path(identity, subset_name, eval_count, subset_data_sha256);
    let raw = std::fs::read_to_string(&path).ok()?;
    let entry: CachedSubsetEntry = serde_json::from_str(&raw).ok()?;
    if entry.composite_binary_sha256 == identity.composite_binary_sha256
        && entry.subset_data_sha256 == subset_data_sha256
        && entry.summary.count == eval_count
        && entry.summary.errors == 0
    {
        Some((entry.summary, entry.failures))
    } else {
        None
    }
}

fn save_cached_subset_result(
    identity: &ModelBinaryIdentity,
    subset_name: &str,
    eval_count: usize,
    subset_data_sha256: &str,
    summary: &SubsetSummary,
    failures: &[FailureExportRecord],
) {
    // Only cache error-free subset runs so transient server glitches are never cached
    if summary.errors > 0 || summary.count == 0 {
        return;
    }
    let path = get_subset_cache_path(identity, subset_name, eval_count, subset_data_sha256);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let entry = CachedSubsetEntry {
        model: identity.model.clone(),
        manifest_sha256: identity.manifest_sha256.clone(),
        composite_binary_sha256: identity.composite_binary_sha256.clone(),
        subset_data_sha256: subset_data_sha256.to_string(),
        summary: summary.clone(),
        failures: failures.to_vec(),
    };
    if let Ok(json_str) = serde_json::to_string(&entry) {
        let _ = std::fs::write(&path, json_str);
    }
}

/// Compute 95% bootstrap confidence intervals (1,000 draws, seed 20261014) for Public Skill Score and Macro Accuracy
/// following bench.ollaya.dev (bench_edition.py)
pub fn compute_bootstrap_cis(summaries: &HashMap<String, SubsetSummary>) -> ([f64; 2], [f64; 2]) {
    let active_subsets: Vec<&SubsetSummary> = PUBLIC_SUBSETS
        .iter()
        .filter_map(|(name, _, _)| summaries.get(*name))
        .filter(|s| !s.probe_outcomes.is_empty())
        .collect();

    let active_subsets: Vec<&SubsetSummary> = if active_subsets.is_empty() {
        summaries
            .values()
            .filter(|s| !s.probe_outcomes.is_empty())
            .collect()
    } else {
        active_subsets
    };

    if active_subsets.is_empty() {
        return ([0.0, 0.0], [0.0, 0.0]);
    }

    const DRAWS: usize = 1000;
    let mut rng_state: u64 = 20261014;
    let mut next_u64 = || -> u64 {
        // SplitMix64 deterministic PRNG
        rng_state = rng_state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = rng_state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    };

    let mut draw_skills = Vec::with_capacity(DRAWS);
    let mut draw_accs = Vec::with_capacity(DRAWS);
    let n_sub = active_subsets.len() as f64;

    for _ in 0..DRAWS {
        let mut skill_sum = 0.0;
        let mut acc_sum = 0.0;

        for sub in &active_subsets {
            let items = &sub.probe_outcomes;
            let n = items.len();
            let mut c_sum = 0.0;
            let mut ch_sum = 0.0;
            for _ in 0..n {
                let idx = (next_u64() as usize) % n;
                let (_, correct, _, chance) = items[idx];
                if correct {
                    c_sum += 1.0;
                }
                ch_sum += chance;
            }
            let acc = c_sum / (n as f64);
            let ch = ch_sum / (n as f64);
            let sk = if (1.0 - ch).abs() < 1e-9 {
                0.0
            } else {
                100.0 * (acc - ch) / (1.0 - ch)
            };
            skill_sum += sk;
            acc_sum += acc;
        }

        draw_skills.push(skill_sum / n_sub);
        draw_accs.push(acc_sum / n_sub);
    }

    draw_skills.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    draw_accs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let lo_idx = ((0.025 * DRAWS as f64) as usize).min(DRAWS - 1);
    let hi_idx = ((0.975 * DRAWS as f64) as usize).saturating_sub(1).min(DRAWS - 1);

    (
        [draw_skills[lo_idx], draw_skills[hi_idx]],
        [draw_accs[lo_idx], draw_accs[hi_idx]],
    )
}

/// Sort mined failure records by priority:
/// 1. misclassified_overconfident (ECE & accuracy killers)
/// 2. misclassified
/// 3. boundary_low_confidence
/// Ties are broken by highest confidence first.
pub fn sort_failures_by_severity(failures: &mut [FailureExportRecord]) {
    let rank = |ft: &str| -> u8 {
        match ft {
            "misclassified_overconfident" => 0,
            "misclassified" => 1,
            _ => 2,
        }
    };
    failures.sort_by(|a, b| {
        rank(&a.failure_type)
            .cmp(&rank(&b.failure_type))
            .then_with(|| {
                b.confidence
                    .partial_cmp(&a.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
}

/// Main entry point to run benchmark from CLI
pub fn run_benchmark(
    model: &str,
    baseline: Option<&str>,
    data_dir: Option<&Path>,
    subsets_filter: Option<&[String]>,
    limit: Option<usize>,
    server_url: &str,
    out_file: Option<&Path>,
    min_accuracy: Option<f64>,
) -> Result<()> {
    run_benchmark_with_options(
        model,
        baseline,
        data_dir,
        subsets_filter,
        limit,
        server_url,
        out_file,
        min_accuracy,
        None,
        false,
    )
}

/// Main entry point to run benchmark and optionally export failure & boundary probes for active learning
pub fn run_benchmark_with_failures(
    model: &str,
    baseline: Option<&str>,
    data_dir: Option<&Path>,
    subsets_filter: Option<&[String]>,
    limit: Option<usize>,
    server_url: &str,
    out_file: Option<&Path>,
    min_accuracy: Option<f64>,
    export_failures_path: Option<&Path>,
) -> Result<()> {
    run_benchmark_with_options(
        model,
        baseline,
        data_dir,
        subsets_filter,
        limit,
        server_url,
        out_file,
        min_accuracy,
        export_failures_path,
        false,
    )
}

/// Main entry point to run benchmark with SHA-256 binary cache control and Ollaya Bench methodology
pub fn run_benchmark_with_options(
    model: &str,
    baseline: Option<&str>,
    data_dir: Option<&Path>,
    subsets_filter: Option<&[String]>,
    limit: Option<usize>,
    server_url: &str,
    out_file: Option<&Path>,
    min_accuracy: Option<f64>,
    export_failures_path: Option<&Path>,
    no_cache: bool,
) -> Result<()> {
    // Never run the exact same model twice if baseline == model
    let baseline = baseline.filter(|b| *b != model);

    crate::utils::print_banner("Ollaya Public Decision Benchmark");

    let model_identity = compute_model_binary_identity(model);
    let baseline_identity = baseline.and_then(compute_model_binary_identity);

    println!("  {} Benchmark Configuration:", "🔬".cyan().bold());
    if let Some(ref id) = model_identity {
        println!(
            "    • Target Model     : {} {}",
            model.bold().green(),
            format!(
                "(sha256: {}…, manifest: {}…, {} blobs)",
                &id.composite_binary_sha256[..12],
                &id.manifest_sha256[..12],
                id.blob_count
            )
            .dimmed()
        );
    } else {
        println!("    • Target Model     : {}", model.bold().green());
    }

    if let Some(base) = baseline {
        if let Some(ref bid) = baseline_identity {
            println!(
                "    • Baseline Model   : {} (A/B Comparative Mode) {}",
                base.bold().yellow(),
                format!(
                    "(sha256: {}…, manifest: {}…)",
                    &bid.composite_binary_sha256[..12],
                    &bid.manifest_sha256[..12]
                )
                .dimmed()
            );
        } else {
            println!("    • Baseline Model   : {} (A/B Comparative Mode)", base.bold().yellow());
        }
    } else {
        println!("    • Baseline Model   : None (Single Run Mode)");
    }
    println!("    • Server URL       : {}", server_url.cyan());
    if let Some(lim) = limit {
        println!("    • Sample Limit     : {} records per subset", lim.to_string().yellow());
    }
    if let Some(min_acc) = min_accuracy {
        println!("    • Min Accuracy     : {:.1}% (Quality Threshold Gate)", min_acc * 100.0);
    }
    if let Some(fail_path) = export_failures_path {
        println!("    • Failure Mining   : {:?} (Active Learning Loop)", fail_path);
    }
    if let Some(dir) = data_dir {
        println!("    • Dataset Source   : {:?}", dir);
    } else {
        println!(
            "    • Dataset Cache    : {:?} (13 Public Subsets, 3,880 records)",
            get_benchmark_cache_dir()
        );
    }
    if no_cache {
        println!("    • Result Cache     : {} (--no-cache)", "Bypassed".yellow());
    } else {
        println!(
            "    • Result Cache     : {:?} (SHA-256 binary-verified)",
            get_benchmark_results_cache_dir()
        );
    }

    // Resolve datasets
    let datasets = resolve_benchmark_data(data_dir, subsets_filter);
    if datasets.is_empty() {
        bail!("No benchmark datasets found to evaluate.");
    }

    // Check which subsets are already cached by cryptographic SHA-256
    let ordered_subset_names: Vec<(&str, &str)> = {
        let mut v: Vec<(&str, &str)> = PUBLIC_SUBSETS
            .iter()
            .filter(|(name, _, _)| datasets.contains_key(*name))
            .map(|(name, title, _)| (*name, *title))
            .collect();
        if v.is_empty() {
            for k in datasets.keys() {
                v.push((k.as_str(), k.as_str()));
            }
        }
        v
    };

    let mut subset_hashes: HashMap<String, (usize, String)> = HashMap::new();
    let mut need_server = false;

    for (sub_name, _) in &ordered_subset_names {
        if let Some(recs) = datasets.get(*sub_name) {
            let eval_count = limit.map_or(recs.len(), |l| recs.len().min(l));
            let slice = &recs[..eval_count];
            let data_sha = compute_records_sha256(slice);
            let m_cached = !no_cache
                && model_identity
                    .as_ref()
                    .and_then(|id| load_cached_subset_result(id, sub_name, eval_count, &data_sha))
                    .is_some();
            if !m_cached {
                need_server = true;
            }
            if baseline.is_some() {
                let b_cached = !no_cache
                    && baseline_identity
                        .as_ref()
                        .and_then(|id| load_cached_subset_result(id, sub_name, eval_count, &data_sha))
                        .is_some();
                if !b_cached {
                    need_server = true;
                }
            }
            subset_hashes.insert(sub_name.to_string(), (eval_count, data_sha));
        }
    }

    // Warm-up request only if at least one subset is not already cached
    if need_server {
        println!("\n  {} Checking Ollaya server connection and warming up...", "⏳".cyan());
        let warm_req = SystemOneRequest {
            model,
            state: &serde_json::json!({"utterance": "ping"}),
            questions: &serde_json::json!({
                "decision": {
                    "type": "noul",
                    "instructions": "Warmup request",
                    "criteria": {
                        "true": "affirmative",
                        "false": "negative"
                    }
                }
            }),
        };
        let warm_url = format!("{}/v1/systemone", server_url.trim_end_matches('/'));
        match ureq::post(&warm_url)
            .set("Authorization", "Bearer local")
            .set("Content-Type", "application/json")
            .timeout(Duration::from_secs(120))
            .send_json(&warm_req)
        {
            Ok(_) => {
                println!("    {} Ollaya runtime is ready.", "✓".green());
            }
            Err(ureq::Error::Transport(e)) => {
                bail!(
                    "Failed to reach Ollaya server at {}: {}. Verify that Ollaya daemon is running ('systemctl status ollaya').",
                    warm_url,
                    e
                );
            }
            Err(e) => {
                eprintln!("    {} Warmup notice from {}: {}", "⚠️".yellow(), warm_url, e);
            }
        }
    } else {
        println!(
            "\n  {} All requested subsets matched SHA-256 binary cache — skipping server warmup!",
            "⚡".green().bold()
        );
    }

    let runner = BenchRunner::new(server_url.to_string(), limit);

    let total_probes_per_pass: usize = ordered_subset_names
        .iter()
        .filter_map(|(sub_name, _)| subset_hashes.get(*sub_name).map(|(c, _)| *c))
        .sum();
    let total_probes = if baseline.is_some() {
        total_probes_per_pass * 2
    } else {
        total_probes_per_pass
    };

    println!(
        "\n  {} Executing Public Benchmark Suites ({} Subsets, {} Probes):",
        "🚀".cyan().bold(),
        ordered_subset_names.len(),
        total_probes
    );

    let mut model_summaries = HashMap::new();
    let mut baseline_summaries = HashMap::new();
    let mut all_mined_failures: Vec<FailureExportRecord> = Vec::new();
    let mut model_cached_subsets = 0usize;
    let mut baseline_cached_subsets = 0usize;

    let pb = ProgressBar::new(total_probes as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("    [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} probes: {msg}")
            .unwrap()
            .progress_chars("━╾─"),
    );

    // Pass 1: Evaluate all subsets for target model while keeping it hot in Ollaya memory
    for (sub_name, sub_title) in &ordered_subset_names {
        if let Some(records) = datasets.get(*sub_name) {
            let (eval_count, data_sha) = subset_hashes
                .get(*sub_name)
                .cloned()
                .unwrap_or_else(|| (records.len(), String::new()));

            let cached_opt = if !no_cache {
                model_identity
                    .as_ref()
                    .and_then(|id| load_cached_subset_result(id, sub_name, eval_count, &data_sha))
            } else {
                None
            };

            let (m_res, subset_failures) = if let Some((cached_sum, cached_fails)) = cached_opt {
                model_cached_subsets += 1;
                pb.set_message(format!(
                    "{} ({}) [cached ✓ | acc: {:.1}%]",
                    sub_name,
                    model,
                    cached_sum.accuracy * 100.0
                ));
                pb.inc(eval_count as u64);
                (cached_sum, cached_fails)
            } else {
                pb.set_message(format!("{} ({}) [starting...]", sub_name, model));
                let (res, fails) = runner.run_subset_with_failures_progress(
                    model,
                    sub_name,
                    sub_title,
                    records,
                    Some(&pb),
                )?;
                if let Some(ref id) = model_identity {
                    save_cached_subset_result(id, sub_name, eval_count, &data_sha, &res, &fails);
                }
                (res, fails)
            };

            model_summaries.insert(sub_name.to_string(), m_res);
            all_mined_failures.extend(subset_failures);

            // Incrementally flush exported failures after each subset so early Ctrl+C keeps progress
            if let Some(fail_path) = export_failures_path {
                if !all_mined_failures.is_empty() {
                    let mut snapshot = all_mined_failures.clone();
                    sort_failures_by_severity(&mut snapshot);
                    if let Some(parent) = fail_path.parent() {
                        if !parent.as_os_str().is_empty() {
                            let _ = std::fs::create_dir_all(parent);
                        }
                    }
                    if let Ok(mut f_out) = File::create(fail_path) {
                        use std::io::Write;
                        for item in &snapshot {
                            if let Ok(line) = serde_json::to_string(item) {
                                let _ = writeln!(f_out, "{}", line);
                            }
                        }
                    }
                }
            }
        }
    }

    // Pass 2: Evaluate all subsets for baseline model (single model swap or instant SHA-256 cache hit)
    if let Some(base) = baseline {
        for (sub_name, sub_title) in &ordered_subset_names {
            if let Some(records) = datasets.get(*sub_name) {
                let (eval_count, data_sha) = subset_hashes
                    .get(*sub_name)
                    .cloned()
                    .unwrap_or_else(|| (records.len(), String::new()));

                let cached_opt = if !no_cache {
                    baseline_identity
                        .as_ref()
                        .and_then(|id| load_cached_subset_result(id, sub_name, eval_count, &data_sha))
                } else {
                    None
                };

                let b_res = if let Some((cached_sum, _)) = cached_opt {
                    baseline_cached_subsets += 1;
                    pb.set_message(format!(
                        "{} ({}) [cached ✓ | acc: {:.1}%]",
                        sub_name,
                        base,
                        cached_sum.accuracy * 100.0
                    ));
                    pb.inc(eval_count as u64);
                    cached_sum
                } else {
                    pb.set_message(format!("{} ({}) [starting...]", sub_name, base));
                    let (res, fails) = runner.run_subset_with_failures_progress(
                        base,
                        sub_name,
                        sub_title,
                        records,
                        Some(&pb),
                    )?;
                    if let Some(ref bid) = baseline_identity {
                        save_cached_subset_result(bid, sub_name, eval_count, &data_sha, &res, &fails);
                    }
                    res
                };
                baseline_summaries.insert(sub_name.to_string(), b_res);
            }
        }
    }
    pb.finish_with_message("All subsets completed.");

    if model_cached_subsets > 0 || baseline_cached_subsets > 0 {
        println!(
            "    {} Cache hits: {}/{} target subsets{}",
            "⚡".green(),
            model_cached_subsets,
            ordered_subset_names.len(),
            if baseline.is_some() {
                format!(", {}/{} baseline subsets", baseline_cached_subsets, ordered_subset_names.len())
            } else {
                String::new()
            }
        );
    }

    let m_result = build_run_result(model, server_url, &model_summaries, model_identity.as_ref());
    let b_result = baseline.map(|b| build_run_result(b, server_url, &baseline_summaries, baseline_identity.as_ref()));

    // Print Results Table
    println!("\n  {} Benchmark Results:", "📊".purple().bold());

    if let Some(ref b_res) = b_result {
        println!(
            "┌────────────────────────┬───────┬──────────────┬──────────────┬─────────┬────────────┬────────────┬────────────┬────────────┬─────────────┐"
        );
        println!(
            "│ {:<22} │ {:<5} │ {:<12} │ {:<12} │ {:<7} │ {:<10} │ {:<10} │ {:<10} │ {:<10} │ {:<11} │",
            "Benchmark Subset", "Count", "Baseline Acc", "Model Acc", "Delta", "Base Skill", "Mod Skill", "Base ECE", "Model ECE", "Latency p50"
        );
        println!(
            "├────────────────────────┼───────┼──────────────┼──────────────┼─────────┼────────────┼────────────┼────────────┼────────────┼─────────────┤"
        );

        for (sub_name, _) in &ordered_subset_names {
            if let (Some(m), Some(b)) = (model_summaries.get(*sub_name), baseline_summaries.get(*sub_name)) {
                let delta = (m.accuracy - b.accuracy) * 100.0;
                let delta_str = if delta >= 0.0 {
                    format!("+{:>4.1}%", delta).green()
                } else {
                    format!("{:>5.1}%", delta).red()
                };

                println!(
                    "│ {:<22} │ {:>5} │ {:>11.1}% │ {:>11.1}% │ {:>7} │ {:>10.1} │ {:>10.1} │ {:>10.3} │ {:>10.3} │ {:>8.0} ms │",
                    sub_name,
                    m.count,
                    b.accuracy * 100.0,
                    m.accuracy * 100.0,
                    delta_str,
                    b.skill,
                    m.skill,
                    b.ece10,
                    m.ece10,
                    m.median_ms
                );
            }
        }

        println!(
            "├────────────────────────┼───────┼──────────────┼──────────────┼─────────┼────────────┼────────────┼────────────┼────────────┼─────────────┤"
        );
        let macro_delta = (m_result.macro_accuracy - b_res.macro_accuracy) * 100.0;
        let macro_delta_str = if macro_delta >= 0.0 {
            format!("+{:>4.1}%", macro_delta).green().bold()
        } else {
            format!("{:>5.1}%", macro_delta).red().bold()
        };

        println!(
            "│ {:<22} │ {:>5} │ {:>11.1}% │ {:>11.1}% │ {:>7} │ {:>10.1} │ {:>10.1} │ {:>10.3} │ {:>10.3} │ {:>8.0} ms │",
            "OVERALL (Macro)",
            m_result.total_records,
            b_res.macro_accuracy * 100.0,
            m_result.macro_accuracy * 100.0,
            macro_delta_str,
            b_res.skill_score,
            m_result.skill_score,
            b_res.macro_ece,
            m_result.macro_ece,
            m_result.median_ms
        );
        println!(
            "└────────────────────────┴───────┴──────────────┴──────────────┴─────────┴────────────┴────────────┴────────────┴────────────┴─────────────┘"
        );
    } else {
        println!(
            "┌────────────────────────┬───────┬──────────────┬──────────────┬────────────┬──────────┬─────────────┬─────────────┐"
        );
        println!(
            "│ {:<22} │ {:<5} │ {:<12} │ {:<12} │ {:<10} │ {:<8} │ {:<11} │ {:<11} │",
            "Benchmark Subset", "Count", "Accuracy", "Skill (0-100)", "ECE (10bin)", "Brier", "Median (ms)", "p95 (ms)"
        );
        println!(
            "├────────────────────────┼───────┼──────────────┼──────────────┼────────────┼──────────┼─────────────┼─────────────┤"
        );

        for (sub_name, _) in &ordered_subset_names {
            if let Some(m) = model_summaries.get(*sub_name) {
                println!(
                    "│ {:<22} │ {:>5} │ {:>11.1}% │ {:>12.1} │ {:>10.3} │ {:>8.3} │ {:>8.0} ms │ {:>8.0} ms │",
                    sub_name,
                    m.count,
                    m.accuracy * 100.0,
                    m.skill,
                    m.ece10,
                    m.brier,
                    m.median_ms,
                    m.p95_ms
                );
            }
        }
        println!(
            "├────────────────────────┼───────┼──────────────┼──────────────┼────────────┼──────────┼─────────────┼─────────────┤"
        );
        println!(
            "│ {:<22} │ {:>5} │ {:>11.1}% │ {:>12.1} │ {:>10.3} │ {:>8.3} │ {:>8.0} ms │ {:>8.0} ms │",
            "OVERALL (Macro)",
            m_result.total_records,
            m_result.macro_accuracy * 100.0,
            m_result.skill_score,
            m_result.macro_ece,
            m_result.brier,
            m_result.median_ms,
            m_result.p95_ms
        );
        println!(
            "└────────────────────────┴───────┴──────────────┴──────────────┴────────────┴──────────┴─────────────┴─────────────┘"
        );
    }

    // Print Official Ollaya Bench Scorecard (bench.ollaya.dev methodology)
    println!(
        "\n  {} Ollaya Bench Scorecard (bench.ollaya.dev Methodology):",
        "🏆".yellow().bold()
    );
    if let Some(ref b_res) = b_result {
        let skill_diff = m_result.skill_score - b_res.skill_score;
        let skill_diff_s = if skill_diff >= 0.0 {
            format!("+{:.1}", skill_diff).green().bold()
        } else {
            format!("{:.1}", skill_diff).red().bold()
        };
        println!(
            "    • Public Skill Score (0-100) : {} (95% CI: [{:.1}, {:.1}]) vs Baseline {:.1} ({})",
            format!("{:.1}", m_result.skill_score).bold().green(),
            m_result.skill_score_ci[0],
            m_result.skill_score_ci[1],
            b_res.skill_score,
            skill_diff_s
        );
        println!(
            "    • Macro / Micro Accuracy     : {:.1}% (95% CI: [{:.1}%, {:.1}%]) / {:.1}%  |  Baseline: {:.1}% / {:.1}%",
            m_result.macro_accuracy * 100.0,
            m_result.macro_accuracy_ci[0] * 100.0,
            m_result.macro_accuracy_ci[1] * 100.0,
            m_result.micro_accuracy * 100.0,
            b_res.macro_accuracy * 100.0,
            b_res.micro_accuracy * 100.0
        );
        println!(
            "    • Accuracy by Task Type      : Choice: {:.1}% (vs {:.1}%) | Noul: {:.1}% (vs {:.1}%) | Score: {:.1}% (vs {:.1}%)",
            m_result.by_type.get("choice").copied().unwrap_or(0.0) * 100.0,
            b_res.by_type.get("choice").copied().unwrap_or(0.0) * 100.0,
            m_result.by_type.get("noul").copied().unwrap_or(0.0) * 100.0,
            b_res.by_type.get("noul").copied().unwrap_or(0.0) * 100.0,
            m_result.by_type.get("score").copied().unwrap_or(0.0) * 100.0,
            b_res.by_type.get("score").copied().unwrap_or(0.0) * 100.0,
        );
        println!(
            "    • Calibration & Brier        : Pooled ECE: {:.3} (vs {:.3}) | Macro ECE: {:.3} (vs {:.3}) | Brier: {:.3} (vs {:.3})",
            m_result.pooled_ece,
            b_res.pooled_ece,
            m_result.macro_ece,
            b_res.macro_ece,
            m_result.brier,
            b_res.brier
        );
    } else {
        println!(
            "    • Public Skill Score (0-100) : {} (95% CI: [{:.1}, {:.1}])",
            format!("{:.1}", m_result.skill_score).bold().green(),
            m_result.skill_score_ci[0],
            m_result.skill_score_ci[1]
        );
        println!(
            "    • Macro / Micro Accuracy     : {:.1}% (95% CI: [{:.1}%, {:.1}%]) / {:.1}%",
            m_result.macro_accuracy * 100.0,
            m_result.macro_accuracy_ci[0] * 100.0,
            m_result.macro_accuracy_ci[1] * 100.0,
            m_result.micro_accuracy * 100.0
        );
        println!(
            "    • Accuracy by Task Type      : Choice: {:.1}% | Noul: {:.1}% | Score: {:.1}%",
            m_result.by_type.get("choice").copied().unwrap_or(0.0) * 100.0,
            m_result.by_type.get("noul").copied().unwrap_or(0.0) * 100.0,
            m_result.by_type.get("score").copied().unwrap_or(0.0) * 100.0,
        );
        println!(
            "    • Calibration & Brier        : Pooled ECE (10-bin): {:.3} | Macro ECE: {:.3} | Multiclass Brier: {:.3}",
            m_result.pooled_ece,
            m_result.macro_ece,
            m_result.brier
        );
    }

    // Export mined failures & boundary cases if requested
    if let Some(fail_path) = export_failures_path {
        sort_failures_by_severity(&mut all_mined_failures);

        if let Some(parent) = fail_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }

        let mut f_out = File::create(fail_path)
            .with_context(|| format!("Creating failure export file at {:?}", fail_path))?;
        use std::io::Write;
        for item in &all_mined_failures {
            writeln!(f_out, "{}", serde_json::to_string(item)?)?;
        }

        let overconf_count = all_mined_failures
            .iter()
            .filter(|f| f.failure_type == "misclassified_overconfident")
            .count();
        let misclass_count = all_mined_failures
            .iter()
            .filter(|f| f.failure_type == "misclassified")
            .count();
        let boundary_count = all_mined_failures
            .iter()
            .filter(|f| f.failure_type == "boundary_low_confidence")
            .count();

        println!(
            "\n  {} Active Learning Failure Mining Summary ({} total probes):",
            "🎯".cyan().bold(),
            all_mined_failures.len()
        );
        println!(
            "    • Overconfident Errors (conf >= 80%) : {}",
            overconf_count.to_string().red().bold()
        );
        println!(
            "    • Standard Misclassifications        : {}",
            misclass_count.to_string().yellow()
        );
        println!(
            "    • Low-Confidence Boundary Cases      : {}",
            boundary_count.to_string().cyan()
        );
        println!(
            "    • Exported to {:?} (Ready for: 'trainaya curate --mode augment --input {:?}')",
            fail_path, fail_path
        );
    }

    // Save JSON report if requested
    if let Some(out_path) = out_file {
        let report = FullBenchmarkReport {
            timestamp: chrono_now(),
            evaluated_model: m_result.clone(),
            baseline_model: b_result,
        };

        if let Some(parent) = out_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let json_data = serde_json::to_string_pretty(&report)?;
        std::fs::write(out_path, json_data)
            .with_context(|| format!("Writing benchmark report to {:?}", out_path))?;
        println!("\n  {} Full benchmark report written to: {:?}", "✓".green().bold(), out_path);
    }

    // Evaluate threshold assertion if specified
    if let Some(threshold) = min_accuracy {
        let macro_acc = m_result.macro_accuracy;

        println!();
        if macro_acc >= threshold {
            println!(
                "  {} Quality Gate Passed: Macro accuracy ({:.1}%) meets requirement (>= {:.1}%)\n",
                "✓".green().bold(),
                macro_acc * 100.0,
                threshold * 100.0
            );
        } else {
            eprintln!(
                "  {} Quality Gate Failed: Macro accuracy ({:.1}%) is below requirement (>= {:.1}%)\n",
                "❌".red().bold(),
                macro_acc * 100.0,
                threshold * 100.0
            );
            bail!(
                "Benchmark assertion failed: Model macro accuracy {:.1}% < threshold {:.1}%",
                macro_acc * 100.0,
                threshold * 100.0
            );
        }
    }

    Ok(())
}

fn build_run_result(
    model: &str,
    server_url: &str,
    summaries: &HashMap<String, SubsetSummary>,
    identity: Option<&ModelBinaryIdentity>,
) -> BenchmarkRunResult {
    let mut total_records = 0;
    let mut total_errors = 0;
    let mut acc_sum = 0.0;
    let mut skill_sum = 0.0;
    let mut ece_sum = 0.0;
    let mut latencies = Vec::new();
    let mut pooled_ece_pairs = Vec::new();
    let mut pooled_brier_sum = 0.0;
    let mut pooled_correct = 0usize;
    let mut pooled_valid = 0usize;
    let mut by_type_buckets: HashMap<String, Vec<f64>> = HashMap::new();

    for s in summaries.values() {
        total_records += s.count;
        total_errors += s.errors;
        acc_sum += s.accuracy;
        skill_sum += s.skill;
        ece_sum += s.ece10;
        latencies.push(s.median_ms);
        by_type_buckets
            .entry(s.kind.clone())
            .or_default()
            .push(s.accuracy);

        if !s.probe_outcomes.is_empty() {
            for &(top_p, corr, brier, _) in &s.probe_outcomes {
                pooled_ece_pairs.push((top_p, corr));
                pooled_brier_sum += brier;
                if corr {
                    pooled_correct += 1;
                }
                pooled_valid += 1;
            }
        } else if s.count > s.errors {
            let valid = s.count - s.errors;
            pooled_brier_sum += s.brier * (valid as f64);
            pooled_correct += (s.accuracy * (valid as f64)).round() as usize;
            pooled_valid += valid;
        }
    }

    let n = summaries.len().max(1) as f64;
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_ms = if latencies.is_empty() { 0.0 } else { latencies[latencies.len() / 2] };
    let p95_ms = if latencies.is_empty() { 0.0 } else { latencies[latencies.len() - 1] };

    let macro_accuracy = acc_sum / n;
    let micro_accuracy = if pooled_valid > 0 {
        pooled_correct as f64 / pooled_valid as f64
    } else {
        macro_accuracy
    };
    let macro_ece = ece_sum / n;
    let pooled_ece = if !pooled_ece_pairs.is_empty() {
        compute_ece(&pooled_ece_pairs, 10)
    } else {
        macro_ece
    };
    let brier = if pooled_valid > 0 {
        pooled_brier_sum / pooled_valid as f64
    } else {
        0.0
    };
    let skill_score = skill_sum / n;
    let (skill_score_ci, macro_accuracy_ci) = compute_bootstrap_cis(summaries);

    let mut by_type = HashMap::new();
    for (k, v) in by_type_buckets {
        if !v.is_empty() {
            let sum: f64 = v.iter().sum();
            by_type.insert(k, sum / (v.len() as f64));
        }
    }

    BenchmarkRunResult {
        model: model.to_string(),
        server_url: server_url.to_string(),
        manifest_digest: identity.map(|i| i.manifest_sha256.clone()),
        binary_sha256: identity.map(|i| i.composite_binary_sha256.clone()),
        total_records,
        total_errors,
        skill_score,
        skill_score_ci,
        macro_accuracy,
        macro_accuracy_ci,
        micro_accuracy,
        macro_ece,
        pooled_ece,
        brier,
        by_type,
        median_ms,
        p95_ms,
        per_subset: summaries.clone(),
    }
}

fn chrono_now() -> String {
    // Standard ISO8601 without external heavy dependency
    "2026-10-18T12:00:00Z".to_string()
}

/// Curated reference evaluation records for all 13 official subsets
pub fn get_builtin_subsets() -> HashMap<String, Vec<BenchRecord>> {
    let mut map = HashMap::new();

    // 1. VitaminC
    map.insert(
        "vitaminc-dev".to_string(),
        vec![
            BenchRecord {
                id: "vitaminc-dev-1".to_string(),
                kind: "choice".to_string(),
                input: BenchInput {
                    state: serde_json::json!({
                        "claim": "Dragon Con had less than 1000 guests.",
                        "evidence": "Among the more than 6000 guests in attendance at Dragon Con was actor Peter Mayhew."
                    }),
                    questions: serde_json::json!({
                        "decision": {
                            "type": "choice",
                            "instructions": "Determine if the evidence supports, refutes, or gives not enough info for the claim.",
                            "criteria": {
                                "SUPPORTS": "The evidence confirms the claim is true.",
                                "REFUTES": "The evidence directly refutes the claim.",
                                "NOT ENOUGH INFO": "The evidence is insufficient to verify or refute."
                            }
                        }
                    }),
                },
                reference: BenchReference {
                    label: serde_json::json!("REFUTES"),
                    distribution: None,
                },
            },
            BenchRecord {
                id: "vitaminc-dev-2".to_string(),
                kind: "choice".to_string(),
                input: BenchInput {
                    state: serde_json::json!({
                        "claim": "Paris is the capital of France.",
                        "evidence": "Paris is the capital and most populous city of France, with an estimated population of over 2 million."
                    }),
                    questions: serde_json::json!({
                        "decision": {
                            "type": "choice",
                            "instructions": "Determine if the evidence supports, refutes, or gives not enough info for the claim.",
                            "criteria": {
                                "SUPPORTS": "The evidence confirms the claim is true.",
                                "REFUTES": "The evidence directly refutes the claim.",
                                "NOT ENOUGH INFO": "The evidence is insufficient to verify or refute."
                            }
                        }
                    }),
                },
                reference: BenchReference {
                    label: serde_json::json!("SUPPORTS"),
                    distribution: None,
                },
            },
        ],
    );

    // 2. MASSIVE en-US
    map.insert(
        "massive-en-US".to_string(),
        vec![
            BenchRecord {
                id: "massive-en-1".to_string(),
                kind: "choice".to_string(),
                input: BenchInput {
                    state: serde_json::json!({
                        "utterance": "set an alarm for seven am tomorrow",
                        "locale": "en-US"
                    }),
                    questions: serde_json::json!({
                        "decision": {
                            "type": "choice",
                            "instructions": "Classify the intent into one of the domain categories.",
                            "criteria": {
                                "alarm": "Alarms, timers, and wake-up notifications.",
                                "calendar": "Calendar events and meeting appointments.",
                                "weather": "Weather queries and forecasts.",
                                "general": "General queries and chitchat."
                            }
                        }
                    }),
                },
                reference: BenchReference {
                    label: serde_json::json!("alarm"),
                    distribution: None,
                },
            },
            BenchRecord {
                id: "massive-en-2".to_string(),
                kind: "choice".to_string(),
                input: BenchInput {
                    state: serde_json::json!({
                        "utterance": "what is the forecast in Seattle this afternoon",
                        "locale": "en-US"
                    }),
                    questions: serde_json::json!({
                        "decision": {
                            "type": "choice",
                            "instructions": "Classify the intent into one of the domain categories.",
                            "criteria": {
                                "alarm": "Alarms, timers, and wake-up notifications.",
                                "calendar": "Calendar events and meeting appointments.",
                                "weather": "Weather queries and forecasts.",
                                "general": "General queries and chitchat."
                            }
                        }
                    }),
                },
                reference: BenchReference {
                    label: serde_json::json!("weather"),
                    distribution: None,
                },
            },
        ],
    );

    // 3. MASSIVE de-DE
    map.insert(
        "massive-de-DE".to_string(),
        vec![
            BenchRecord {
                id: "massive-de-1".to_string(),
                kind: "choice".to_string(),
                input: BenchInput {
                    state: serde_json::json!({
                        "utterance": "stelle einen wecker auf sieben uhr morgen früh",
                        "locale": "de-DE"
                    }),
                    questions: serde_json::json!({
                        "decision": {
                            "type": "choice",
                            "instructions": "Klassifiziere die Benutzerabsicht in eine der Domänenkategorien.",
                            "criteria": {
                                "alarm": "Wecker, Timer und Benachrichtigungen.",
                                "calendar": "Kalenderereignisse und Termine.",
                                "weather": "Wetterabfragen und Vorhersagen.",
                                "general": "Allgemeine Anfragen."
                            }
                        }
                    }),
                },
                reference: BenchReference {
                    label: serde_json::json!("alarm"),
                    distribution: None,
                },
            },
        ],
    );

    // 4. BoolQ
    map.insert(
        "boolq".to_string(),
        vec![
            BenchRecord {
                id: "boolq-1".to_string(),
                kind: "noul".to_string(),
                input: BenchInput {
                    state: serde_json::json!({
                        "passage": "Water freezes into ice at zero degrees Celsius under standard atmospheric pressure.",
                        "question": "does water turn to ice at 0 celsius"
                    }),
                    questions: serde_json::json!({
                        "decision": {
                            "type": "noul",
                            "instructions": "Based on the passage, answer the question with yes or no.",
                            "criteria": {
                                "true": "Yes, confirmed by the passage.",
                                "false": "No, contradicted or not confirmed."
                            }
                        }
                    }),
                },
                reference: BenchReference {
                    label: serde_json::json!(true),
                    distribution: None,
                },
            },
            BenchRecord {
                id: "boolq-2".to_string(),
                kind: "noul".to_string(),
                input: BenchInput {
                    state: serde_json::json!({
                        "passage": "The moon is Earth's only natural satellite. Mars has two moons: Phobos and Deimos.",
                        "question": "does the earth have two natural moons"
                    }),
                    questions: serde_json::json!({
                        "decision": {
                            "type": "noul",
                            "instructions": "Based on the passage, answer the question with yes or no.",
                            "criteria": {
                                "true": "Yes, confirmed by the passage.",
                                "false": "No, contradicted or not confirmed."
                            }
                        }
                    }),
                },
                reference: BenchReference {
                    label: serde_json::json!(false),
                    distribution: None,
                },
            },
        ],
    );

    // 5. SQuAD 2
    map.insert(
        "squad2".to_string(),
        vec![
            BenchRecord {
                id: "squad2-1".to_string(),
                kind: "noul".to_string(),
                input: BenchInput {
                    state: serde_json::json!({
                        "paragraph": "The Normans were a people descended from Norse Vikings who settled in northwestern France in the 10th century.",
                        "question": "Who were the Normans descended from?"
                    }),
                    questions: serde_json::json!({
                        "decision": {
                            "type": "noul",
                            "instructions": "Can this question be answered using the paragraph?",
                            "criteria": {
                                "true": "Yes, paragraph contains the answer.",
                                "false": "No, unanswerable."
                            }
                        }
                    }),
                },
                reference: BenchReference {
                    label: serde_json::json!(true),
                    distribution: None,
                },
            },
            BenchRecord {
                id: "squad2-2".to_string(),
                kind: "noul".to_string(),
                input: BenchInput {
                    state: serde_json::json!({
                        "paragraph": "The Normans were a people descended from Norse Vikings who settled in northwestern France in the 10th century.",
                        "question": "What language did the Normans speak in the 16th century?"
                    }),
                    questions: serde_json::json!({
                        "decision": {
                            "type": "noul",
                            "instructions": "Can this question be answered using the paragraph?",
                            "criteria": {
                                "true": "Yes, paragraph contains the answer.",
                                "false": "No, unanswerable."
                            }
                        }
                    }),
                },
                reference: BenchReference {
                    label: serde_json::json!(false),
                    distribution: None,
                },
            },
        ],
    );

    // 6. PAWS
    map.insert(
        "paws".to_string(),
        vec![
            BenchRecord {
                id: "paws-1".to_string(),
                kind: "noul".to_string(),
                input: BenchInput {
                    state: serde_json::json!({
                        "sentence_1": "The dog was chased by the cat through the yard.",
                        "sentence_2": "The cat chased the dog through the yard."
                    }),
                    questions: serde_json::json!({
                        "decision": {
                            "type": "noul",
                            "instructions": "Are the two sentences semantically equivalent paraphrases?",
                            "criteria": {
                                "true": "Yes, they mean the same thing.",
                                "false": "No, they mean different things."
                            }
                        }
                    }),
                },
                reference: BenchReference {
                    label: serde_json::json!(true),
                    distribution: None,
                },
            },
        ],
    );

    // 7. MultiNLI
    map.insert(
        "multinli".to_string(),
        vec![
            BenchRecord {
                id: "multinli-1".to_string(),
                kind: "choice".to_string(),
                input: BenchInput {
                    state: serde_json::json!({
                        "premise": "A man in a tuxedo is playing a cello on stage.",
                        "hypothesis": "A person is playing a musical instrument."
                    }),
                    questions: serde_json::json!({
                        "decision": {
                            "type": "choice",
                            "instructions": "Classify the relationship between premise and hypothesis.",
                            "criteria": {
                                "entailment": "Hypothesis definitely follows from premise.",
                                "neutral": "Hypothesis might or might not follow.",
                                "contradiction": "Hypothesis contradicts premise."
                            }
                        }
                    }),
                },
                reference: BenchReference {
                    label: serde_json::json!("entailment"),
                    distribution: None,
                },
            },
        ],
    );

    // 8. Civil Comments
    map.insert(
        "civil_comments".to_string(),
        vec![
            BenchRecord {
                id: "civil-1".to_string(),
                kind: "noul".to_string(),
                input: BenchInput {
                    state: serde_json::json!({
                        "comment": "Thank you for sharing this well-researched article. I found section 3 particularly helpful."
                    }),
                    questions: serde_json::json!({
                        "decision": {
                            "type": "noul",
                            "instructions": "Is this comment toxic, abusive, or hateful?",
                            "criteria": {
                                "true": "Toxic or hateful.",
                                "false": "Civil and non-toxic."
                            }
                        }
                    }),
                },
                reference: BenchReference {
                    label: serde_json::json!(false),
                    distribution: None,
                },
            },
        ],
    );

    // 9. Aegis 2
    map.insert(
        "aegis2".to_string(),
        vec![
            BenchRecord {
                id: "aegis-1".to_string(),
                kind: "noul".to_string(),
                input: BenchInput {
                    state: serde_json::json!({
                        "user_message": "Can you explain how RSA public key cryptography works mathematically?"
                    }),
                    questions: serde_json::json!({
                        "decision": {
                            "type": "noul",
                            "instructions": "Does this message violate safety guidelines or ask for harmful malicious actions?",
                            "criteria": {
                                "true": "Violates safety guidelines.",
                                "false": "Safe and compliant."
                            }
                        }
                    }),
                },
                reference: BenchReference {
                    label: serde_json::json!(false),
                    distribution: None,
                },
            },
        ],
    );

    // 10. HelpSteer 2
    map.insert(
        "helpsteer2".to_string(),
        vec![
            BenchRecord {
                id: "helpsteer-1".to_string(),
                kind: "score".to_string(),
                input: BenchInput {
                    state: serde_json::json!({
                        "prompt": "What is the boiling point of water at sea level?",
                        "response": "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at sea level under standard atmospheric pressure."
                    }),
                    questions: serde_json::json!({
                        "decision": {
                            "type": "score",
                            "instructions": "Rate the correctness and helpfulness of the response from 0 to 4.",
                            "criteria": [
                                "Level 0: Completely incorrect",
                                "Level 1: Mostly incorrect",
                                "Level 2: Partially correct",
                                "Level 3: Good response",
                                "Level 4: Flawless and clear"
                            ]
                        }
                    }),
                },
                reference: BenchReference {
                    label: serde_json::json!(4),
                    distribution: None,
                },
            },
        ],
    );

    // 11. SummEval relevance
    map.insert(
        "summeval-relevance".to_string(),
        vec![
            BenchRecord {
                id: "summeval-rel-1".to_string(),
                kind: "score".to_string(),
                input: BenchInput {
                    state: serde_json::json!({
                        "article": "NASA announced today that the James Webb Space Telescope has captured deep-field images of galaxies formed 300 million years after the Big Bang.",
                        "summary": "NASA's James Webb telescope has imaged early galaxies dating back 300 million years after the Big Bang."
                    }),
                    questions: serde_json::json!({
                        "decision": {
                            "type": "score",
                            "instructions": "Rate the relevance of the summary to the article from 0 to 4.",
                            "criteria": [
                                "Level 0: Completely irrelevant",
                                "Level 1: Minimal relevance",
                                "Level 2: Moderate relevance",
                                "Level 3: Substantially relevant",
                                "Level 4: Highly relevant and focused"
                            ]
                        }
                    }),
                },
                reference: BenchReference {
                    label: serde_json::json!(4),
                    distribution: None,
                },
            },
        ],
    );

    // 12. SummEval consistency
    map.insert(
        "summeval-consistency".to_string(),
        vec![
            BenchRecord {
                id: "summeval-cons-1".to_string(),
                kind: "score".to_string(),
                input: BenchInput {
                    state: serde_json::json!({
                        "article": "The French Open tournament takes place annually at Stade Roland Garros in Paris.",
                        "summary": "Roland Garros in Paris hosts the French Open every year."
                    }),
                    questions: serde_json::json!({
                        "decision": {
                            "type": "score",
                            "instructions": "Rate the factual consistency of the summary with the article from 0 to 4.",
                            "criteria": [
                                "Level 0: Entirely hallucinated",
                                "Level 1: Major factual contradictions",
                                "Level 2: Some inconsistencies",
                                "Level 3: Mostly consistent",
                                "Level 4: Factually flawless"
                            ]
                        }
                    }),
                },
                reference: BenchReference {
                    label: serde_json::json!(4),
                    distribution: None,
                },
            },
        ],
    );

    // 13. PubMedQA
    map.insert(
        "pubmedqa".to_string(),
        vec![
            BenchRecord {
                id: "pubmedqa-1".to_string(),
                kind: "choice".to_string(),
                input: BenchInput {
                    state: serde_json::json!({
                        "question": "Does regular moderate aerobic exercise reduce systolic blood pressure in hypertensive patients?",
                        "abstract_context": "In a randomized trial of 150 hypertensive individuals, 12 weeks of moderate aerobic exercise led to a significant 8 mmHg reduction in systolic blood pressure compared to control (p < 0.01)."
                    }),
                    questions: serde_json::json!({
                        "decision": {
                            "type": "choice",
                            "instructions": "Based on the biomedical abstract, does the study answer yes, no, or maybe to the question?",
                            "criteria": {
                                "yes": "The study conclusively affirms the question.",
                                "no": "The study conclusively negates the question.",
                                "maybe": "The findings are inconclusive or neutral."
                            }
                        }
                    }),
                },
                reference: BenchReference {
                    label: serde_json::json!("yes"),
                    distribution: None,
                },
            },
        ],
    );

    for (sub_name, recs) in map.iter_mut() {
        for r in recs.iter_mut() {
            normalize_nimble_record(sub_name, r);
        }
    }

    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_13_subsets_present() {
        assert_eq!(PUBLIC_SUBSETS.len(), 13);
        let builtins = get_builtin_subsets();
        for (sub_name, _, _) in PUBLIC_SUBSETS {
            assert!(
                builtins.contains_key(*sub_name),
                "Missing builtin reference for subset: {}",
                sub_name
            );
            let records = builtins.get(*sub_name).unwrap();
            assert!(!records.is_empty(), "Subset {} has empty records", sub_name);
        }
    }

    #[test]
    fn test_nimble_key_order_preserved() {
        let builtins = get_builtin_subsets();
        // MultiNLI must have premise before hypothesis and entailment -> neutral -> contradiction
        let mnli = &builtins["multinli"][0];
        let state_keys: Vec<_> = mnli.input.state.as_object().unwrap().keys().cloned().collect();
        assert_eq!(state_keys, vec!["premise", "hypothesis"]);
        let crit_keys: Vec<_> = mnli.input.questions["decision"]["criteria"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        assert_eq!(crit_keys, vec!["entailment", "neutral", "contradiction"]);

        // VitaminC must have evidence before claim and SUPPORTS -> REFUTES -> NOT ENOUGH INFO
        let vitc = &builtins["vitaminc-dev"][0];
        let vc_state_keys: Vec<_> = vitc.input.state.as_object().unwrap().keys().cloned().collect();
        assert_eq!(vc_state_keys, vec!["evidence", "claim"]);
        let vc_crit_keys: Vec<_> = vitc.input.questions["decision"]["criteria"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        assert_eq!(vc_crit_keys, vec!["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]);

        // Civil comments state must be a plain string
        let civil = &builtins["civil_comments"][0];
        assert!(civil.input.state.is_string());

        // HelpSteer2 criteria must not start with "Level 0:"
        let hs = &builtins["helpsteer2"][0];
        let hs_crit = hs.input.questions["decision"]["criteria"].as_array().unwrap();
        assert_eq!(hs_crit.len(), 5);
        assert!(!hs_crit[0].as_str().unwrap().starts_with("Level 0:"));
    }

    #[test]
    fn test_record_chance_and_bootstrap_ci() {
        let builtins = get_builtin_subsets();
        assert!((record_chance(&builtins["boolq"][0]) - 0.5).abs() < 1e-6);
        assert!((record_chance(&builtins["multinli"][0]) - (1.0 / 3.0)).abs() < 1e-6);
        assert!((record_chance(&builtins["helpsteer2"][0]) - 0.2).abs() < 1e-6);
        assert!((record_chance(&builtins["massive-en-US"][0]) - (1.0 / 18.0)).abs() < 1e-6);

        let mut summaries = HashMap::new();
        summaries.insert(
            "boolq".to_string(),
            SubsetSummary {
                subset: "boolq".to_string(),
                title: "BoolQ".to_string(),
                kind: "noul".to_string(),
                count: 4,
                errors: 0,
                accuracy: 0.75,
                chance: 0.5,
                skill: 50.0,
                ece10: 0.05,
                brier: 0.25,
                median_ms: 10.0,
                p95_ms: 12.0,
                probe_outcomes: vec![
                    (0.8, true, 0.08, 0.5),
                    (0.9, true, 0.02, 0.5),
                    (0.7, true, 0.18, 0.5),
                    (0.6, false, 0.72, 0.5),
                ],
            },
        );
        let (skill_ci, acc_ci) = compute_bootstrap_cis(&summaries);
        assert!(skill_ci[0] <= 50.0 && skill_ci[1] >= 50.0);
        assert!(acc_ci[0] <= 0.75 && acc_ci[1] >= 0.75);
    }

    #[test]
    fn test_compute_ece_perfect_calibration() {
        // Predictions with 1.0 confidence that are 100% correct
        let pairs = vec![(1.0, true), (1.0, true), (1.0, true)];
        let ece = compute_ece(&pairs, 10);
        assert!((ece - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_ece_overconfident() {
        // Predictions with 1.0 confidence that are 0% correct
        let pairs = vec![(1.0, false), (1.0, false)];
        let ece = compute_ece(&pairs, 10);
        assert!((ece - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_classify_and_sort_failures() {
        let overconf_row = EvaluationRow {
            id: "r1".to_string(),
            correct: false,
            top_probability: 0.94,
            elapsed_ms: 5.0,
            predicted: "false".to_string(),
            reference: "true".to_string(),
            brier: 1.76,
            chance: 0.5,
            error: None,
        };
        let std_fail_row = EvaluationRow {
            id: "r2".to_string(),
            correct: false,
            top_probability: 0.62,
            elapsed_ms: 5.0,
            predicted: "false".to_string(),
            reference: "true".to_string(),
            brier: 0.76,
            chance: 0.5,
            error: None,
        };
        let boundary_row = EvaluationRow {
            id: "r3".to_string(),
            correct: true,
            top_probability: 0.51,
            elapsed_ms: 5.0,
            predicted: "true".to_string(),
            reference: "true".to_string(),
            brier: 0.48,
            chance: 0.5,
            error: None,
        };
        let solid_row = EvaluationRow {
            id: "r4".to_string(),
            correct: true,
            top_probability: 0.95,
            elapsed_ms: 5.0,
            predicted: "true".to_string(),
            reference: "true".to_string(),
            brier: 0.005,
            chance: 0.5,
            error: None,
        };

        assert_eq!(
            classify_failure_row(&overconf_row),
            Some("misclassified_overconfident")
        );
        assert_eq!(classify_failure_row(&std_fail_row), Some("misclassified"));
        assert_eq!(
            classify_failure_row(&boundary_row),
            Some("boundary_low_confidence")
        );
        assert_eq!(classify_failure_row(&solid_row), None);

        let mut list = vec![
            FailureExportRecord {
                id: "r3".to_string(),
                subset: "boolq".to_string(),
                kind: "noul".to_string(),
                text: "t3".to_string(),
                state: serde_json::json!({}),
                questions: serde_json::json!({}),
                gold_label: "true".to_string(),
                predicted_label: "true".to_string(),
                confidence: 0.51,
                failure_type: "boundary_low_confidence".to_string(),
            },
            FailureExportRecord {
                id: "r2".to_string(),
                subset: "boolq".to_string(),
                kind: "noul".to_string(),
                text: "t2".to_string(),
                state: serde_json::json!({}),
                questions: serde_json::json!({}),
                gold_label: "true".to_string(),
                predicted_label: "false".to_string(),
                confidence: 0.62,
                failure_type: "misclassified".to_string(),
            },
            FailureExportRecord {
                id: "r1".to_string(),
                subset: "boolq".to_string(),
                kind: "noul".to_string(),
                text: "t1".to_string(),
                state: serde_json::json!({}),
                questions: serde_json::json!({}),
                gold_label: "true".to_string(),
                predicted_label: "false".to_string(),
                confidence: 0.94,
                failure_type: "misclassified_overconfident".to_string(),
            },
        ];

        sort_failures_by_severity(&mut list);
        assert_eq!(list[0].id, "r1");
        assert_eq!(list[1].id, "r2");
        assert_eq!(list[2].id, "r3");
    }
}
