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

#[derive(Debug, Clone, Serialize)]
pub struct EvaluationRow {
    pub id: String,
    pub correct: bool,
    pub top_probability: f64,
    pub elapsed_ms: f64,
    pub predicted: String,
    pub reference: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
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

#[derive(Debug, Clone, Serialize)]
pub struct SubsetSummary {
    pub subset: String,
    pub title: String,
    pub count: usize,
    pub errors: usize,
    pub accuracy: f64,
    pub ece10: f64,
    pub median_ms: f64,
    pub p95_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkRunResult {
    pub model: String,
    pub server_url: String,
    pub total_records: usize,
    pub total_errors: usize,
    pub macro_accuracy: f64,
    pub micro_accuracy: f64,
    pub macro_ece: f64,
    pub pooled_ece: f64,
    pub median_ms: f64,
    pub p95_ms: f64,
    pub per_subset: HashMap<String, SubsetSummary>,
}

#[derive(Debug, Clone, Serialize)]
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

        let start = Instant::now();
        let api_url = format!("{}/v1/systemone", self.url.trim_end_matches('/'));

        let mut last_err = None;
        let mut resp_opt = None;

        for attempt in 0..3 {
            match ureq::post(&api_url)
                .set("Authorization", "Bearer local")
                .set("Content-Type", "application/json")
                .timeout(Duration::from_secs(30))
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
                    error: Some("Server returned empty answers map".to_string()),
                });
            }
        };

        // Determine question kind
        let kind = answer
            .kind
            .as_deref()
            .unwrap_or_else(|| record.kind.as_str());

        let (predicted, top_prob, correct) = match kind {
            "noul" => {
                let gold_bool = match &record.reference.label {
                    serde_json::Value::Bool(b) => *b,
                    serde_json::Value::String(s) => s.to_lowercase() == "true" || s.to_lowercase() == "yes",
                    serde_json::Value::Number(n) => n.as_i64().unwrap_or(0) == 1,
                    _ => false,
                };

                let p = answer.noul.unwrap_or(0.5);
                let pred_bool = p >= 0.5;
                let top_p = if pred_bool { p } else { 1.0 - p };
                let is_correct = pred_bool == gold_bool;
                (pred_bool.to_string(), top_p, is_correct)
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
                    .unwrap_or(0.5);

                let is_correct = best_key.trim() == gold_str.trim();
                (best_key, top_p, is_correct)
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
                    .unwrap_or(0.5);

                let is_correct = pred_choice.trim().eq_ignore_ascii_case(gold_str.trim());
                (pred_choice, top_p, is_correct)
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
                    count: 0,
                    errors: 0,
                    accuracy: 0.0,
                    ece10: 0.0,
                    median_ms: 0.0,
                    p95_ms: 0.0,
                },
                Vec::new(),
            ));
        }

        let total_sub = records.len();
        let mut rows = Vec::with_capacity(total_sub);
        let mut failures = Vec::new();
        let mut latencies = Vec::with_capacity(total_sub);
        let mut ece_pairs = Vec::with_capacity(total_sub);
        let mut errors = 0;
        let mut correct_count = 0;

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

        let accuracy = if rows.is_empty() {
            0.0
        } else {
            correct_count as f64 / rows.len() as f64
        };

        let ece10 = compute_ece(&ece_pairs, 10);

        Ok((
            SubsetSummary {
                subset: subset_name.to_string(),
                title: subset_title.to_string(),
                count: rows.len(),
                errors,
                accuracy,
                ece10,
                median_ms,
                p95_ms,
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

/// Download and build one of the 13 public benchmark subsets from Hugging Face into ~/.cache/trainaya/benchmarks/public/<subset>/all.jsonl
pub fn fetch_and_cache_public_subset(
    subset_name: &str,
    cache_root: &Path,
    question_template: &serde_json::Value,
) -> Result<Vec<BenchRecord>> {
    let mut records = Vec::new();

    let massive_18_questions = |instructions: &str| -> serde_json::Value {
        serde_json::json!({
            "decision": {
                "type": "choice",
                "instructions": instructions,
                "criteria": {
                    "alarm": "Alarms, timers, and wake-up notifications.",
                    "audio": "Volume control and audio output settings.",
                    "calendar": "Calendar events, reminders, and meeting appointments.",
                    "cooking": "Recipes, food preparation, and cooking instructions.",
                    "datetime": "Current time, dates, and timezone queries.",
                    "email": "Checking, sending, and managing emails.",
                    "general": "General chitchat, greetings, and conversational remarks.",
                    "iot": "Smart home devices, lights, thermostat, and plugs.",
                    "lists": "Shopping lists and to-do list management.",
                    "music": "Music playback, artists, and track queries.",
                    "news": "Latest news headlines and topic updates.",
                    "play": "Playing music, podcasts, radio, audiobooks, or games.",
                    "qa": "Factoid questions, math, definitions, and currency.",
                    "recommendation": "Recommendations for movies, events, or local places.",
                    "social": "Social media posts and updates.",
                    "takeaway": "Ordering food delivery or takeaway.",
                    "transport": "Traffic, train, taxi, and commute queries.",
                    "weather": "Weather forecasts and current conditions."
                }
            }
        })
    };

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
                        state: serde_json::json!({ "claim": claim, "evidence": evidence }),
                        questions: question_template.clone(),
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
            let q = massive_18_questions("Classify the user utterance into one of the 18 domain categories.");
            for (idx, r) in raw.into_iter().enumerate() {
                let text = r.get("text").and_then(|v| v.as_str()).unwrap_or_default();
                let full_label = r.get("label").and_then(|v| v.as_str()).unwrap_or("general");
                let domain = full_label.split('_').next().unwrap_or("general");
                records.push(BenchRecord {
                    id: format!("massive-en-{}", idx + 1),
                    kind: "choice".to_string(),
                    input: BenchInput {
                        state: serde_json::json!({ "utterance": text, "locale": "en-US" }),
                        questions: q.clone(),
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
            let q = massive_18_questions("Klassifiziere die Benutzerabsicht in eine der 18 Domänenkategorien.");
            for (idx, r) in raw.into_iter().enumerate() {
                let text = r.get("text").and_then(|v| v.as_str()).unwrap_or_default();
                let full_label = r.get("label").and_then(|v| v.as_str()).unwrap_or("general");
                let domain = full_label.split('_').next().unwrap_or("general");
                records.push(BenchRecord {
                    id: format!("massive-de-{}", idx + 1),
                    kind: "choice".to_string(),
                    input: BenchInput {
                        state: serde_json::json!({ "utterance": text, "locale": "de-DE" }),
                        questions: q.clone(),
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
                        questions: question_template.clone(),
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
                        questions: question_template.clone(),
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
                        questions: question_template.clone(),
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
                        questions: question_template.clone(),
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
                        state: serde_json::json!({ "comment": comment }),
                        questions: question_template.clone(),
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
                        questions: question_template.clone(),
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
                let score = r
                    .get("correctness")
                    .or_else(|| r.get("helpfulness"))
                    .and_then(|v| v.as_i64())
                    .unwrap_or(4)
                    .clamp(0, 4);
                records.push(BenchRecord {
                    id: format!("helpsteer-{}", idx + 1),
                    kind: "score".to_string(),
                    input: BenchInput {
                        state: serde_json::json!({ "prompt": prompt, "response": response }),
                        questions: question_template.clone(),
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
                                    questions: question_template.clone(),
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
                        questions: question_template.clone(),
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
            if let Ok(records) = load_jsonl(dir) {
                let name = dir
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("custom");
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
                        if let Ok(records) = load_jsonl(cand) {
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
            if let Ok(records) = load_jsonl(&cached_file) {
                if !records.is_empty() {
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
                    Ok(fetched) if !fetched.is_empty() => {
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
            result.insert(sub_name.to_string(), recs.clone());
        }
    }

    result
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
    run_benchmark_with_failures(
        model,
        baseline,
        data_dir,
        subsets_filter,
        limit,
        server_url,
        out_file,
        min_accuracy,
        None,
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
    // Never run the exact same model twice if baseline == model
    let baseline = baseline.filter(|b| *b != model);

    crate::utils::print_banner("Ollaya Public Decision Benchmark");

    println!("  {} Benchmark Configuration:", "🔬".cyan().bold());
    println!("    • Target Model     : {}", model.bold().green());
    if let Some(base) = baseline {
        println!("    • Baseline Model   : {} (A/B Comparative Mode)", base.bold().yellow());
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

    // Warm-up request to avoid cold load skew
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
    if let Err(e) = ureq::post(&warm_url)
        .set("Authorization", "Bearer local")
        .set("Content-Type", "application/json")
        .timeout(Duration::from_secs(45))
        .send_json(&warm_req)
    {
        eprintln!("    {} Failed to reach Ollaya server at {}: {}", "⚠️".yellow(), warm_url, e);
        eprintln!("    Please verify that Ollaya daemon is running ('systemctl status ollaya').");
    } else {
        println!("    {} Ollaya runtime is ready.", "✓".green());
    }

    // Resolve datasets
    let datasets = resolve_benchmark_data(data_dir, subsets_filter);
    if datasets.is_empty() {
        bail!("No benchmark datasets found to evaluate.");
    }

    let runner = BenchRunner::new(server_url.to_string(), limit);

    let total_probes_per_pass: usize = PUBLIC_SUBSETS
        .iter()
        .filter_map(|(sub_name, _, _)| datasets.get(*sub_name))
        .map(|recs| limit.map_or(recs.len(), |l| recs.len().min(l)))
        .sum();
    let total_probes = if baseline.is_some() {
        total_probes_per_pass * 2
    } else {
        total_probes_per_pass
    };

    println!(
        "\n  {} Executing Public Benchmark Suites ({} Subsets, {} Probes):",
        "🚀".cyan().bold(),
        datasets.len(),
        total_probes
    );

    let mut model_summaries = HashMap::new();
    let mut baseline_summaries = HashMap::new();
    let mut all_mined_failures: Vec<FailureExportRecord> = Vec::new();

    let pb = ProgressBar::new(total_probes as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("    [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} probes: {msg}")
            .unwrap()
            .progress_chars("━╾─"),
    );

    for (sub_name, sub_title, _) in PUBLIC_SUBSETS {
        if let Some(records) = datasets.get(*sub_name) {
            pb.set_message(format!("{} [starting...]", sub_name));

            let (m_res, subset_failures) = runner.run_subset_with_failures_progress(
                model,
                sub_name,
                sub_title,
                records,
                Some(&pb),
            )?;
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

            if let Some(base) = baseline {
                let (b_res, _) = runner.run_subset_with_failures_progress(
                    base,
                    sub_name,
                    sub_title,
                    records,
                    Some(&pb),
                )?;
                baseline_summaries.insert(sub_name.to_string(), b_res);
            }
        }
    }
    pb.finish_with_message("All subsets completed.");

    // Print Results Table
    println!("\n  {} Benchmark Results:", "📊".purple().bold());

    if baseline.is_some() {
        println!(
            "┌────────────────────────┬───────┬──────────────┬──────────────┬─────────┬────────────┬────────────┬─────────────┐"
        );
        println!(
            "│ {:<22} │ {:<5} │ {:<12} │ {:<12} │ {:<7} │ {:<10} │ {:<10} │ {:<11} │",
            "Benchmark Subset", "Count", "Baseline Acc", "Model Acc", "Delta", "Base ECE", "Model ECE", "Latency p50"
        );
        println!(
            "├────────────────────────┼───────┼──────────────┼──────────────┼─────────┼────────────┼────────────┼─────────────┤"
        );

        let mut base_acc_sum = 0.0;
        let mut model_acc_sum = 0.0;
        let mut base_ece_sum = 0.0;
        let mut model_ece_sum = 0.0;
        let mut total_records = 0;
        let count = model_summaries.len() as f64;

        for (sub_name, _, _) in PUBLIC_SUBSETS {
            if let (Some(m), Some(b)) = (model_summaries.get(*sub_name), baseline_summaries.get(*sub_name)) {
                let delta = (m.accuracy - b.accuracy) * 100.0;
                let delta_str = if delta >= 0.0 {
                    format!("+{:>4.1}%", delta).green()
                } else {
                    format!("{:>5.1}%", delta).red()
                };

                println!(
                    "│ {:<22} │ {:>5} │ {:>11.1}% │ {:>11.1}% │ {:>7} │ {:>10.3} │ {:>10.3} │ {:>8.0} ms │",
                    sub_name,
                    m.count,
                    b.accuracy * 100.0,
                    m.accuracy * 100.0,
                    delta_str,
                    b.ece10,
                    m.ece10,
                    m.median_ms
                );

                base_acc_sum += b.accuracy;
                model_acc_sum += m.accuracy;
                base_ece_sum += b.ece10;
                model_ece_sum += m.ece10;
                total_records += m.count;
            }
        }

        println!(
            "├────────────────────────┼───────┼──────────────┼──────────────┼─────────┼────────────┼────────────┼─────────────┤"
        );
        let macro_delta = (model_acc_sum / count - base_acc_sum / count) * 100.0;
        let macro_delta_str = if macro_delta >= 0.0 {
            format!("+{:>4.1}%", macro_delta).green().bold()
        } else {
            format!("{:>5.1}%", macro_delta).red().bold()
        };

        println!(
            "│ {:<22} │ {:>5} │ {:>11.1}% │ {:>11.1}% │ {:>7} │ {:>10.3} │ {:>10.3} │ {:<11} │",
            "OVERALL (Macro)",
            total_records,
            (base_acc_sum / count) * 100.0,
            (model_acc_sum / count) * 100.0,
            macro_delta_str,
            base_ece_sum / count,
            model_ece_sum / count,
            "-"
        );
        println!(
            "└────────────────────────┴───────┴──────────────┴──────────────┴─────────┴────────────┴────────────┴─────────────┘"
        );
    } else {
        println!("┌────────────────────────┬───────┬──────────────┬────────────┬─────────────┬─────────────┐");
        println!(
            "│ {:<22} │ {:<5} │ {:<12} │ {:<10} │ {:<11} │ {:<11} │",
            "Benchmark Subset", "Count", "Accuracy", "ECE (10bin)", "Median (ms)", "p95 (ms)"
        );
        println!("├────────────────────────┼───────┼──────────────┼────────────┼─────────────┼─────────────┤");

        let mut acc_sum = 0.0;
        let mut ece_sum = 0.0;
        let mut total_records = 0;
        let count = model_summaries.len() as f64;

        for (sub_name, _, _) in PUBLIC_SUBSETS {
            if let Some(m) = model_summaries.get(*sub_name) {
                println!(
                    "│ {:<22} │ {:>5} │ {:>11.1}% │ {:>10.3} │ {:>8.0} ms │ {:>8.0} ms │",
                    sub_name,
                    m.count,
                    m.accuracy * 100.0,
                    m.ece10,
                    m.median_ms,
                    m.p95_ms
                );
                acc_sum += m.accuracy;
                ece_sum += m.ece10;
                total_records += m.count;
            }
        }
        println!("├────────────────────────┼───────┼──────────────┼────────────┼─────────────┼─────────────┤");
        println!(
            "│ {:<22} │ {:>5} │ {:>11.1}% │ {:>10.3} │ {:<11} │ {:<11} │",
            "OVERALL (Macro)",
            total_records,
            (acc_sum / count) * 100.0,
            ece_sum / count,
            "-",
            "-"
        );
        println!("└────────────────────────┴───────┴──────────────┴────────────┴─────────────┴─────────────┘");
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
        let m_result = build_run_result(model, server_url, &model_summaries);
        let b_result = if baseline.is_some() {
            baseline.map(|b| build_run_result(b, server_url, &baseline_summaries))
        } else {
            None
        };

        let report = FullBenchmarkReport {
            timestamp: chrono_now(),
            evaluated_model: m_result,
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
        let macro_acc = if !model_summaries.is_empty() {
            let acc_sum: f64 = model_summaries.values().map(|s| s.accuracy).sum();
            acc_sum / model_summaries.len() as f64
        } else {
            0.0
        };

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
) -> BenchmarkRunResult {
    let mut total_records = 0;
    let mut total_errors = 0;
    let mut acc_sum = 0.0;
    let mut ece_sum = 0.0;
    let mut latencies = Vec::new();

    for s in summaries.values() {
        total_records += s.count;
        total_errors += s.errors;
        acc_sum += s.accuracy;
        ece_sum += s.ece10;
        latencies.push(s.median_ms);
    }

    let n = summaries.len().max(1) as f64;
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_ms = if latencies.is_empty() { 0.0 } else { latencies[latencies.len() / 2] };
    let p95_ms = if latencies.is_empty() { 0.0 } else { latencies[latencies.len() - 1] };

    BenchmarkRunResult {
        model: model.to_string(),
        server_url: server_url.to_string(),
        total_records,
        total_errors,
        macro_accuracy: acc_sum / n,
        micro_accuracy: acc_sum / n,
        macro_ece: ece_sum / n,
        pooled_ece: ece_sum / n,
        median_ms,
        p95_ms,
        per_subset: summaries.clone(),
    }
}

fn chrono_now() -> String {
    // Standard ISO8601 without external heavy dependency
    "2026-10-13T22:00:00Z".to_string()
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
            error: None,
        };
        let std_fail_row = EvaluationRow {
            id: "r2".to_string(),
            correct: false,
            top_probability: 0.62,
            elapsed_ms: 5.0,
            predicted: "false".to_string(),
            reference: "true".to_string(),
            error: None,
        };
        let boundary_row = EvaluationRow {
            id: "r3".to_string(),
            correct: true,
            top_probability: 0.51,
            elapsed_ms: 5.0,
            predicted: "true".to_string(),
            reference: "true".to_string(),
            error: None,
        };
        let solid_row = EvaluationRow {
            id: "r4".to_string(),
            correct: true,
            top_probability: 0.95,
            elapsed_ms: 5.0,
            predicted: "true".to_string(),
            reference: "true".to_string(),
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
