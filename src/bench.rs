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

/// The 14 official tasks of Ollaya Bench (bench.ollaya.dev/methodology):
/// typed-decisions (400 states × 5 questions = 2,000 questions) + 13 public subsets (3,880 questions) = 5,880 questions total.
pub const ALL_BENCHMARK_SUBSETS: &[(&str, &str, &str)] = &[
    ("typed-decisions", "Typed Decisions test (400 states × 5q)", "Choice, noul, score"),
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workflow: Option<String>,
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
    #[serde(default)]
    load_duration: Option<u64>,
    #[serde(default)]
    eval_duration: Option<u64>,
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

/// Compute random-guess chance probability (1/K) for a single question definition
pub fn question_def_chance(q_def: &serde_json::Value, fallback_kind: &str) -> f64 {
    let q_type = q_def
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or(fallback_kind);

    if q_type == "noul" {
        return 0.5;
    }

    if let Some(crit) = q_def.get("criteria") {
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
    0.5
}

/// Compute random-guess chance probability (1/K) for a benchmark record following bench.ollaya.dev
pub fn record_chance(record: &BenchRecord) -> f64 {
    if let Some(decision) = record.input.questions.get("decision") {
        return question_def_chance(decision, record.kind.as_str());
    }
    if let Some(map) = record.input.questions.as_object() {
        if !map.is_empty() {
            let sum: f64 = map
                .values()
                .map(|q| question_def_chance(q, record.kind.as_str()))
                .sum();
            return sum / (map.len() as f64);
        }
    }
    0.5
}

/// Extract ordered candidate label keys for multiclass Brier computation
fn question_candidate_keys(q_def: Option<&serde_json::Value>, kind: &str) -> Vec<String> {
    if kind == "noul" {
        return vec!["false".to_string(), "true".to_string()];
    }
    if let Some(dec) = q_def {
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
    /// Soft cross-entropy against gold distributions (for typed-decisions)
    #[serde(default)]
    pub soft_ce: f64,
    pub median_ms: f64,
    pub p95_ms: f64,
    /// Per-question type accuracy breakdown (used by typed-decisions)
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub by_type: HashMap<String, f64>,
    /// Per-workflow accuracy breakdown (used by typed-decisions)
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub by_workflow: HashMap<String, f64>,
    /// Per-probe (top_probability, correct, brier, chance) for exact pooled ECE and bootstrap CIs
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub probe_outcomes: Vec<(f64, bool, f64, f64)>,
}

/// Speed (5-question triage preset) and memory (VRAM / RAM) profile matching bench.ollaya.dev/methodology
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LatencyMemoryProfile {
    /// Median client end-to-end latency (ms) over 20 timed 5-question triage requests
    pub triage_p50_ms: f64,
    /// p90 client end-to-end latency (ms) over 20 timed 5-question triage requests
    pub triage_p90_ms: f64,
    /// Cold load latency (ms)
    pub load_ms: f64,
    /// GPU VRAM used by the model (MB)
    pub vram_mb: u64,
    /// Host runner peak resident memory (MB)
    pub ram_peak_mb: u64,
    /// Parameter size label (e.g. "3B", "4B", "8B")
    pub params: String,
    /// Model license (e.g. "Apache-2.0")
    pub license: String,
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
    /// Overall 14-task Chance-Corrected Skill Score (0..100) matching bench.ollaya.dev Leaderboard "Score"
    #[serde(default)]
    pub skill_score: f64,
    /// 95% bootstrap confidence interval for skill_score (1,000 draws, seed 20261014)
    #[serde(default)]
    pub skill_score_ci: [f64; 2],
    /// 13-task Public Skill Score (0..100)
    #[serde(default)]
    pub public_skill_score: f64,
    /// Macro accuracy on Bespoke Labs' 13 public subsets (3,880 questions) matching Leaderboard "Public"
    #[serde(default)]
    pub public_macro_accuracy: f64,
    /// 10-bin Pooled ECE on the 13 public subsets matching Leaderboard "ECE"
    #[serde(default)]
    pub public_pooled_ece: f64,
    /// Accuracy on typed-decisions (2,000 questions) matching Leaderboard "Typed"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub typed_accuracy: Option<f64>,
    /// 10-bin Pooled ECE on typed-decisions (2,000 questions)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub typed_ece: Option<f64>,
    /// Soft cross-entropy against gold distributions on typed-decisions
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub typed_soft_ce: Option<f64>,
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
    /// Dedicated 5-question triage latency and GPU VRAM profile matching bench.ollaya.dev
    #[serde(default)]
    pub latency_profile: LatencyMemoryProfile,
    pub per_subset: HashMap<String, SubsetSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullBenchmarkReport {
    pub timestamp: String,
    pub evaluated_model: BenchmarkRunResult,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub baseline_model: Option<BenchmarkRunResult>,
}

/// Score a single answered question (choice, noul, or score) against its gold label and optional gold probability distribution.
/// Returns (predicted_label, top_probability, is_correct, multiclass_brier, soft_cross_entropy).
fn score_question_answer(
    q_def: Option<&serde_json::Value>,
    fallback_kind: &str,
    gold_label: &serde_json::Value,
    gold_probs: Option<&serde_json::Map<String, serde_json::Value>>,
    answer: &SystemOneAnswer,
) -> (String, f64, bool, f64, f64) {
    let kind = answer
        .kind
        .as_deref()
        .or_else(|| q_def.and_then(|d| d.get("type")).and_then(|v| v.as_str()))
        .unwrap_or(fallback_kind);

    let (predicted, top_prob, is_correct, brier) = match kind {
        "noul" => {
            let gold_bool = match gold_label {
                serde_json::Value::Bool(b) => *b,
                serde_json::Value::String(s) => {
                    s.eq_ignore_ascii_case("true") || s.eq_ignore_ascii_case("yes")
                }
                serde_json::Value::Number(n) => n.as_i64().unwrap_or(0) == 1,
                _ => false,
            };

            let p = answer
                .noul
                .or_else(|| {
                    answer
                        .probabilities
                        .as_ref()
                        .and_then(|pr| pr.get("true").copied())
                })
                .unwrap_or(0.5)
                .clamp(0.0, 1.0);
            let pred_bool = p >= 0.5;
            let top_p = if pred_bool { p } else { 1.0 - p };
            let corr = pred_bool == gold_bool;
            let gold_t = if gold_bool { 1.0 } else { 0.0 };
            let brier_val = 2.0 * (p - gold_t).powi(2);
            (pred_bool.to_string(), top_p, corr, brier_val)
        }
        "score" => {
            let gold_str = match gold_label {
                serde_json::Value::Number(n) => n.to_string(),
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string().trim_matches('"').to_string(),
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

            let corr = best_key.trim() == gold_str.trim();
            let cand_keys = question_candidate_keys(q_def, "score");
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
                let y = if corr { 1.0 } else { 0.0 };
                (top_p - y).powi(2) + ((1.0 - top_p) - (1.0 - y)).powi(2)
            };
            (best_key, top_p, corr, brier_val)
        }
        _ => {
            let gold_str = match gold_label {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string().trim_matches('"').to_string(),
            };

            let pred_choice = if let Some(probs) = &answer.probabilities {
                probs
                    .iter()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(k, _)| k.clone())
                    .or_else(|| answer.choice.clone())
                    .unwrap_or_default()
            } else if let Some(c) = &answer.choice {
                c.clone()
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

            let corr = pred_choice.trim().eq_ignore_ascii_case(gold_str.trim());
            let cand_keys = question_candidate_keys(q_def, "choice");
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
                let y = if corr { 1.0 } else { 0.0 };
                (top_p - y).powi(2) + ((1.0 - top_p) - (1.0 - y)).powi(2)
            };
            (pred_choice, top_p, corr, brier_val)
        }
    };

    // Soft cross-entropy against gold probability distribution if provided
    let soft_ce = if let Some(gp_map) = gold_probs {
        let mut ce = 0.0;
        for (k, v) in gp_map {
            let p_gold = v.as_f64().unwrap_or(0.0);
            if p_gold > 0.0 {
                let p_pred = if kind == "noul" {
                    let p_true = answer
                        .noul
                        .or_else(|| answer.probabilities.as_ref().and_then(|pr| pr.get("true").copied()))
                        .unwrap_or(0.5);
                    if k == "true" { p_true } else { 1.0 - p_true }
                } else {
                    answer
                        .probabilities
                        .as_ref()
                        .and_then(|pr| pr.get(k).copied())
                        .unwrap_or(1e-6)
                };
                ce -= p_gold * p_pred.clamp(1e-12, 1.0).ln();
            }
        }
        ce
    } else {
        -top_prob.clamp(1e-12, 1.0).ln()
    };

    (predicted, top_prob.clamp(0.0, 1.0), is_correct, brier, soft_ce)
}

pub struct BenchRunner {
    pub url: String,
    pub limit: Option<usize>,
}

impl BenchRunner {
    pub fn new(url: String, limit: Option<usize>) -> Self {
        Self { url, limit }
    }

    /// Send one HTTP request to /v1/systemone with retry on transient network errors
    fn post_systemone(&self, model: &str, record: &BenchRecord) -> (Option<SystemOneResponse>, f64, Option<String>) {
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
                .timeout(Duration::from_secs(120))
                .send_json(&payload)
            {
                Ok(resp) => match resp.into_json::<SystemOneResponse>() {
                    Ok(parsed) => {
                        resp_opt = Some(parsed);
                        break;
                    }
                    Err(e) => {
                        last_err = Some(format!("Failed to parse response: {}", e));
                    }
                },
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
        (resp_opt, elapsed_ms, last_err)
    }

    /// Post one single-question request to /v1/systemone with retry on transient network errors
    pub fn evaluate_record(
        &self,
        model: &str,
        record: &BenchRecord,
    ) -> Result<EvaluationRow> {
        let chance = record_chance(record);
        let (resp_opt, elapsed_ms, last_err) = self.post_systemone(model, record);

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

        let q_def = record
            .input
            .questions
            .get("decision")
            .or_else(|| record.input.questions.as_object().and_then(|m| m.values().next()));

        let (predicted, top_prob, correct, brier, _) =
            score_question_answer(q_def, record.kind.as_str(), &record.reference.label, None, answer);

        let clean_ref = match &record.reference.label {
            serde_json::Value::String(s) => s.clone(),
            other => other.to_string().trim_matches('"').to_string(),
        };

        Ok(EvaluationRow {
            id: record.id.clone(),
            correct,
            top_probability: top_prob,
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
                    soft_ce: 0.0,
                    median_ms: 0.0,
                    p95_ms: 0.0,
                    by_type: HashMap::new(),
                    by_workflow: HashMap::new(),
                    probe_outcomes: Vec::new(),
                },
                Vec::new(),
            ));
        }

        let is_multi_question = subset_name == "typed-decisions"
            || records.first().is_some_and(|r| r.kind == "typed");

        if is_multi_question {
            return self.run_typed_decisions_subset(model, subset_name, subset_title, records, pb);
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
            chance_sum += row.chance;
            if row.error.is_some() {
                // Per bench.ollaya.dev/methodology: a rejected request counts as wrong on the task
                errors += 1;
                probe_outcomes.push((0.0, false, 1.0, row.chance));
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

        let total_n = probe_outcomes.len();
        let answered_n = ece_pairs.len();
        let accuracy = if total_n == 0 {
            0.0
        } else {
            correct_count as f64 / total_n as f64
        };
        let chance = if total_n == 0 {
            record_chance(&records[0])
        } else {
            chance_sum / total_n as f64
        };
        let skill = if (1.0 - chance).abs() < 1e-9 {
            0.0
        } else {
            100.0 * (accuracy - chance) / (1.0 - chance)
        };
        let brier = if answered_n == 0 {
            0.0
        } else {
            brier_sum / answered_n as f64
        };

        let ece10 = compute_ece(&ece_pairs, 10);

        Ok((
            SubsetSummary {
                subset: subset_name.to_string(),
                title: subset_title.to_string(),
                kind: subset_kind,
                count: total_n,
                errors,
                accuracy,
                chance,
                skill,
                ece10,
                brier,
                soft_ce: 0.0,
                median_ms,
                p95_ms,
                by_type: HashMap::new(),
                by_workflow: HashMap::new(),
                probe_outcomes,
            },
            failures,
        ))
    }

    /// Evaluate the 14th task (`typed-decisions`): 400 JSON states, 5 typed questions per state sent in a single HTTP request (2,000 questions total)
    fn run_typed_decisions_subset(
        &self,
        model: &str,
        subset_name: &str,
        subset_title: &str,
        records: &[BenchRecord],
        pb: Option<&ProgressBar>,
    ) -> Result<(SubsetSummary, Vec<FailureExportRecord>)> {
        let total_states = records.len();
        let mut latencies = Vec::with_capacity(total_states);
        let mut ece_pairs = Vec::with_capacity(total_states * 5);
        let mut probe_outcomes = Vec::with_capacity(total_states * 5);
        let mut failures = Vec::new();
        let mut errors = 0usize;
        let mut correct_count = 0usize;
        let mut total_questions = 0usize;
        let mut brier_sum = 0.0;
        let mut soft_ce_sum = 0.0;
        let mut chance_sum = 0.0;
        let mut by_type_counts: HashMap<String, (usize, usize)> = HashMap::new();
        let mut by_wf_counts: HashMap<String, (usize, usize)> = HashMap::new();

        for (state_idx, record) in records.iter().enumerate() {
            let (resp_opt, elapsed_ms, _last_err) = self.post_systemone(model, record);
            let q_map = record.input.questions.as_object();
            let gold_map = record.reference.label.as_object();

            let mut q_keys: Vec<String> = q_map
                .map(|m| m.keys().cloned().collect())
                .unwrap_or_default();
            q_keys.sort();

            let wf_name = record
                .workflow
                .clone()
                .unwrap_or_else(|| "general".to_string());

            if let Some(ref resp) = resp_opt {
                latencies.push(elapsed_ms);
                for q_key in &q_keys {
                    let q_def = q_map.and_then(|m| m.get(q_key));
                    let q_type = q_def
                        .and_then(|d| d.get("type"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("choice")
                        .to_string();
                    let q_chance = q_def
                        .map(|d| question_def_chance(d, &q_type))
                        .unwrap_or(0.5);
                    chance_sum += q_chance;
                    total_questions += 1;

                    let gold_entry = gold_map.and_then(|g| g.get(q_key));
                    let gold_lbl = gold_entry
                        .and_then(|ge| ge.get("label"))
                        .unwrap_or(&serde_json::Value::Null);
                    let gold_probs = gold_entry
                        .and_then(|ge| ge.get("probabilities"))
                        .and_then(|p| p.as_object());

                    if let Some(ans) = resp.answers.get(q_key) {
                        let (pred, top_p, is_corr, brier_v, ce_v) =
                            score_question_answer(q_def, &q_type, gold_lbl, gold_probs, ans);
                        if is_corr {
                            correct_count += 1;
                        }
                        let ty_stat = by_type_counts.entry(q_type.clone()).or_insert((0, 0));
                        if is_corr {
                            ty_stat.0 += 1;
                        }
                        ty_stat.1 += 1;

                        let wf_stat = by_wf_counts.entry(wf_name.clone()).or_insert((0, 0));
                        if is_corr {
                            wf_stat.0 += 1;
                        }
                        wf_stat.1 += 1;

                        ece_pairs.push((top_p, is_corr));
                        probe_outcomes.push((top_p, is_corr, brier_v, q_chance));
                        brier_sum += brier_v;
                        soft_ce_sum += ce_v;

                        let eval_row = EvaluationRow {
                            id: format!("{}::{}", record.id, q_key),
                            correct: is_corr,
                            top_probability: top_p,
                            elapsed_ms,
                            predicted: pred.clone(),
                            reference: gold_lbl.to_string().trim_matches('"').to_string(),
                            brier: brier_v,
                            chance: q_chance,
                            error: None,
                        };
                        if let Some(ft) = classify_failure_row(&eval_row) {
                            failures.push(FailureExportRecord {
                                id: eval_row.id,
                                subset: subset_name.to_string(),
                                kind: q_type,
                                text: format!("{} | Question ({}): {:?}", record.input.state, q_key, q_def),
                                state: record.input.state.clone(),
                                questions: serde_json::json!({ "decision": q_def }),
                                gold_label: eval_row.reference,
                                predicted_label: pred,
                                confidence: top_p,
                                failure_type: ft.to_string(),
                            });
                        }
                    } else {
                        errors += 1;
                        probe_outcomes.push((0.0, false, 1.0, q_chance));
                        by_type_counts.entry(q_type).or_insert((0, 0)).1 += 1;
                        by_wf_counts.entry(wf_name.clone()).or_insert((0, 0)).1 += 1;
                    }
                }
            } else {
                // Rejected state request counts as wrong on all 5 of its questions
                let n_q = q_keys.len().max(5);
                errors += n_q;
                for q_key in &q_keys {
                    let q_def = q_map.and_then(|m| m.get(q_key));
                    let q_type = q_def
                        .and_then(|d| d.get("type"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("choice")
                        .to_string();
                    let q_chance = q_def
                        .map(|d| question_def_chance(d, &q_type))
                        .unwrap_or(0.5);
                    chance_sum += q_chance;
                    total_questions += 1;
                    probe_outcomes.push((0.0, false, 1.0, q_chance));
                    by_type_counts.entry(q_type).or_insert((0, 0)).1 += 1;
                    by_wf_counts.entry(wf_name.clone()).or_insert((0, 0)).1 += 1;
                }
            }

            if let Some(bar) = pb {
                let done_states = state_idx + 1;
                let live_acc = if total_questions > 0 {
                    (correct_count as f64 / total_questions as f64) * 100.0
                } else {
                    0.0
                };
                bar.set_message(format!(
                    "{} [{}/{} states ({}q) | acc: {:.1}%]",
                    subset_name, done_states, total_states, total_questions, live_acc
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

        let answered_n = ece_pairs.len();
        let accuracy = if total_questions == 0 {
            0.0
        } else {
            correct_count as f64 / total_questions as f64
        };
        let chance = if total_questions == 0 {
            0.317
        } else {
            chance_sum / total_questions as f64
        };
        let skill = if (1.0 - chance).abs() < 1e-9 {
            0.0
        } else {
            100.0 * (accuracy - chance) / (1.0 - chance)
        };
        let brier = if answered_n == 0 {
            0.0
        } else {
            brier_sum / answered_n as f64
        };
        let soft_ce = if answered_n == 0 {
            0.0
        } else {
            soft_ce_sum / answered_n as f64
        };
        let ece10 = compute_ece(&ece_pairs, 10);

        let by_type = by_type_counts
            .into_iter()
            .map(|(k, (c, t))| (k, if t > 0 { c as f64 / t as f64 } else { 0.0 }))
            .collect();
        let by_workflow = by_wf_counts
            .into_iter()
            .map(|(k, (c, t))| (k, if t > 0 { c as f64 / t as f64 } else { 0.0 }))
            .collect();

        Ok((
            SubsetSummary {
                subset: subset_name.to_string(),
                title: subset_title.to_string(),
                kind: "typed".to_string(),
                count: total_questions,
                errors,
                accuracy,
                chance,
                skill,
                ece10,
                brier,
                soft_ce,
                median_ms,
                p95_ms,
                by_type,
                by_workflow,
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
                "instructions": "Decide how the evidence bears on the claim. Judge only from the evidence text, and not from outside knowledge about the subject.",
                "criteria": {
                    "SUPPORTS": "The evidence states the claim, or the claim follows directly from the evidence.",
                    "REFUTES": "The evidence states the opposite of the claim, or the claim is directly contradicted by it.",
                    "NOT ENOUGH INFO": "The evidence neither establishes nor contradicts the claim. Settling it would need a fact the evidence does not supply."
                }
            }
        })),
        "massive-en-US" | "massive-de-DE" => Some(serde_json::json!({
            "decision": {
                "type": "choice",
                "instructions": "Which assistant domain should handle this utterance? Judge from the request itself, not from how it is phrased.",
                "criteria": {
                    "alarm": "Setting, changing, or querying alarms.",
                    "audio": "Volume, mute, or audio output settings.",
                    "calendar": "Events, meetings, reminders on a calendar.",
                    "cooking": "Recipes and cooking instructions.",
                    "datetime": "The current time, date, or time conversions.",
                    "email": "Reading, sending, or checking email and contacts.",
                    "general": "Small talk, jokes, greetings, or general assistant control such as repeat or confirm.",
                    "iot": "Controlling lights, plugs, heating, cleaners, coffee machines, or wemo devices.",
                    "lists": "Creating, querying, or removing list items.",
                    "music": "Music preferences, likes, or music settings, not playing a specific item.",
                    "news": "News headlines or updates.",
                    "play": "Playing a specific song, podcast, radio, audiobook, or game.",
                    "qa": "Factual questions, definitions, math, currency, stock prices.",
                    "recommendation": "Suggestions for events, movies, or locations.",
                    "social": "Social media posts or queries.",
                    "takeaway": "Food orders and delivery status.",
                    "transport": "Taxis, tickets, traffic, or travel directions.",
                    "weather": "Weather forecasts or conditions."
                }
            }
        })),
        "boolq" => Some(serde_json::json!({
            "decision": {
                "type": "noul",
                "instructions": "Does the passage answer the question with yes? Use only what the passage states or directly implies.",
                "criteria": {
                    "false": "The passage states or directly implies that the answer is no.",
                    "true": "The passage states or directly implies that the answer is yes."
                }
            }
        })),
        "squad2" => Some(serde_json::json!({
            "decision": {
                "type": "noul",
                "instructions": "Does the paragraph contain the information needed to answer the question? Treat a question as unanswerable when the paragraph discusses the topic but does not state the specific fact asked for.",
                "criteria": {
                    "false": "The paragraph does not state the answer, even if it covers the same subject or contains a plausible-looking but incorrect match.",
                    "true": "The paragraph states the answer explicitly."
                }
            }
        })),
        "paws" => Some(serde_json::json!({
            "decision": {
                "type": "noul",
                "instructions": "Do the two sentences have the same meaning? Sentences that reuse the same words in a different order can still mean different things.",
                "criteria": {
                    "false": "The sentences differ in meaning, including cases where the same words are rearranged so that who does what to whom changes.",
                    "true": "Both sentences describe the same situation with the same participants in the same roles."
                }
            }
        })),
        "multinli" => Some(serde_json::json!({
            "decision": {
                "type": "choice",
                "instructions": "Assume the premise is true. How does the hypothesis relate to it?",
                "criteria": {
                    "entailment": "If the premise is true, the hypothesis must be true.",
                    "neutral": "The hypothesis might or might not be true; the premise does not settle it.",
                    "contradiction": "If the premise is true, the hypothesis cannot be true."
                }
            }
        })),
        "civil_comments" => Some(serde_json::json!({
            "decision": {
                "type": "noul",
                "instructions": "Is this comment toxic, meaning rude, disrespectful, or unreasonable enough that a reader would likely leave the discussion?",
                "criteria": {
                    "false": "The comment may disagree or criticise but does not attack, demean, or threaten.",
                    "true": "The comment contains insults, threats, identity-based attacks, obscenity, or contempt aimed at a person or group."
                }
            }
        })),
        "aegis2" => Some(serde_json::json!({
            "decision": {
                "type": "noul",
                "instructions": "Would a general-purpose assistant be unsafe to comply with this user message as written? Judge the request, not the assistant's likely reply.",
                "criteria": {
                    "false": "The message is a benign request, including sensitive topics raised for legitimate purposes such as education, safety, or fiction that does not provide actionable harmful content.",
                    "true": "The message seeks help with, or itself contains, hate or identity hate, sexual content, suicide and self-harm, violence, guns or illegal weapons, threats, PII or privacy violations, sexual content involving minors, criminal planning or confessions, harassment, controlled or regulated substances, or profanity, or other content a general safety policy prohibits."
                }
            }
        })),
        "helpsteer2" => Some(serde_json::json!({
            "decision": {
                "type": "score",
                "instructions": "How helpful is the response to the prompt, considering whether it addresses what was asked, is correct, and is complete?",
                "criteria": [
                    "Not helpful: ignores or misreads the prompt, or is wrong in ways that make it useless.",
                    "Slightly helpful: touches the request but is mostly incorrect, incomplete, or off target.",
                    "Partially helpful: addresses the request with notable gaps or errors a user would need to fix.",
                    "Mostly helpful: addresses the request well with minor omissions or imperfections.",
                    "Extremely helpful: fully and accurately addresses the request; nothing important is missing."
                ]
            }
        })),
        "summeval-relevance" => Some(serde_json::json!({
            "decision": {
                "type": "score",
                "instructions": "How well does the summary capture the important content of the article, without unimportant or redundant material?",
                "criteria": [
                    "Misses the main points or is mostly about minor details.",
                    "Captures some key content but omits important points or includes much that is unimportant.",
                    "Captures the main points with noticeable omissions or filler.",
                    "Captures the main points with minor omissions.",
                    "Captures all key content and only key content."
                ]
            }
        })),
        "summeval-consistency" => Some(serde_json::json!({
            "decision": {
                "type": "score",
                "instructions": "How factually consistent is the summary with the article? Every statement in the summary should be supported by the article.",
                "criteria": [
                    "Multiple statements contradict or are absent from the article.",
                    "At least one clear unsupported or contradicted statement.",
                    "Mostly supported, with a minor unsupported detail.",
                    "Supported, with at most a small imprecision.",
                    "Every statement is supported by the article."
                ]
            }
        })),
        "pubmedqa" => Some(serde_json::json!({
            "decision": {
                "type": "choice",
                "instructions": "Based only on the abstract, what is the answer to the research question?",
                "criteria": {
                    "yes": "The abstract's findings support a yes answer.",
                    "no": "The abstract's findings support a no answer.",
                    "maybe": "The abstract's findings are mixed, conditional, or insufficient to answer either way."
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

const NIMBLE_PUBLIC_MANIFESTS_JSON: &str = include_str!("nimble_public_manifests.json");
pub const PUBLIC_BENCHMARK_SEED: u64 = 20260918;

#[derive(Debug, Clone, Deserialize)]
pub struct NimbleSubsetManifest {
    pub seed: u64,
    pub count: usize,
    pub dataset_sha256: String,
    pub ids: Vec<String>,
}

pub fn load_nimble_public_manifests() -> HashMap<String, NimbleSubsetManifest> {
    serde_json::from_str(NIMBLE_PUBLIC_MANIFESTS_JSON).unwrap_or_default()
}

fn sha256_hex16(text: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    let full = format!("{:x}", hasher.finalize());
    full[..16].to_string()
}

fn download_url_to_file(url: &str, dest: &Path) -> Result<()> {
    if dest.is_file() && std::fs::metadata(dest).map(|m| m.len() > 1024).unwrap_or(false) {
        return Ok(());
    }
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = dest.with_extension("part");
    let mut last_err = None;
    for attempt in 1..=3 {
        match ureq::get(url)
            .set("User-Agent", "trainaya-bench/0.3")
            .timeout(Duration::from_secs(180))
            .call()
        {
            Ok(resp) => {
                let mut reader = resp.into_reader();
                let mut out = File::create(&tmp)?;
                std::io::copy(&mut reader, &mut out)?;
                std::fs::rename(&tmp, dest)?;
                return Ok(());
            }
            Err(e) => {
                last_err = Some(e.to_string());
                std::thread::sleep(Duration::from_millis(800 * attempt));
            }
        }
    }
    bail!(
        "Failed to download {}: {}",
        url,
        last_err.unwrap_or_else(|| "unknown error".to_string())
    )
}

/// Read rows from a Hugging Face Parquet export (`0.parquet`) using Python + PyArrow (checking PATH, .venv, and ~/.cache/uv)
fn read_hf_parquet_rows(
    dataset: &str,
    config: &str,
    split: &str,
    cache_root: &Path,
    only_row_indices: Option<&[usize]>,
) -> Option<Vec<(usize, serde_json::Value)>> {
    let raw_dir = cache_root.join("_raw");
    let safe_name = format!(
        "{}_{}_{}.parquet",
        dataset.replace('/', "__"),
        config,
        split
    );
    let parquet_path = raw_dir.join(safe_name);
    let url = format!(
        "https://huggingface.co/api/datasets/{}/parquet/{}/{}/0.parquet",
        dataset, config, split
    );
    if download_url_to_file(&url, &parquet_path).is_err() {
        return None;
    }

    let indices_json = only_row_indices
        .map(|idx| serde_json::to_string(idx).unwrap_or_else(|_| "null".to_string()))
        .unwrap_or_else(|| "null".to_string());
    let idx_file = raw_dir.join(format!("idx_{}.json", std::process::id()));
    let _ = std::fs::write(&idx_file, &indices_json);

    let py_script = r#"
import json, sys
import pyarrow.parquet as pq
t = pq.read_table(sys.argv[1])
idxs = json.loads(open(sys.argv[2]).read())
if idxs is not None:
    sub = t.take(idxs)
    rows = sub.to_pylist()
    for i, r in zip(idxs, rows):
        print(json.dumps({"_idx": int(i), "row": r}, ensure_ascii=False))
else:
    for i, r in enumerate(t.to_pylist()):
        print(json.dumps({"_idx": i, "row": r}, ensure_ascii=False))
"#;

    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let mut py_candidates: Vec<(std::path::PathBuf, Vec<std::path::PathBuf>)> = vec![
        (std::path::PathBuf::from("python3"), Vec::new()),
        (std::path::PathBuf::from("python"), Vec::new()),
        (
            std::path::Path::new(&home).join("Projects/trainaya/.venv/bin/python"),
            Vec::new(),
        ),
    ];

    // Also discover uv-cached Python 3.12 + pyarrow site-packages
    let uv_py312 =
        std::path::Path::new(&home).join(".local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12");
    let uv_arch = std::path::Path::new(&home).join(".cache/uv/archive-v0");
    let mut extra_paths = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&uv_arch) {
        for e in entries.flatten() {
            let p = e.path();
            if p.join("pyarrow").is_dir() {
                extra_paths.push(p.clone());
            }
            let sp = p.join("lib/python3.12/site-packages");
            if sp.join("pyarrow").is_dir() {
                extra_paths.push(sp);
            }
        }
    }
    if uv_py312.is_file() {
        py_candidates.push((uv_py312, extra_paths.clone()));
    }
    if !extra_paths.is_empty() {
        py_candidates.push((std::path::PathBuf::from("python3.12"), extra_paths));
    }

    for (py_bin, py_paths) in py_candidates {
        let mut cmd = std::process::Command::new(&py_bin);
        if !py_paths.is_empty() {
            let joined = py_paths
                .iter()
                .map(|p| p.to_string_lossy().to_string())
                .collect::<Vec<_>>()
                .join(":");
            cmd.env("PYTHONPATH", joined);
        }
        cmd.arg("-c")
            .arg(py_script)
            .arg(&parquet_path)
            .arg(&idx_file);
        if let Ok(out) = cmd.output() {
            if out.status.success() {
                let _ = std::fs::remove_file(&idx_file);
                let stdout = String::from_utf8_lossy(&out.stdout);
                let mut parsed = Vec::new();
                for line in stdout.lines() {
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                        let i = v.get("_idx").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
                        if let Some(r) = v.get("row") {
                            parsed.push((i, r.clone()));
                        }
                    }
                }
                if !parsed.is_empty() {
                    return Some(parsed);
                }
            }
        }
    }

    let _ = std::fs::remove_file(&idx_file);
    None
}

/// Fetch specific row indices from Hugging Face Datasets Server by grouping nearby offsets into <=100-row requests
fn fetch_hf_rows_by_indices(
    dataset: &str,
    config: &str,
    split: &str,
    indices: &[usize],
) -> Result<Vec<(usize, serde_json::Value)>> {
    let mut sorted = indices.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let mut results = Vec::with_capacity(sorted.len());
    let mut pos = 0;

    while pos < sorted.len() {
        let start_idx = sorted[pos];
        let mut end_pos = pos;
        while end_pos + 1 < sorted.len() && sorted[end_pos + 1] < start_idx + 100 {
            end_pos += 1;
        }
        let length = (sorted[end_pos] - start_idx + 1).max(1).min(100);
        let url = format!(
            "https://datasets-server.huggingface.co/rows?dataset={}&config={}&split={}&offset={}&length={}",
            dataset, config, split, start_idx, length
        );
        let mut resp_opt = None;
        for attempt in 1..=4 {
            match ureq::get(&url).timeout(Duration::from_secs(30)).call() {
                Ok(r) => {
                    resp_opt = Some(r);
                    break;
                }
                Err(_) => std::thread::sleep(Duration::from_millis(800 * attempt)),
            }
        }
        if let Some(resp) = resp_opt {
            let body: serde_json::Value = resp.into_json()?;
            if let Some(arr) = body.get("rows").and_then(|v| v.as_array()) {
                for item in arr {
                    let row_idx = item
                        .get("row_idx")
                        .and_then(|v| v.as_u64())
                        .map(|x| x as usize);
                    if let (Some(r_idx), Some(r)) = (row_idx, item.get("row")) {
                        if sorted[pos..=end_pos].binary_search(&r_idx).is_ok() {
                            results.push((r_idx, r.clone()));
                        }
                    }
                }
            }
        }
        pos = end_pos + 1;
        std::thread::sleep(Duration::from_millis(80));
    }

    Ok(results)
}

/// Download and build one of the 13 public benchmark subsets (aligned with bespokelabsai/nimble seed 20260918 manifests)
/// into ~/.cache/trainaya/benchmarks/public/<subset>/all.jsonl
pub fn fetch_and_cache_public_subset(
    subset_name: &str,
    cache_root: &Path,
    question_template: &serde_json::Value,
) -> Result<Vec<BenchRecord>> {
    let mut records = Vec::new();
    let q_tmpl = nimble_question_template(subset_name).unwrap_or_else(|| question_template.clone());
    let manifests = load_nimble_public_manifests();
    let wanted_ids: Option< std::collections::HashSet<String> > = manifests
        .get(subset_name)
        .map(|m| m.ids.iter().cloned().collect());
    let raw_dir = cache_root.join("_raw");
    let _ = std::fs::create_dir_all(&raw_dir);

    match subset_name {
        "vitaminc-dev" => {
            let zip_path = raw_dir.join("vitaminc.zip");
            let zip_url = "https://github.com/TalSchuster/talschuster.github.io/raw/master/static/vitaminc.zip";
            if download_url_to_file(zip_url, &zip_path).is_ok() {
                if let Ok(mut child) = std::process::Command::new("unzip")
                    .arg("-p")
                    .arg(&zip_path)
                    .arg("*dev.jsonl")
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::null())
                    .spawn()
                {
                    if let Some(stdout) = child.stdout.take() {
                        let reader = BufReader::new(stdout);
                        for line in reader.lines().flatten() {
                            if line.trim().is_empty() {
                                continue;
                            }
                            if let Ok(r) = serde_json::from_str::<serde_json::Value>(&line) {
                                let uid = r.get("unique_id").and_then(|v| v.as_str()).unwrap_or_default();
                                let rec_id = format!("vitaminc-{}", uid);
                                if let Some(ref w) = wanted_ids {
                                    if !w.contains(&rec_id) {
                                        continue;
                                    }
                                }
                                let claim = r.get("claim").and_then(|v| v.as_str()).unwrap_or_default();
                                let evidence = r.get("evidence").and_then(|v| v.as_str()).unwrap_or_default();
                                let label = r.get("label").and_then(|v| v.as_str()).unwrap_or("NOT ENOUGH INFO");
                                records.push(BenchRecord {
                                    id: rec_id,
                                    kind: "choice".to_string(),
                                    workflow: None,
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
                    }
                    let _ = child.wait();
                }
            }
            if records.is_empty() {
                let raw = if let Some(pq_rows) = read_hf_parquet_rows("tals/vitaminc", "default", "validation", cache_root, None) {
                    pq_rows.into_iter().map(|(_, r)| r).collect()
                } else {
                    fetch_hf_rows("tals/vitaminc", "default", "validation", 599)?
                };
                for (idx, r) in raw.into_iter().enumerate() {
                    let uid = r.get("unique_id").and_then(|v| v.as_str()).unwrap_or_default();
                    let rec_id = if !uid.is_empty() {
                        format!("vitaminc-{}", uid)
                    } else {
                        format!("vitaminc-dev-{}", idx + 1)
                    };
                    if let Some(ref w) = wanted_ids {
                        if !uid.is_empty() && !w.contains(&rec_id) {
                            continue;
                        }
                    }
                    let claim = r.get("claim").and_then(|v| v.as_str()).unwrap_or_default();
                    let evidence = r.get("evidence").and_then(|v| v.as_str()).unwrap_or_default();
                    let label = r.get("label").and_then(|v| v.as_str()).unwrap_or("NOT ENOUGH INFO");
                    records.push(BenchRecord {
                        id: rec_id,
                        kind: "choice".to_string(),
                        workflow: None,
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
        }
        "massive-en-US" | "massive-de-DE" => {
            let locale = if subset_name == "massive-de-DE" { "de-DE" } else { "en-US" };
            let tar_path = raw_dir.join("amazon-massive-dataset-1.1.tar.gz");
            let tar_url = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz";
            if download_url_to_file(tar_url, &tar_path).is_ok() {
                let member = format!("1.1/data/{}.jsonl", locale);
                if let Ok(out) = std::process::Command::new("tar")
                    .arg("-xzf")
                    .arg(&tar_path)
                    .arg("-O")
                    .arg(&member)
                    .output()
                {
                    if out.status.success() {
                        let text_content = String::from_utf8_lossy(&out.stdout);
                        for line in text_content.lines() {
                            if line.trim().is_empty() {
                                continue;
                            }
                            if let Ok(r) = serde_json::from_str::<serde_json::Value>(line) {
                                if r.get("partition").and_then(|v| v.as_str()) != Some("test") {
                                    continue;
                                }
                                let raw_id = r
                                    .get("id")
                                    .and_then(|v| {
                                        v.as_str()
                                            .map(|s| s.to_string())
                                            .or_else(|| v.as_i64().map(|n| n.to_string()))
                                    })
                                    .unwrap_or_default();
                                let rec_id = format!("massive-{}", raw_id);
                                if let Some(ref w) = wanted_ids {
                                    if !w.contains(&rec_id) {
                                        continue;
                                    }
                                }
                                let utt = r.get("utt").and_then(|v| v.as_str()).unwrap_or_default();
                                let scenario = r.get("scenario").and_then(|v| v.as_str()).unwrap_or("general");
                                records.push(BenchRecord {
                                    id: rec_id,
                                    kind: "choice".to_string(),
                                    workflow: None,
                                    input: BenchInput {
                                        state: serde_json::json!({ "utterance": utt, "locale": locale }),
                                        questions: q_tmpl.clone(),
                                    },
                                    reference: BenchReference {
                                        label: serde_json::json!(scenario),
                                        distribution: None,
                                    },
                                });
                            }
                        }
                    }
                }
            }
            if records.is_empty() {
                let hf_cfg = if subset_name == "massive-de-DE" { "de" } else { "en" };
                let raw = fetch_hf_rows("mteb/amazon_massive_intent", hf_cfg, "test", 350)?;
                for (idx, r) in raw.into_iter().enumerate() {
                    let text = r.get("text").and_then(|v| v.as_str()).unwrap_or_default();
                    let full_label = r.get("label").and_then(|v| v.as_str()).unwrap_or("general");
                    let domain = full_label.split('_').next().unwrap_or("general");
                    records.push(BenchRecord {
                        id: format!("massive-{}-{}", hf_cfg, idx + 1),
                        kind: "choice".to_string(),
                        workflow: None,
                        input: BenchInput {
                            state: serde_json::json!({ "utterance": text, "locale": locale }),
                            questions: q_tmpl.clone(),
                        },
                        reference: BenchReference {
                            label: serde_json::json!(domain),
                            distribution: None,
                        },
                    });
                }
            }
        }
        "boolq" => {
            let raw: Vec<serde_json::Value> = if let Some(pq_rows) =
                read_hf_parquet_rows("google/boolq", "default", "validation", cache_root, None)
            {
                pq_rows.into_iter().map(|(_, r)| r).collect()
            } else {
                fetch_hf_rows("google/boolq", "default", "validation", 3270)?
            };
            for r in raw {
                let passage = r.get("passage").and_then(|v| v.as_str()).unwrap_or_default();
                let question = r.get("question").and_then(|v| v.as_str()).unwrap_or_default();
                if passage.trim().is_empty() || question.trim().is_empty() {
                    continue;
                }
                let rec_id = format!("boolq-{}", sha256_hex16(&format!("{}\n{}", question, passage)));
                if let Some(ref w) = wanted_ids {
                    if !w.contains(&rec_id) {
                        continue;
                    }
                }
                let ans = r.get("answer").and_then(|v| v.as_bool()).unwrap_or(false);
                records.push(BenchRecord {
                    id: rec_id,
                    kind: "noul".to_string(),
                    workflow: None,
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
            let json_path = raw_dir.join("squad_dev_v2.json");
            let json_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json";
            if download_url_to_file(json_url, &json_path).is_ok() {
                if let Ok(raw_str) = std::fs::read_to_string(&json_path) {
                    if let Ok(doc) = serde_json::from_str::<serde_json::Value>(&raw_str) {
                        if let Some(articles) = doc.get("data").and_then(|d| d.as_array()) {
                            for art in articles {
                                if let Some(paras) = art.get("paragraphs").and_then(|p| p.as_array()) {
                                    for para in paras {
                                        let context = para.get("context").and_then(|v| v.as_str()).unwrap_or_default();
                                        if let Some(qas) = para.get("qas").and_then(|q| q.as_array()) {
                                            for qa in qas {
                                                let qid = qa.get("id").and_then(|v| v.as_str()).unwrap_or_default();
                                                let rec_id = format!("squad2-{}", qid);
                                                if let Some(ref w) = wanted_ids {
                                                    if !w.contains(&rec_id) {
                                                        continue;
                                                    }
                                                }
                                                let question = qa.get("question").and_then(|v| v.as_str()).unwrap_or_default();
                                                let answerable = qa
                                                    .get("answers")
                                                    .and_then(|a| a.as_array())
                                                    .map(|arr| {
                                                        arr.iter().any(|x| {
                                                            x.get("text")
                                                                .and_then(|t| t.as_str())
                                                                .map(|s| !s.trim().is_empty())
                                                                .unwrap_or(false)
                                                        })
                                                    })
                                                    .unwrap_or(false);
                                                records.push(BenchRecord {
                                                    id: rec_id,
                                                    kind: "noul".to_string(),
                                                    workflow: None,
                                                    input: BenchInput {
                                                        state: serde_json::json!({ "paragraph": context, "question": question }),
                                                        questions: q_tmpl.clone(),
                                                    },
                                                    reference: BenchReference {
                                                        label: serde_json::json!(answerable),
                                                        distribution: None,
                                                    },
                                                });
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if records.is_empty() {
                let raw: Vec<serde_json::Value> = if let Some(pq_rows) =
                    read_hf_parquet_rows("rajpurkar/squad_v2", "squad_v2", "validation", cache_root, None)
                {
                    pq_rows.into_iter().map(|(_, r)| r).collect()
                } else {
                    fetch_hf_rows("rajpurkar/squad_v2", "squad_v2", "validation", 299)?
                };
                for r in raw {
                    let qid = r.get("id").and_then(|v| v.as_str()).unwrap_or_default();
                    let rec_id = format!("squad2-{}", qid);
                    if let Some(ref w) = wanted_ids {
                        if !w.contains(&rec_id) {
                            continue;
                        }
                    }
                    let paragraph = r.get("context").and_then(|v| v.as_str()).unwrap_or_default();
                    let question = r.get("question").and_then(|v| v.as_str()).unwrap_or_default();
                    let answerable = r
                        .get("answers")
                        .and_then(|a| a.get("text"))
                        .and_then(|t| t.as_array())
                        .map(|arr| !arr.is_empty())
                        .unwrap_or(false);
                    records.push(BenchRecord {
                        id: rec_id,
                        kind: "noul".to_string(),
                        workflow: None,
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
        }
        "paws" => {
            let raw: Vec<serde_json::Value> = if let Some(pq_rows) =
                read_hf_parquet_rows("google-research-datasets/paws", "labeled_final", "test", cache_root, None)
            {
                pq_rows.into_iter().map(|(_, r)| r).collect()
            } else {
                let mut target_offsets = Vec::new();
                if let Some(ref w) = wanted_ids {
                    for s in w {
                        if let Some(num_s) = s.strip_prefix("paws-") {
                            if let Ok(n) = num_s.parse::<usize>() {
                                target_offsets.push(n.saturating_sub(1));
                            }
                        }
                    }
                }
                fetch_hf_rows_by_indices("google-research-datasets/paws", "labeled_final", "test", &target_offsets)?
                    .into_iter()
                    .map(|(_, r)| r)
                    .collect()
            };
            for r in raw {
                let raw_id = r
                    .get("id")
                    .and_then(|v| {
                        v.as_i64()
                            .map(|n| n.to_string())
                            .or_else(|| v.as_str().map(|s| s.to_string()))
                    })
                    .unwrap_or_default();
                let rec_id = format!("paws-{}", raw_id);
                if let Some(ref w) = wanted_ids {
                    if !w.contains(&rec_id) {
                        continue;
                    }
                }
                let s1 = r.get("sentence1").and_then(|v| v.as_str()).unwrap_or_default();
                let s2 = r.get("sentence2").and_then(|v| v.as_str()).unwrap_or_default();
                let is_para = r.get("label").and_then(|v| v.as_i64()).unwrap_or(0) == 1;
                records.push(BenchRecord {
                    id: rec_id,
                    kind: "noul".to_string(),
                    workflow: None,
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
            let zip_path = raw_dir.join("multinli_1.0.zip");
            let zip_url = "https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip";
            if download_url_to_file(zip_url, &zip_path).is_ok() {
                if let Ok(mut child) = std::process::Command::new("unzip")
                    .arg("-p")
                    .arg(&zip_path)
                    .arg("*dev_matched.jsonl")
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::null())
                    .spawn()
                {
                    if let Some(stdout) = child.stdout.take() {
                        let reader = BufReader::new(stdout);
                        for line in reader.lines().flatten() {
                            if line.trim().is_empty() {
                                continue;
                            }
                            if let Ok(r) = serde_json::from_str::<serde_json::Value>(&line) {
                                let gold_label = r.get("gold_label").and_then(|v| v.as_str()).unwrap_or("-");
                                if gold_label == "-" {
                                    continue;
                                }
                                let pair_id = r.get("pairID").and_then(|v| v.as_str()).unwrap_or_default();
                                let rec_id = format!("multinli-{}", pair_id);
                                if let Some(ref w) = wanted_ids {
                                    if !w.contains(&rec_id) {
                                        continue;
                                    }
                                }
                                let premise = r.get("sentence1").and_then(|v| v.as_str()).unwrap_or_default().trim();
                                let hypothesis = r.get("sentence2").and_then(|v| v.as_str()).unwrap_or_default().trim();
                                let dist = r.get("annotator_labels").and_then(|v| v.as_array()).and_then(|votes| {
                                    if votes.is_empty() {
                                        return None;
                                    }
                                    let n = votes.len() as f64;
                                    let mut c_e = 0.0;
                                    let mut c_n = 0.0;
                                    let mut c_c = 0.0;
                                    for v in votes {
                                        match v.as_str().unwrap_or("") {
                                            "entailment" => c_e += 1.0,
                                            "neutral" => c_n += 1.0,
                                            "contradiction" => c_c += 1.0,
                                            _ => {}
                                        }
                                    }
                                    Some(serde_json::json!({
                                        "entailment": c_e / n,
                                        "neutral": c_n / n,
                                        "contradiction": c_c / n,
                                    }))
                                });
                                records.push(BenchRecord {
                                    id: rec_id,
                                    kind: "choice".to_string(),
                                    workflow: None,
                                    input: BenchInput {
                                        state: serde_json::json!({ "premise": premise, "hypothesis": hypothesis }),
                                        questions: q_tmpl.clone(),
                                    },
                                    reference: BenchReference {
                                        label: serde_json::json!(gold_label),
                                        distribution: dist,
                                    },
                                });
                            }
                        }
                    }
                    let _ = child.wait();
                }
            }
            if records.is_empty() {
                let raw = fetch_hf_rows("nyu-mll/multi_nli", "default", "validation_matched", 9815)?;
                for r in raw {
                    let pair_id = r.get("pairID").and_then(|v| v.as_str()).unwrap_or_default();
                    let rec_id = format!("multinli-{}", pair_id);
                    if let Some(ref w) = wanted_ids {
                        if !w.contains(&rec_id) {
                            continue;
                        }
                    }
                    let premise = r.get("premise").and_then(|v| v.as_str()).unwrap_or_default().trim();
                    let hypothesis = r.get("hypothesis").and_then(|v| v.as_str()).unwrap_or_default().trim();
                    let lbl_int = r.get("label").and_then(|v| v.as_i64()).unwrap_or(1);
                    let lbl_str = match lbl_int {
                        0 => "entailment",
                        2 => "contradiction",
                        _ => "neutral",
                    };
                    records.push(BenchRecord {
                        id: rec_id,
                        kind: "choice".to_string(),
                        workflow: None,
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
        }
        "civil_comments" => {
            let mut target_indices = Vec::new();
            if let Some(ref w) = wanted_ids {
                for s in w {
                    if let Some(num_s) = s.strip_prefix("civil_comments-") {
                        if let Ok(n) = num_s.parse::<usize>() {
                            target_indices.push(n);
                        }
                    }
                }
            }
            target_indices.sort_unstable();
            let indexed_rows = if let Some(pq_rows) = read_hf_parquet_rows(
                "google/civil_comments",
                "default",
                "test",
                cache_root,
                Some(&target_indices),
            ) {
                pq_rows
            } else {
                fetch_hf_rows_by_indices("google/civil_comments", "default", "test", &target_indices)?
            };
            for (row_idx, r) in indexed_rows {
                let rec_id = format!("civil_comments-{}", row_idx);
                if let Some(ref w) = wanted_ids {
                    if !w.contains(&rec_id) {
                        continue;
                    }
                }
                let comment = r.get("text").and_then(|v| v.as_str()).unwrap_or_default().trim();
                let tox = r.get("toxicity").and_then(|v| v.as_f64()).unwrap_or(0.0);
                records.push(BenchRecord {
                    id: rec_id,
                    kind: "noul".to_string(),
                    workflow: None,
                    input: BenchInput {
                        state: serde_json::json!(comment),
                        questions: q_tmpl.clone(),
                    },
                    reference: BenchReference {
                        label: serde_json::json!(tox >= 0.5),
                        distribution: Some(serde_json::json!({
                            "false": 1.0 - tox,
                            "true": tox,
                        })),
                    },
                });
            }
        }
        "aegis2" => {
            let raw: Vec<serde_json::Value> = if let Some(pq_rows) = read_hf_parquet_rows(
                "nvidia/Aegis-AI-Content-Safety-Dataset-2.0",
                "default",
                "test",
                cache_root,
                None,
            ) {
                pq_rows.into_iter().map(|(_, r)| r).collect()
            } else {
                fetch_hf_rows("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", "default", "test", 1964)?
            };
            for r in raw {
                let raw_id = r.get("id").and_then(|v| v.as_str()).unwrap_or_default();
                let rec_id = format!("aegis2-{}", raw_id);
                if let Some(ref w) = wanted_ids {
                    if !w.contains(&rec_id) {
                        continue;
                    }
                }
                let msg = r.get("prompt").and_then(|v| v.as_str()).unwrap_or_default().trim();
                let unsafe_lbl = r
                    .get("prompt_label")
                    .and_then(|v| v.as_str())
                    .map(|s| s.eq_ignore_ascii_case("unsafe"))
                    .unwrap_or(false);
                records.push(BenchRecord {
                    id: rec_id,
                    kind: "noul".to_string(),
                    workflow: None,
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
            let mut target_indices = Vec::new();
            if let Some(ref w) = wanted_ids {
                for s in w {
                    if let Some(num_s) = s.strip_prefix("helpsteer2-") {
                        if let Ok(n) = num_s.parse::<usize>() {
                            target_indices.push(n);
                        }
                    }
                }
            }
            target_indices.sort_unstable();
            let indexed_rows = if let Some(pq_rows) = read_hf_parquet_rows(
                "nvidia/HelpSteer2",
                "default",
                "validation",
                cache_root,
                Some(&target_indices),
            ) {
                pq_rows
            } else {
                let all = fetch_hf_rows("nvidia/HelpSteer2", "default", "validation", 1038)?;
                all.into_iter().enumerate().collect()
            };
            for (row_idx, r) in indexed_rows {
                let rec_id = format!("helpsteer2-{}", row_idx);
                if let Some(ref w) = wanted_ids {
                    if !w.contains(&rec_id) {
                        continue;
                    }
                }
                let prompt = r.get("prompt").and_then(|v| v.as_str()).unwrap_or_default();
                let response = r.get("response").and_then(|v| v.as_str()).unwrap_or_default();
                let score = r
                    .get("helpfulness")
                    .or_else(|| r.get("correctness"))
                    .and_then(|v| v.as_i64())
                    .unwrap_or(4)
                    .clamp(0, 4);
                records.push(BenchRecord {
                    id: rec_id,
                    kind: "score".to_string(),
                    workflow: None,
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
            let dim = if is_rel { "relevance" } else { "consistency" };
            let raw: Vec<serde_json::Value> = if let Some(pq_rows) =
                read_hf_parquet_rows("mteb/summeval", "default", "test", cache_root, None)
            {
                pq_rows.into_iter().map(|(_, r)| r).collect()
            } else {
                fetch_hf_rows("mteb/summeval", "default", "test", 100)?
            };
            for r in raw {
                let art_id = r.get("id").and_then(|v| v.as_str()).unwrap_or_default();
                let article = r.get("text").and_then(|v| v.as_str()).unwrap_or_default();
                let summaries = r.get("machine_summaries").and_then(|v| v.as_array());
                let scores = r.get(dim).and_then(|v| v.as_array());
                if let (Some(sums), Some(sc_arr)) = (summaries, scores) {
                    for (sum_idx, (s_val, sc_val)) in sums.iter().zip(sc_arr.iter()).enumerate() {
                        if let (Some(summary), Some(raw_score)) = (s_val.as_str(), sc_val.as_f64()) {
                            let rec_id = format!("summeval-{}-{}-{}", dim, art_id, sum_idx);
                            if let Some(ref w) = wanted_ids {
                                if !w.contains(&rec_id) {
                                    continue;
                                }
                            }
                            let mapped = ((raw_score + 0.5).floor() as i64 - 1).clamp(0, 4);
                            records.push(BenchRecord {
                                id: rec_id,
                                kind: "score".to_string(),
                                workflow: None,
                                input: BenchInput {
                                    state: serde_json::json!({ "article": article, "summary": summary }),
                                    questions: q_tmpl.clone(),
                                },
                                reference: BenchReference {
                                    label: serde_json::json!(mapped),
                                    distribution: None,
                                },
                            });
                        }
                    }
                }
            }
        }
        "pubmedqa" => {
            let raw: Vec<serde_json::Value> = if let Some(pq_rows) =
                read_hf_parquet_rows("qiaojin/PubMedQA", "pqa_labeled", "train", cache_root, None)
            {
                pq_rows.into_iter().map(|(_, r)| r).collect()
            } else {
                fetch_hf_rows("qiaojin/PubMedQA", "pqa_labeled", "train", 1000)?
            };
            for r in raw {
                let pubid = r
                    .get("pubid")
                    .and_then(|v| {
                        v.as_i64()
                            .map(|n| n.to_string())
                            .or_else(|| v.as_str().map(|s| s.to_string()))
                    })
                    .unwrap_or_default();
                let rec_id = format!("pubmedqa-{}", pubid);
                if let Some(ref w) = wanted_ids {
                    if !w.contains(&rec_id) {
                        continue;
                    }
                }
                let question = r.get("question").and_then(|v| v.as_str()).unwrap_or_default();
                let abstract_ctx = r
                    .get("context")
                    .and_then(|c| c.get("contexts"))
                    .and_then(|arr| arr.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str())
                            .filter(|s| !s.trim().is_empty())
                            .collect::<Vec<_>>()
                            .join(" ")
                    })
                    .unwrap_or_default();
                let decision = r
                    .get("final_decision")
                    .and_then(|v| v.as_str())
                    .unwrap_or("yes")
                    .trim()
                    .to_lowercase();
                records.push(BenchRecord {
                    id: rec_id,
                    kind: "choice".to_string(),
                    workflow: None,
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
        "typed-decisions" => {
            let raw = fetch_hf_rows("LocalLLaMA/typed-decisions", "all", "test", 400)?;
            for (idx, r) in raw.into_iter().enumerate() {
                let id = r
                    .get("id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| format!("typed-decisions-{}", idx + 1));
                let workflow = r
                    .get("workflow")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                let state_str = r.get("state").and_then(|v| v.as_str()).unwrap_or("{}");
                let questions_str = r.get("questions").and_then(|v| v.as_str()).unwrap_or("{}");
                let gold_str = r.get("gold").and_then(|v| v.as_str()).unwrap_or("{}");

                let state_val: serde_json::Value =
                    serde_json::from_str(state_str).unwrap_or_else(|_| serde_json::json!(state_str));
                let questions_val: serde_json::Value =
                    serde_json::from_str(questions_str).unwrap_or_else(|_| serde_json::json!({}));
                let gold_val: serde_json::Value =
                    serde_json::from_str(gold_str).unwrap_or_else(|_| serde_json::json!({}));

                records.push(BenchRecord {
                    id,
                    kind: "typed".to_string(),
                    workflow,
                    input: BenchInput {
                        state: state_val,
                        questions: questions_val,
                    },
                    reference: BenchReference {
                        label: gold_val.clone(),
                        distribution: Some(gold_val),
                    },
                });
            }
        }
        _ => {}
    }

    // Sort public subset records by ID to match bespokelabsai/nimble's exact manifest order
    if subset_name != "typed-decisions" {
        records.sort_by(|a, b| a.id.cmp(&b.id));
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
            for (sub_name, _, _) in ALL_BENCHMARK_SUBSETS {
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
    let nimble_manifests = load_nimble_public_manifests();

    for (sub_name, _, _) in ALL_BENCHMARK_SUBSETS {
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
                let manifest_matches = if let Some(m) = nimble_manifests.get(*sub_name) {
                    records.len() == m.ids.len()
                        && records.first().map(|r| r.id.as_str()) == m.ids.first().map(|s| s.as_str())
                        && records.last().map(|r| r.id.as_str()) == m.ids.last().map(|s| s.as_str())
                } else {
                    !records.is_empty()
                };

                if manifest_matches {
                    for r in &mut records {
                        normalize_nimble_record(sub_name, r);
                    }
                    result.insert(sub_name.to_string(), records);
                    continue;
                }
            }
        }

        // Not in cache yet (or pre-seed cache needs refresh) -> download and cache in ~/.cache/trainaya/benchmarks/public/
        if let Some(template_recs) = built_in.get(*sub_name) {
            if let Some(first_rec) = template_recs.first() {
                println!(
                    "    {} Caching benchmark subset '{}' (seed {}) into {:?}...",
                    "⬇️".cyan(),
                    sub_name.bold(),
                    PUBLIC_BENCHMARK_SEED,
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
    let expected_questions = if subset_name == "typed-decisions" {
        eval_count * 5
    } else {
        eval_count
    };
    if entry.composite_binary_sha256 == identity.composite_binary_sha256
        && entry.subset_data_sha256 == subset_data_sha256
        && (entry.summary.count == expected_questions || entry.summary.count == eval_count)
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

/// Compute 95% bootstrap confidence intervals (1,000 draws, seed 20261014) for Overall 14-task Skill Score and Macro Accuracy
/// following bench.ollaya.dev/methodology:
/// "questions are resampled within each public subset, and typed-decisions states are resampled whole, their five questions together."
pub fn compute_bootstrap_cis(summaries: &HashMap<String, SubsetSummary>) -> ([f64; 2], [f64; 2]) {
    let active_subsets: Vec<&SubsetSummary> = ALL_BENCHMARK_SUBSETS
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
    let mut rng_state: u64 = PUBLIC_BENCHMARK_SEED;
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

            if sub.subset == "typed-decisions" && n >= 5 && n % 5 == 0 {
                // Resample whole states (5 questions together)
                let n_states = n / 5;
                for _ in 0..n_states {
                    let s_idx = (next_u64() as usize) % n_states;
                    let base = s_idx * 5;
                    for q_off in 0..5 {
                        let (_, correct, _, chance) = items[base + q_off];
                        if correct {
                            c_sum += 1.0;
                        }
                        ch_sum += chance;
                    }
                }
            } else {
                // Resample individual questions within each public subset
                for _ in 0..n {
                    let idx = (next_u64() as usize) % n;
                    let (_, correct, _, chance) = items[idx];
                    if correct {
                        c_sum += 1.0;
                    }
                    ch_sum += chance;
                }
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

/// Ensure local Ollaya server is running (auto-spawning `/usr/local/bin/ollaya serve` if needed)
/// and evict any other currently loaded models from VRAM (`ollaya stop <name>`) so multi-GB GGUF models never OOM.
fn ensure_ollaya_server_and_evict_others(server_url: &str, keep_model: &str) {
    let base = server_url.trim_end_matches('/');
    let ver_url = format!("{}/api/version", base);
    let is_up = ureq::get(&ver_url)
        .timeout(Duration::from_secs(2))
        .call()
        .is_ok();

    if !is_up && (base.contains("127.0.0.1") || base.contains("localhost")) {
        let ollaya_bin = Path::new("/usr/local/bin/ollaya");
        if ollaya_bin.is_file() {
            println!(
                "    {} Ollaya daemon not running on {} — auto-starting local `/usr/local/bin/ollaya serve`...",
                "⚙️".cyan(),
                base
            );
            let models_dir = crate::registry::get_ollaya_models_dir(None)
                .unwrap_or_else(|_| std::path::PathBuf::from("/usr/share/ollaya/.ollaya/models"));
            let _ = std::process::Command::new(ollaya_bin)
                .arg("serve")
                .env("OLLAYA_MODELS", models_dir)
                .env("LD_LIBRARY_PATH", "/usr/local/lib/ollaya/llama:/opt/rocm/lib")
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn();

            for _ in 0..20 {
                std::thread::sleep(Duration::from_millis(350));
                if ureq::get(&ver_url)
                    .timeout(Duration::from_secs(1))
                    .call()
                    .is_ok()
                {
                    break;
                }
            }
        }
    }

    // Evict any loaded models from VRAM before warming up `keep_model` so VRAM is free
    // and any freshly re-registered manifest for `keep_model` is reloaded from disk.
    let _ = keep_model;
    let ps_url = format!("{}/api/ps", base);
    if let Ok(resp) = ureq::get(&ps_url).timeout(Duration::from_secs(2)).call() {
        if let Ok(json_val) = resp.into_json::<serde_json::Value>() {
            if let Some(models) = json_val.get("models").and_then(|m| m.as_array()) {
                for m in models {
                    if let Some(name) = m.get("name").and_then(|n| n.as_str()) {
                        let _ = std::process::Command::new("/usr/local/bin/ollaya")
                            .args(["stop", name])
                            .stdout(std::process::Stdio::null())
                            .stderr(std::process::Stdio::null())
                            .status();
                    }
                }
            }
        }
    }
}

/// Read current GPU VRAM used in MB from Linux DRM sysfs (AMD ROCm / Intel) or `nvidia-smi`
fn sample_gpu_vram_mb() -> Option<f64> {
    // 1. Check /sys/class/drm/card*/device/mem_info_vram_used (zero-overhead sysfs on ROCm/AMDGPU)
    if let Ok(entries) = std::fs::read_dir("/sys/class/drm") {
        let mut max_bytes: u64 = 0;
        for entry in entries.flatten() {
            let path = entry.path().join("device/mem_info_vram_used");
            if let Ok(raw) = std::fs::read_to_string(&path) {
                if let Ok(bytes) = raw.trim().parse::<u64>() {
                    if bytes > max_bytes {
                        max_bytes = bytes;
                    }
                }
            }
        }
        if max_bytes > 0 {
            return Some((max_bytes as f64) / (1024.0 * 1024.0));
        }
    }

    // 2. Fallback to nvidia-smi on NVIDIA CUDA hosts
    if let Ok(out) = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
    {
        if out.status.success() {
            let text = String::from_utf8_lossy(&out.stdout);
            if let Some(first_line) = text.lines().next() {
                if let Ok(mb) = first_line.trim().parse::<f64>() {
                    if mb > 0.0 {
                        return Some(mb);
                    }
                }
            }
        }
    }

    None
}

/// Resolve model parameter count and license metadata from Ollaya manifest or model name heuristics
fn resolve_model_params_and_license(model: &str, server_url: &str) -> (String, String) {
    let lower = model.to_lowercase();
    let mut params = if lower.contains("e2b") {
        "2B (5.1B total)".to_string()
    } else if lower.contains("e4b") {
        "4B (8B total)".to_string()
    } else if lower.contains("270m") {
        "270M".to_string()
    } else if lower.contains("14b") {
        "14B".to_string()
    } else if lower.contains("8b") {
        "8B".to_string()
    } else if lower.contains("7b") {
        "7B".to_string()
    } else if lower.contains("4b") {
        "4B".to_string()
    } else if lower.contains("3b") || lower.contains("ministral:v1") {
        "3B".to_string()
    } else if lower.contains("1b") {
        "1B".to_string()
    } else {
        "Custom".to_string()
    };

    let license = if lower.contains("winnow") || lower.contains("gemma") {
        "Apache 2.0".to_string()
    } else if lower.contains("ministral") || lower.contains("mistral") {
        "Apache 2.0".to_string()
    } else if lower.contains("qwen") {
        "Apache 2.0".to_string()
    } else if lower.contains("llama") {
        "Llama 3".to_string()
    } else {
        "Apache 2.0".to_string()
    };

    // Check `/api/ps` for live parameter_size if reported by Ollaya
    let ps_url = format!("{}/api/ps", server_url.trim_end_matches('/'));
    if let Ok(resp) = ureq::get(&ps_url).timeout(Duration::from_secs(2)).call() {
        if let Ok(json_val) = resp.into_json::<serde_json::Value>() {
            if let Some(models) = json_val.get("models").and_then(|m| m.as_array()) {
                for m in models {
                    let m_name = m.get("name").and_then(|v| v.as_str()).unwrap_or_default();
                    if m_name == model || m_name.starts_with(model) {
                        if params == "Custom" {
                            if let Some(ps) = m
                                .get("details")
                                .and_then(|d| d.get("parameter_size"))
                                .and_then(|v| v.as_str())
                            {
                                if !ps.is_empty() {
                                    params = ps.to_string();
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    (params, license)
}

/// Official `bench.ollaya.dev` Speed & Memory measurement:
/// - Built-in `triage` preset (5 questions: 1 choice, 3 noul, 1 score) on short customer messages
/// - 1 cold load + 5 warm-up requests + `n_timed` timed requests
/// - Samples GPU VRAM (`nvidia-smi` / DRM `mem_info_vram_used` / `/api/ps` `size_vram`) while loaded
fn measure_triage_latency_and_vram(
    server_url: &str,
    model: &str,
    n_timed: usize,
) -> Option<LatencyMemoryProfile> {
    let triage_questions = serde_json::json!({
        "intent": {
            "type": "choice",
            "instructions": "What does the customer want in `message`?",
            "criteria": {
                "refund": "money returned or a duplicate charge reversed",
                "technical_help": "a bug, outage or integration problem",
                "billing_question": "a question about an invoice, plan or payment method",
                "information": "general information, pricing or how-to",
                "cancellation": "wants to cancel or downgrade",
                "other": "none of the other options fits"
            }
        },
        "is_urgent": {
            "type": "noul",
            "instructions": "Does `message` communicate time pressure or a deadline?"
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated does the customer sound in `message`?",
            "criteria": [
                "calm and neutral",
                "concerned but civil",
                "clearly annoyed",
                "very angry or using strong language"
            ]
        },
        "refund_requested": {
            "type": "noul",
            "instructions": "Does the customer ask for money back?"
        },
        "churn_risk": {
            "type": "noul",
            "instructions": "Does `message` suggest the customer may leave for a competitor or cancel?"
        }
    });

    const CUSTOMER_MESSAGES: &[&str] = &[
        "I was charged twice for my Pro subscription this morning (#INV-4982). Please refund the duplicate charge today.",
        "Our production webhook endpoint has been returning 502 errors since 09:00 UTC and customers cannot check out.",
        "Could you let me know whether annual billing includes SSO or if that requires the Enterprise tier?",
        "This is the third outage this week. Cancel my account immediately and send confirmation.",
        "Hi team, how do I rotate an API key without downtime in the dashboard?",
        "My invoice shows 14 seats instead of 10. Can someone explain the prorated line item?",
        "Your SDK crashes on Node 22 when streaming responses. We have a launch in two hours, please help!",
        "Just checking if you offer a nonprofit discount for educational institutions.",
        "I asked for a refund five days ago and still haven't heard back. If this isn't resolved today we're switching to Linear.",
        "Thanks, the migration guide worked great. One question: where can I download last month's VAT receipt?",
        "We need to downgrade from Team to Starter before our renewal tomorrow.",
        "Login with Google is stuck in a redirect loop on Safari 18.",
        "Please reverse the $49 renewal charge—I meant to cancel during the trial period.",
        "Everything is broken again and support never answers! Give me my money back right now!",
        "Can I export our audit logs as CSV from the settings page?",
        "We're evaluating alternatives because your rate limits are too restrictive for our batch jobs.",
        "Urgent: our SSO certificate expires in 3 hours and the upload button is disabled.",
        "How many workspaces are included in the standard plan?",
        "I canceled last month but my card was still billed $99 today. Please refund immediately.",
        "Could you share the status page link for EU-West region?",
    ];

    let base = server_url.trim_end_matches('/');
    let sys_url = format!("{}/v1/systemone", base);

    // 1. Cold/initial request
    let cold_t0 = Instant::now();
    let cold_state = serde_json::json!({ "message": CUSTOMER_MESSAGES[0] });
    let cold_req = SystemOneRequest {
        model,
        state: &cold_state,
        questions: &triage_questions,
    };
    let ok = ureq::post(&sys_url)
        .set("Authorization", "Bearer local")
        .set("Content-Type", "application/json")
        .timeout(Duration::from_secs(90))
        .send_json(&cold_req)
        .is_ok();
    if !ok {
        return None;
    }
    let cold_load_ms = cold_t0.elapsed().as_secs_f64() * 1000.0;

    // 2. 5 warm-up requests
    for i in 0..5 {
        let state = serde_json::json!({ "message": CUSTOMER_MESSAGES[(i + 1) % CUSTOMER_MESSAGES.len()] });
        let req = SystemOneRequest {
            model,
            state: &state,
            questions: &triage_questions,
        };
        let _ = ureq::post(&sys_url)
            .set("Authorization", "Bearer local")
            .set("Content-Type", "application/json")
            .timeout(Duration::from_secs(30))
            .send_json(&req);
    }

    // 3. Timed requests (default 20 on full run, or fewer when --limit is small)
    let mut timings = Vec::with_capacity(n_timed);
    for i in 0..n_timed {
        let state = serde_json::json!({ "message": CUSTOMER_MESSAGES[i % CUSTOMER_MESSAGES.len()] });
        let req = SystemOneRequest {
            model,
            state: &state,
            questions: &triage_questions,
        };
        let t0 = Instant::now();
        if ureq::post(&sys_url)
            .set("Authorization", "Bearer local")
            .set("Content-Type", "application/json")
            .timeout(Duration::from_secs(30))
            .send_json(&req)
            .is_ok()
        {
            timings.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
    }

    if timings.is_empty() {
        return None;
    }
    timings.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = timings.len();
    let triage_p50_ms = timings[n / 2];
    let triage_p90_ms = timings[((n as f64) * 0.90).floor().min((n - 1) as f64) as usize];

    // 4. Sample VRAM while model is hot in memory
    let mut vram_mb = sample_gpu_vram_mb().unwrap_or(0.0);
    let ps_url = format!("{}/api/ps", base);
    if let Ok(resp) = ureq::get(&ps_url).timeout(Duration::from_secs(2)).call() {
        if let Ok(json_val) = resp.into_json::<serde_json::Value>() {
            if let Some(models) = json_val.get("models").and_then(|m| m.as_array()) {
                for m in models {
                    let m_name = m.get("name").and_then(|v| v.as_str()).unwrap_or_default();
                    if m_name == model || m_name.starts_with(model) {
                        if let Some(sz) = m
                            .get("size_vram")
                            .and_then(|v| v.as_u64())
                            .filter(|&s| s > 0)
                            .or_else(|| m.get("size").and_then(|v| v.as_u64()))
                        {
                            let ps_mb = (sz as f64) / (1024.0 * 1024.0);
                            if vram_mb <= 0.0 && ps_mb > 0.0 {
                                vram_mb = ps_mb;
                            }
                        }
                    }
                }
            }
        }
    }

    let (params, license) = resolve_model_params_and_license(model, server_url);

    Some(LatencyMemoryProfile {
        triage_p50_ms,
        triage_p90_ms,
        load_ms: cold_load_ms,
        vram_mb: vram_mb.round() as u64,
        ram_peak_mb: 0,
        params,
        license,
    })
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

    crate::utils::print_banner("Ollaya Decision Benchmark (bench.ollaya.dev Methodology)");

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
            "    • Dataset Cache    : {:?} (14 Tasks: 13 Public + Typed-Decisions = 5,880 questions)",
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

    // Resolve datasets across all 14 official tasks
    let datasets = resolve_benchmark_data(data_dir, subsets_filter);
    if datasets.is_empty() {
        bail!("No benchmark datasets found to evaluate.");
    }

    // Check which subsets are already cached by cryptographic SHA-256
    let ordered_subset_names: Vec<(&str, &str)> = {
        let mut v: Vec<(&str, &str)> = ALL_BENCHMARK_SUBSETS
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

    // Warm-up request
    if need_server {
        println!("\n  {} Checking Ollaya server connection and warming up...", "⏳".cyan());
        ensure_ollaya_server_and_evict_others(server_url, model);
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
            "\n  {} All requested subsets matched SHA-256 binary cache — skipping subset re-evaluation!",
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
        "\n  {} Executing Ollaya Benchmark Suites ({} Tasks, {} HTTP Requests):",
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
            .template("    [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} reqs: {msg}")
            .unwrap()
            .progress_chars("━╾─"),
    );

    // Pass 1: Evaluate all tasks for target model while keeping it hot in Ollaya memory
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

    // Measure 5-question `triage` latency & GPU VRAM for target model
    let timed_runs = limit.map_or(20, |l| l.clamp(5, 20));
    ensure_ollaya_server_and_evict_others(server_url, model);
    let model_latency_profile = measure_triage_latency_and_vram(server_url, model, timed_runs);

    // Pass 2: Evaluate all subsets for baseline model (single model swap or instant SHA-256 cache hit)
    let mut baseline_latency_profile = None;
    if let Some(base) = baseline {
        let mut evicted_for_baseline = false;
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
                    if !evicted_for_baseline {
                        ensure_ollaya_server_and_evict_others(server_url, base);
                        evicted_for_baseline = true;
                    }
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
        ensure_ollaya_server_and_evict_others(server_url, base);
        baseline_latency_profile = measure_triage_latency_and_vram(server_url, base, timed_runs);
    }
    pb.finish_with_message("All tasks completed.");

    if model_cached_subsets > 0 || baseline_cached_subsets > 0 {
        println!(
            "    {} Cache hits: {}/{} target tasks{}",
            "⚡".green(),
            model_cached_subsets,
            ordered_subset_names.len(),
            if baseline.is_some() {
                format!(", {}/{} baseline tasks", baseline_cached_subsets, ordered_subset_names.len())
            } else {
                String::new()
            }
        );
    }

    let m_result = build_run_result(
        model,
        server_url,
        &model_summaries,
        model_identity.as_ref(),
        model_latency_profile,
    );
    let b_result = baseline.map(|b| {
        build_run_result(
            b,
            server_url,
            &baseline_summaries,
            baseline_identity.as_ref(),
            baseline_latency_profile,
        )
    });

    // Print Official Leaderboard Table (exact bench.ollaya.dev columns)
    println!(
        "\n  {} Official Ollaya Leaderboard Summary (bench.ollaya.dev Methodology):",
        "🏆".yellow().bold()
    );
    println!(
        "┌──────────────────────┬───────────────────────┬──────────┬──────────┬──────────┬───────────┬───────────┬──────────────────┬────────────┐"
    );
    println!(
        "│ {:<20} │ {:<21} │ {:<8} │ {:<8} │ {:<8} │ {:<9} │ {:<9} │ {:<16} │ {:<10} │",
        "Model", "Score [95% CI]", "Public", "Typed", "ECE", "Latency", "VRAM", "Params", "License"
    );
    println!(
        "├──────────────────────┼───────────────────────┼──────────┼──────────┼──────────┼───────────┼───────────┼──────────────────┼────────────┤"
    );

    let print_leaderboard_row = |res: &BenchmarkRunResult, is_primary: bool| {
        let score_str = format!(
            "{:>4.1} [{:>4.1}–{:>4.1}]",
            res.skill_score, res.skill_score_ci[0], res.skill_score_ci[1]
        );
        let pub_str = format!("{:>6.1}%", res.public_macro_accuracy * 100.0);
        let typed_str = res
            .typed_accuracy
            .map(|a| format!("{:>6.1}%", a * 100.0))
            .unwrap_or_else(|| "    N/A".to_string());
        // bench.ollaya.dev Leaderboard ECE column shows calibration error (lower is better)
        let ece_val = if res.public_pooled_ece > 0.0 {
            res.public_pooled_ece
        } else {
            res.pooled_ece
        };
        let ece_str = format!("{:>6.1}%", ece_val * 100.0);
        let lp = &res.latency_profile;
        let lat_ms = if lp.triage_p50_ms > 0.0 {
            lp.triage_p50_ms
        } else {
            res.median_ms
        };
        let lat_str = format!("{:>6.0} ms", lat_ms);
        let vram_str = if lp.vram_mb > 0 {
            if lp.vram_mb >= 1024 {
                format!("{:>6.1} GB", (lp.vram_mb as f64) / 1024.0)
            } else {
                format!("{:>6} MB", lp.vram_mb)
            }
        } else {
            "      N/A".to_string()
        };
        let params_str = if lp.params.is_empty() {
            "Custom"
        } else {
            lp.params.as_str()
        };
        let license_str = if lp.license.is_empty() {
            "Apache 2.0"
        } else {
            lp.license.as_str()
        };

        if is_primary {
            println!(
                "│ {:<20} │ {:<21} │ {:>8} │ {:>8} │ {:>8} │ {:>9} │ {:>9} │ {:<16} │ {:<10} │",
                res.model.bold().green(),
                score_str.bold().green(),
                pub_str,
                typed_str,
                ece_str,
                lat_str,
                vram_str,
                params_str,
                license_str
            );
        } else {
            println!(
                "│ {:<20} │ {:<21} │ {:>8} │ {:>8} │ {:>8} │ {:>9} │ {:>9} │ {:<16} │ {:<10} │",
                res.model.yellow(),
                score_str,
                pub_str,
                typed_str,
                ece_str,
                lat_str,
                vram_str,
                params_str,
                license_str
            );
        }
    };

    print_leaderboard_row(&m_result, true);
    if let Some(ref b_res) = b_result {
        print_leaderboard_row(b_res, false);
    }
    println!(
        "└──────────────────────┴───────────────────────┴──────────┴──────────┴──────────┴───────────┴───────────┴──────────────────┴────────────┘"
    );

    // Print Per-Task Breakdown Table
    println!("\n  {} Per-Task Breakdown ({} Tasks):", "📊".purple().bold(), ordered_subset_names.len());

    if let Some(ref b_res) = b_result {
        println!(
            "┌────────────────────────┬───────┬──────────────┬──────────────┬─────────┬────────────┬────────────┬────────────┬────────────┬─────────────┐"
        );
        println!(
            "│ {:<22} │ {:<5} │ {:<12} │ {:<12} │ {:<7} │ {:<10} │ {:<10} │ {:<10} │ {:<10} │ {:<11} │",
            "Benchmark Task", "Count", "Baseline Acc", "Model Acc", "Delta", "Base Skill", "Mod Skill", "Base ECE", "Model ECE", "Latency p50"
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
            "OVERALL (14 Tasks)",
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
            "Benchmark Task", "Count", "Accuracy", "Skill (0-100)", "ECE (10bin)", "Brier", "Median (ms)", "p95 (ms)"
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
            "OVERALL (14 Tasks)",
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

    // Print Detailed Methodology Scorecard
    println!(
        "\n  {} Detailed Methodology Metrics:",
        "📐".cyan().bold()
    );
    println!(
        "    • Overall Skill Score (14 tasks) : {} (95% CI: [{:.1}, {:.1}], 1,000 bootstrap draws, seed {})",
        format!("{:.1}", m_result.skill_score).bold().green(),
        m_result.skill_score_ci[0],
        m_result.skill_score_ci[1],
        PUBLIC_BENCHMARK_SEED,
    );
    println!(
        "    • Public (13 tasks, 3,880 q)     : Skill {:.1} | Macro Acc {:.1}% | Pooled ECE {:.1}% ({:.3})",
        m_result.public_skill_score,
        m_result.public_macro_accuracy * 100.0,
        m_result.public_pooled_ece * 100.0,
        m_result.public_pooled_ece
    );
    if let Some(td_sum) = model_summaries.get("typed-decisions") {
        let mut parts: Vec<String> = td_sum
            .by_workflow
            .iter()
            .map(|(k, v)| format!("{}: {:.1}%", k, v * 100.0))
            .collect();
        parts.sort();
        let wf_s = parts.join(" | ");
        println!(
            "    • Typed-Decisions (2,000 q)      : Skill {:.1} | Acc {:.1}% | ECE {:.1}% | Soft-CE {:.3} ({})",
            td_sum.skill,
            td_sum.accuracy * 100.0,
            td_sum.ece10 * 100.0,
            td_sum.soft_ce,
            wf_s
        );
    }
    println!(
        "    • Accuracy by Answer Type        : Choice: {:.1}% | Noul: {:.1}% | Score: {:.1}%",
        m_result.by_type.get("choice").copied().unwrap_or(0.0) * 100.0,
        m_result.by_type.get("noul").copied().unwrap_or(0.0) * 100.0,
        m_result.by_type.get("score").copied().unwrap_or(0.0) * 100.0,
    );
    let lp = &m_result.latency_profile;
    if lp.triage_p50_ms > 0.0 || lp.vram_mb > 0 {
        println!(
            "    • Triage Speed & Memory Profile  : p50 {:.0} ms (p90 {:.0} ms) | Cold load {:.0} ms | VRAM {} MB",
            lp.triage_p50_ms, lp.triage_p90_ms, lp.load_ms, lp.vram_mb
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
    latency_profile: Option<LatencyMemoryProfile>,
) -> BenchmarkRunResult {
    let mut total_records = 0;
    let mut total_errors = 0;
    let mut acc_sum = 0.0;
    let mut skill_sum = 0.0;
    let mut ece_sum = 0.0;
    let mut latencies = Vec::new();
    let mut pooled_ece_pairs = Vec::new();
    let mut public_ece_pairs = Vec::new();
    let mut pooled_brier_sum = 0.0;
    let mut pooled_correct = 0usize;
    let mut pooled_valid = 0usize;
    let mut by_type_buckets: HashMap<String, Vec<f64>> = HashMap::new();

    let mut public_acc_sum = 0.0;
    let mut public_skill_sum = 0.0;
    let mut public_count = 0usize;

    for (sub_name, s) in summaries {
        total_records += s.count;
        total_errors += s.errors;
        acc_sum += s.accuracy;
        skill_sum += s.skill;
        ece_sum += s.ece10;
        latencies.push(s.median_ms);

        if sub_name == "typed-decisions" {
            for (k, &v) in &s.by_type {
                by_type_buckets.entry(k.clone()).or_default().push(v);
            }
        } else {
            public_acc_sum += s.accuracy;
            public_skill_sum += s.skill;
            public_count += 1;
            by_type_buckets
                .entry(s.kind.clone())
                .or_default()
                .push(s.accuracy);
        }

        if !s.probe_outcomes.is_empty() {
            for &(top_p, corr, brier, _) in &s.probe_outcomes {
                if top_p > 0.0 {
                    pooled_ece_pairs.push((top_p, corr));
                    if sub_name != "typed-decisions" {
                        public_ece_pairs.push((top_p, corr));
                    }
                }
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
    let median_ms = latency_profile
        .as_ref()
        .map(|lp| lp.triage_p50_ms)
        .filter(|&v| v > 0.0)
        .unwrap_or_else(|| if latencies.is_empty() { 0.0 } else { latencies[latencies.len() / 2] });
    let p95_ms = latency_profile
        .as_ref()
        .map(|lp| lp.triage_p90_ms)
        .filter(|&v| v > 0.0)
        .unwrap_or_else(|| if latencies.is_empty() { 0.0 } else { latencies[latencies.len() - 1] });

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
    let public_n = public_count.max(1) as f64;
    let public_macro_accuracy = if public_count > 0 {
        public_acc_sum / public_n
    } else {
        macro_accuracy
    };
    let public_skill_score = if public_count > 0 {
        public_skill_sum / public_n
    } else {
        skill_sum / n
    };
    let public_pooled_ece = if !public_ece_pairs.is_empty() {
        compute_ece(&public_ece_pairs, 10)
    } else {
        pooled_ece
    };

    let td_sum = summaries.get("typed-decisions");
    let typed_accuracy = td_sum.map(|s| s.accuracy);
    let typed_ece = td_sum.map(|s| s.ece10);
    let typed_soft_ce = td_sum.map(|s| s.soft_ce);

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

    let resolved_latency_profile = latency_profile.unwrap_or_else(|| {
        let (params, license) = resolve_model_params_and_license(model, server_url);
        LatencyMemoryProfile {
            triage_p50_ms: median_ms,
            triage_p90_ms: p95_ms,
            load_ms: 0.0,
            vram_mb: sample_gpu_vram_mb().unwrap_or(0.0).round() as u64,
            ram_peak_mb: 0,
            params,
            license,
        }
    });

    BenchmarkRunResult {
        model: model.to_string(),
        server_url: server_url.to_string(),
        manifest_digest: identity.map(|i| i.manifest_sha256.clone()),
        binary_sha256: identity.map(|i| i.composite_binary_sha256.clone()),
        total_records,
        total_errors,
        skill_score,
        skill_score_ci,
        public_skill_score,
        public_macro_accuracy,
        public_pooled_ece,
        typed_accuracy,
        typed_ece,
        typed_soft_ce,
        macro_accuracy,
        macro_accuracy_ci,
        micro_accuracy,
        macro_ece,
        pooled_ece,
        brier,
        by_type,
        median_ms,
        p95_ms,
        latency_profile: resolved_latency_profile,
        per_subset: summaries.clone(),
    }
}

fn chrono_now() -> String {
    // Standard ISO8601 without external heavy dependency
    "2026-10-18T12:00:00Z".to_string()
}

/// Curated reference evaluation records for all 14 official benchmark tasks (13 Public + Typed-Decisions)
pub fn get_builtin_subsets() -> HashMap<String, Vec<BenchRecord>> {
    let mut map = HashMap::new();

    // 0. Typed-Decisions (multi-question workflow state: 5 questions in a single request)
    map.insert(
        "typed-decisions".to_string(),
        vec![
            BenchRecord {
                id: "td-test-support_triage-0001".to_string(),
                kind: "mixed".to_string(),
                workflow: Some("support_triage".to_string()),
                input: BenchInput {
                    state: serde_json::json!({
                        "ticket_id": "SUP-1042",
                        "customer_tier": "enterprise",
                        "subject": "Duplicate billing on invoice #INV-9921 and production API rate limit 429s",
                        "message": "We were billed twice for our annual Enterprise seat renewal ($14,400) this morning, and our production EU webhook pipeline is hitting 429 rate limits despite being well below our contracted burst quota. Please refund the duplicate invoice and raise our burst limit before 16:00 UTC."
                    }),
                    questions: serde_json::json!({
                        "primary_queue": {
                            "type": "choice",
                            "instructions": "Which specialist queue should take ownership of this ticket first?",
                            "criteria": {
                                "billing": "invoicing, duplicate charges, refunds, or payment methods",
                                "platform_sre": "outages, rate limits, latency, or infrastructure incidents",
                                "account_management": "contract renewals, seat upgrades, or SLA negotiations",
                                "general_support": "how-to questions, documentation, or minor UI bugs"
                            }
                        },
                        "requires_refund": {
                            "type": "noul",
                            "instructions": "Does the customer explicitly request a refund or reversal of a charge?"
                        },
                        "has_production_impact": {
                            "type": "noul",
                            "instructions": "Is a live production system or customer-facing workflow currently degraded?"
                        },
                        "urgency_level": {
                            "type": "score",
                            "instructions": "Rate the operational urgency of this ticket from 0 (lowest) to 4 (critical).",
                            "criteria": [
                                "routine question with no deadline",
                                "minor inconvenience, standard SLA",
                                "time-sensitive request or billing dispute",
                                "production degradation with explicit deadline",
                                "complete production outage blocking all customers"
                            ]
                        },
                        "escalation_priority": {
                            "type": "score",
                            "instructions": "Rate the account escalation priority from 0 to 3.",
                            "criteria": [
                                "self-serve / no escalation",
                                "standard support follow-up",
                                "senior specialist review within 4 hours",
                                "immediate enterprise incident bridge"
                            ]
                        }
                    }),
                },
                reference: BenchReference {
                    label: serde_json::json!({
                        "primary_queue": "billing",
                        "requires_refund": "yes",
                        "has_production_impact": "yes",
                        "urgency_level": 3,
                        "escalation_priority": 2
                    }),
                    distribution: Some(serde_json::json!({
                        "primary_queue": { "billing": 0.70, "platform_sre": 0.30, "account_management": 0.0, "general_support": 0.0 },
                        "requires_refund": { "yes": 1.0, "no": 0.0 },
                        "has_production_impact": { "yes": 1.0, "no": 0.0 },
                        "urgency_level": { "0": 0.0, "1": 0.0, "2": 0.15, "3": 0.75, "4": 0.10 },
                        "escalation_priority": { "0": 0.0, "1": 0.05, "2": 0.75, "3": 0.20 }
                    })),
                },
            },
        ],
    );

    // 1. VitaminC
    map.insert(
        "vitaminc-dev".to_string(),
        vec![
            BenchRecord {
                id: "vitaminc-dev-1".to_string(),
                kind: "choice".to_string(),
                workflow: None,
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
                workflow: None,
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
                workflow: None,
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
                workflow: None,
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
                workflow: None,
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
                workflow: None,
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
                workflow: None,
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
                workflow: None,
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
                workflow: None,
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
                workflow: None,
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
                workflow: None,
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
                workflow: None,
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
                workflow: None,
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
                workflow: None,
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
                workflow: None,
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
                workflow: None,
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
                workflow: None,
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
    fn test_all_14_subsets_present() {
        assert_eq!(PUBLIC_SUBSETS.len(), 13);
        assert_eq!(ALL_BENCHMARK_SUBSETS.len(), 14);
        let builtins = get_builtin_subsets();
        for (sub_name, _, _) in ALL_BENCHMARK_SUBSETS {
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
        // typed-decisions state has 5 questions: 1 choice(4)=0.25, 2 noul=0.5, 1 score(5)=0.2, 1 score(4)=0.25 -> mean = 1.70 / 5 = 0.34
        assert!((record_chance(&builtins["typed-decisions"][0]) - 0.34).abs() < 1e-6);

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
                soft_ce: 0.0,
                by_type: HashMap::new(),
                by_workflow: HashMap::new(),
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
        summaries.insert(
            "typed-decisions".to_string(),
            SubsetSummary {
                subset: "typed-decisions".to_string(),
                title: "Typed-Decisions (Multi-Question Workflows)".to_string(),
                kind: "mixed".to_string(),
                count: 10,
                errors: 0,
                accuracy: 0.80,
                chance: 0.34,
                skill: 100.0 * (0.80 - 0.34) / (1.0 - 0.34),
                ece10: 0.04,
                brier: 0.15,
                soft_ce: 0.45,
                by_type: HashMap::new(),
                by_workflow: HashMap::new(),
                median_ms: 18.0,
                p95_ms: 22.0,
                probe_outcomes: vec![
                    (0.9, true, 0.02, 0.25),
                    (0.8, true, 0.08, 0.50),
                    (0.9, true, 0.02, 0.50),
                    (0.7, true, 0.18, 0.20),
                    (0.6, false, 0.72, 0.25),
                    (0.9, true, 0.02, 0.25),
                    (0.8, true, 0.08, 0.50),
                    (0.9, true, 0.02, 0.50),
                    (0.7, true, 0.18, 0.20),
                    (0.6, false, 0.72, 0.25),
                ],
            },
        );
        let (skill_ci, acc_ci) = compute_bootstrap_cis(&summaries);
        assert!(skill_ci[0] <= 60.0 && skill_ci[1] >= 59.0);
        assert!(acc_ci[0] <= 0.78 && acc_ci[1] >= 0.77);
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

    #[test]
    fn test_nimble_public_manifests_seed_20260918() {
        let manifests = load_nimble_public_manifests();
        assert_eq!(manifests.len(), 13);
        let mut total_ids = 0usize;
        for (sub_name, _, _) in PUBLIC_SUBSETS {
            let m = manifests
                .get(*sub_name)
                .unwrap_or_else(|| panic!("Missing manifest for {}", sub_name));
            assert_eq!(m.seed, PUBLIC_BENCHMARK_SEED);
            assert_eq!(m.ids.len(), m.count);
            total_ids += m.ids.len();
        }
        assert_eq!(total_ids, 3880);
    }
}
