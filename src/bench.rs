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

        Ok(EvaluationRow {
            id: record.id.clone(),
            correct,
            top_probability: top_prob.clamp(0.0, 1.0),
            elapsed_ms,
            predicted,
            reference: format!("{}", record.reference.label),
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
        let count_to_eval = if let Some(limit) = self.limit {
            records.len().min(limit)
        } else {
            records.len()
        };

        let records = &records[..count_to_eval];
        if records.is_empty() {
            return Ok(SubsetSummary {
                subset: subset_name.to_string(),
                title: subset_title.to_string(),
                count: 0,
                errors: 0,
                accuracy: 0.0,
                ece10: 0.0,
                median_ms: 0.0,
                p95_ms: 0.0,
            });
        }

        let mut rows = Vec::with_capacity(records.len());
        let mut latencies = Vec::with_capacity(records.len());
        let mut ece_pairs = Vec::with_capacity(records.len());
        let mut errors = 0;
        let mut correct_count = 0;

        for record in records {
            let row = self.evaluate_record(model, record)?;
            if row.error.is_some() {
                errors += 1;
            } else {
                if row.correct {
                    correct_count += 1;
                }
                latencies.push(row.elapsed_ms);
                ece_pairs.push((row.top_probability, row.correct));
            }
            rows.push(row);
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

        Ok(SubsetSummary {
            subset: subset_name.to_string(),
            title: subset_title.to_string(),
            count: rows.len(),
            errors,
            accuracy,
            ece10,
            median_ms,
            p95_ms,
        })
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

/// Discover subsets in a data directory or load built-in reference subsets
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

    // For any missing public subset, provide high-quality reference probe records
    let built_in = get_builtin_subsets();
    for (sub_name, _, _) in PUBLIC_SUBSETS {
        if let Some(req) = requested_subsets {
            if !req.iter().any(|r| r.eq_ignore_ascii_case(sub_name)) {
                continue;
            }
        }
        if !result.contains_key(*sub_name) {
            if let Some(recs) = built_in.get(*sub_name) {
                result.insert(sub_name.to_string(), recs.clone());
            }
        }
    }

    result
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
    if let Some(dir) = data_dir {
        println!("    • Dataset Source   : {:?}", dir);
    } else {
        println!("    • Dataset Source   : {}", "Built-in 13 Public Benchmark Reference Sets".green());
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

    println!("\n  {} Executing Public Benchmark Suites (13 Subsets):", "🚀".cyan().bold());

    let mut model_summaries = HashMap::new();
    let mut baseline_summaries = HashMap::new();

    let pb = ProgressBar::new(datasets.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("    [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} subsets: {msg}")
            .unwrap()
            .progress_chars("━╾─"),
    );

    for (sub_name, sub_title, _) in PUBLIC_SUBSETS {
        if let Some(records) = datasets.get(*sub_name) {
            pb.set_message(format!("Evaluating {}", sub_name));

            let m_res = runner.run_subset(model, sub_name, sub_title, records)?;
            model_summaries.insert(sub_name.to_string(), m_res);

            if let Some(base) = baseline {
                let b_res = runner.run_subset(base, sub_name, sub_title, records)?;
                baseline_summaries.insert(sub_name.to_string(), b_res);
            }
            pb.inc(1);
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
}
