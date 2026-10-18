//! Benchfile parsing and declarative configuration for Ollaya decision benchmarks.
//!
//! ```text
//! # Example Benchfile
//! MODEL winnow:v{latest}
//! BASELINE winnow:e4b
//!
//! SUBSETS multinli, paws, boolq, vitaminc-dev
//! # DATA ./benchmarks/public
//!
//! URL http://127.0.0.1:11435
//! LIMIT 50
//!
//! OUTPUT ./results/bench_{model}_vs_{baseline}.json
//! THRESHOLD min_accuracy 0.70
//! ```

use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct Benchfile {
    /// Target model to benchmark (MODEL)
    pub model: String,
    /// Optional baseline model for A/B comparison (BASELINE)
    pub baseline: Option<String>,
    /// Subsets to evaluate (SUBSETS)
    pub subsets: Option<Vec<String>>,
    /// Custom benchmark data path (DATA)
    pub data: Option<PathBuf>,
    /// Server URL hosting /v1/systemone (URL)
    pub url: String,
    /// Sample limit per subset (LIMIT)
    pub limit: Option<usize>,
    /// Output report path template (OUTPUT)
    pub output: Option<PathBuf>,
    /// Minimum macro accuracy required (THRESHOLD min_accuracy)
    pub min_accuracy: Option<f64>,
    /// Path to export misclassified and overconfident failure records for active learning (EXPORT_FAILURES)
    pub export_failures: Option<PathBuf>,
    /// Bypass reading cached SHA-256 benchmark results (NO_CACHE)
    pub no_cache: bool,
}

impl Default for Benchfile {
    fn default() -> Self {
        Self {
            model: String::new(),
            baseline: None,
            subsets: None,
            data: None,
            url: "http://127.0.0.1:11435".to_string(),
            limit: None,
            output: None,
            min_accuracy: None,
            export_failures: None,
            no_cache: false,
        }
    }
}

impl Benchfile {
    /// Parse a Benchfile from string content
    pub fn parse(content: &str, base_dir: &Path) -> Result<Self> {
        let mut bf = Benchfile::default();

        for (idx, raw_line) in content.lines().enumerate() {
            let line_num = idx + 1;
            let trimmed = raw_line.trim();

            if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("//") {
                continue;
            }

            let (directive, value) = match trimmed.split_once(char::is_whitespace) {
                Some((d, v)) => (d.trim().to_uppercase(), v.trim()),
                None => (trimmed.to_uppercase(), ""),
            };

            match directive.as_str() {
                "MODEL" | "TARGET" => {
                    bf.model = value.to_string();
                }
                "BASELINE" | "BASE" => {
                    if !value.is_empty() {
                        bf.baseline = Some(value.to_string());
                    }
                }
                "SUBSETS" | "SUBSET" => {
                    let list: Vec<String> = value
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect();
                    if !list.is_empty() {
                        bf.subsets = Some(list);
                    }
                }
                "DATA" | "DATASET" => {
                    if !value.is_empty() {
                        bf.data = Some(resolve_path(base_dir, value));
                    }
                }
                "URL" | "SERVER" => {
                    if !value.is_empty() {
                        bf.url = value.to_string();
                    }
                }
                "LIMIT" | "MAX_SAMPLES" => {
                    let lim = value
                        .parse::<usize>()
                        .with_context(|| format!("line {}: invalid integer for LIMIT '{}'", line_num, value))?;
                    bf.limit = Some(lim);
                }
                "OUTPUT" | "REPORT" => {
                    if !value.is_empty() {
                        bf.output = Some(resolve_path(base_dir, value));
                    }
                }
                "EXPORT_FAILURES" | "FAILURES_OUT" | "MINE_FAILURES" => {
                    if !value.is_empty() {
                        bf.export_failures = Some(resolve_path(base_dir, value));
                    }
                }
                "NO_CACHE" | "NOCACHE" => {
                    bf.no_cache = value.is_empty()
                        || value.eq_ignore_ascii_case("true")
                        || value == "1"
                        || value.eq_ignore_ascii_case("yes");
                }
                "THRESHOLD" => {
                    let (metric, val_str) = value
                        .split_once(char::is_whitespace)
                        .ok_or_else(|| anyhow::anyhow!("line {}: THRESHOLD requires 'metric value' (e.g. min_accuracy 0.75)", line_num))?;
                    if metric.eq_ignore_ascii_case("min_accuracy") || metric.eq_ignore_ascii_case("accuracy") {
                        let parsed = val_str
                            .trim_start_matches(">=")
                            .trim()
                            .parse::<f64>()
                            .with_context(|| format!("line {}: invalid float for THRESHOLD accuracy '{}'", line_num, val_str))?;
                        bf.min_accuracy = Some(parsed);
                    } else {
                        bail!("line {}: unsupported THRESHOLD metric '{}'. Supported: min_accuracy", line_num, metric);
                    }
                }
                _ => {
                    bail!("line {}: unrecognized directive '{}' in Benchfile", line_num, directive);
                }
            }
        }

        Ok(bf)
    }

    /// Load a Benchfile from disk
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read Benchfile at {:?}", path))?;
        let base_dir = path.parent().unwrap_or_else(|| Path::new("."));
        Self::parse(&content, base_dir)
    }

    /// Resolve dynamic model tags and output paths (e.g. 'winnow:v{latest}', '{model}_vs_{baseline}')
    pub fn resolve_placeholders(&mut self, candidate_models_dir: Option<&Path>) {
        // Resolve model tag if dynamic
        let models_dir = candidate_models_dir
            .map(|p| p.to_path_buf())
            .or_else(|| crate::registry::get_candidate_models_dirs().into_iter().next())
            .unwrap_or_else(|| PathBuf::from("/usr/share/ollaya/.ollaya/models"));

        self.model = resolve_bench_tag(&self.model, &models_dir);
        if let Some(base) = &self.baseline {
            self.baseline = Some(resolve_bench_tag(base, &models_dir));
        }

        let sanitized_model = self.model.replace(':', "_").replace('/', "_");
        let sanitized_base = self
            .baseline
            .as_deref()
            .unwrap_or("none")
            .replace(':', "_")
            .replace('/', "_");

        // Expand placeholders in output path
        if let Some(out) = &self.output {
            let out_str = out.to_string_lossy().to_string();
            let expanded = out_str
                .replace("{model}", &sanitized_model)
                .replace("{baseline}", &sanitized_base)
                .replace("{tag}", &sanitized_model);

            self.output = Some(PathBuf::from(expanded));
        }

        // Expand placeholders in export_failures path
        if let Some(fail_out) = &self.export_failures {
            let fail_str = fail_out.to_string_lossy().to_string();
            let expanded = fail_str
                .replace("{model}", &sanitized_model)
                .replace("{baseline}", &sanitized_base)
                .replace("{tag}", &sanitized_model);

            self.export_failures = Some(PathBuf::from(expanded));
        }
    }
}

/// Resolve dynamic tags in Benchfile like 'winnow:v{latest}' or 'winnow:v{auto}'
pub fn resolve_bench_tag(tag_spec: &str, models_dir: &Path) -> String {
    let (model, tag) = crate::registry::parse_model_name(tag_spec);

    if tag.contains("{latest}") || tag == "latest" || tag.contains("{auto}") {
        // Search manifests for the highest version v<N>
        let candidate_dirs = crate::registry::get_candidate_models_dirs();
        let mut highest_v = 0;
        let mut found_any = false;

        for dir in &candidate_dirs {
            let manifest_dir = dir.join("manifests/ollaya.dev/library").join(&model);
            if let Ok(entries) = std::fs::read_dir(&manifest_dir) {
                for entry in entries.flatten() {
                    let file_name = entry.file_name().to_string_lossy().to_string();
                    if let Some(num_str) = file_name.strip_prefix('v') {
                        if let Ok(num) = num_str.parse::<usize>() {
                            highest_v = highest_v.max(num);
                            found_any = true;
                        }
                    }
                }
            }
        }

        if found_any && highest_v > 0 {
            format!("{}:v{}", model, highest_v)
        } else {
            // Check direct models_dir if not found yet
            crate::registry::resolve_dynamic_tag(tag_spec, models_dir)
        }
    } else {
        crate::registry::resolve_dynamic_tag(tag_spec, models_dir)
    }
}

fn resolve_path(base_dir: &Path, p: &str) -> PathBuf {
    let path = Path::new(p);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base_dir.join(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_benchfile_basic() {
        let content = r#"
# Benchfile for winnow models
MODEL winnow:custom
BASELINE winnow:e4b
SUBSETS multinli, paws, boolq
URL http://127.0.0.1:11435
LIMIT 25
OUTPUT ./results/test_report.json
EXPORT_FAILURES ./data/mined_failures.jsonl
THRESHOLD min_accuracy 0.75
"#;
        let bf = Benchfile::parse(content, Path::new(".")).unwrap();
        assert_eq!(bf.model, "winnow:custom");
        assert_eq!(bf.baseline, Some("winnow:e4b".to_string()));
        assert_eq!(
            bf.subsets,
            Some(vec![
                "multinli".to_string(),
                "paws".to_string(),
                "boolq".to_string()
            ])
        );
        assert_eq!(bf.url, "http://127.0.0.1:11435");
        assert_eq!(bf.limit, Some(25));
        assert_eq!(bf.output, Some(PathBuf::from("./results/test_report.json")));
        assert_eq!(
            bf.export_failures,
            Some(PathBuf::from("./data/mined_failures.jsonl"))
        );
        assert_eq!(bf.min_accuracy, Some(0.75));
    }

    #[test]
    fn test_resolve_placeholders_expansion() {
        let mut bf = Benchfile {
            model: "winnow:v1".to_string(),
            baseline: Some("winnow:e4b".to_string()),
            subsets: None,
            data: None,
            url: "http://127.0.0.1:11435".to_string(),
            limit: None,
            output: Some(PathBuf::from("./results/{model}_vs_{baseline}.json")),
            min_accuracy: None,
            export_failures: Some(PathBuf::from("./data/failures_{model}.jsonl")),
            no_cache: false,
        };

        bf.resolve_placeholders(None);
        assert_eq!(
            bf.output,
            Some(PathBuf::from("./results/winnow_v1_vs_winnow_e4b.json"))
        );
        assert_eq!(
            bf.export_failures,
            Some(PathBuf::from("./data/failures_winnow_v1.jsonl"))
        );
    }
}
