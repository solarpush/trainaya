//! Curatefile: Declarative & interactive recipe for Use-Case dataset curation,
//! raw dataset reformatting, and teacher agent distillation (e.g., agy, claude, codex, ollama).

use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurateMode {
    /// Reformat & distill raw user datasets (JSONL, CSV, JSON, TXT) with teacher labels and soft targets
    Label,
    /// Synthesize a complete tailored dataset directly from a Use-Case description and/or schema
    Synth,
    /// Reformat raw dataset and generate minimally perturbed counterfactual hard-negative twins
    Augment,
}

impl CurateMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            CurateMode::Label => "reformat",
            CurateMode::Synth => "synth",
            CurateMode::Augment => "augment",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetDatasetFormat {
    /// Decision dataset matching Ollaya & Trainaya (choice, noul, score with calibrated target distribution)
    Decision,
    /// Standard text classification (ModernBERT, BERT, DeBERTa)
    Classification,
    /// Natural Language Inference pairs (Premise, Hypothesis, Label)
    Nli,
}

#[derive(Debug, Clone)]
pub struct Curatefile {
    /// Free-text description of the final Use-Case, domain rules, and decision objective (USE_CASE)
    pub use_case: Option<String>,
    /// Teacher engine (TEACHER): 'agy', 'claude', 'codex', 'gemini', 'ollama:<model>', 'ollaya:<model>'
    pub teacher: String,
    /// Curation mode (MODE): 'reformat' (or 'label'), 'synth', 'augment'
    pub mode: CurateMode,
    /// Target dataset format (FORMAT): 'decision', 'classification', 'nli'
    pub format: TargetDatasetFormat,
    /// Decision primitive kind when synthesizing schema from Use-Case ('choice', 'noul', 'score')
    pub kind: String,
    /// Path to raw input dataset to reformat/distill (INPUT / DATA)
    pub input: Option<PathBuf>,
    /// Path to explicit question schema or policy criteria (SCHEMA)
    pub schema: Option<PathBuf>,
    /// Target output path for the curated dataset (OUTPUT)
    pub output: PathBuf,
    /// Number of examples to generate in synth mode (COUNT / LIMIT)
    pub count: usize,
    /// Batch size for teacher queries (PARAMETER batch_size)
    pub batch_size: usize,
    /// Minimum confidence threshold to accept label without quarantine (PARAMETER min_confidence)
    pub min_confidence: f64,
    /// Generate hard-negative contrastive examples (PARAMETER add_hard_negatives)
    pub add_hard_negatives: bool,
    /// Whether to run the interactive terminal wizard before executing
    pub interactive: bool,
}

impl Default for Curatefile {
    fn default() -> Self {
        Self {
            use_case: None,
            teacher: "agy".to_string(),
            mode: CurateMode::Synth,
            format: TargetDatasetFormat::Decision,
            kind: "choice".to_string(),
            input: None,
            schema: None,
            output: PathBuf::from("./data/curated_train.jsonl"),
            count: 50,
            batch_size: 10,
            min_confidence: 0.80,
            add_hard_negatives: true,
            interactive: false,
        }
    }
}

impl Curatefile {
    /// Parse Curatefile content from string
    pub fn parse(content: &str, base_dir: &Path) -> Result<Self> {
        let mut cf = Curatefile::default();

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

            let unquoted = value.trim_matches('"').trim_matches('\'').trim();

            match directive.as_str() {
                "USE_CASE" | "USECASE" | "DESCRIPTION" | "DOMAIN" => {
                    if !unquoted.is_empty() {
                        cf.use_case = Some(unquoted.to_string());
                    }
                }
                "TEACHER" | "ENGINE" | "AGENT" => {
                    cf.teacher = unquoted.to_string();
                }
                "MODE" => {
                    cf.mode = match unquoted.to_lowercase().as_str() {
                        "reformat" | "distill" | "label" | "annotate" => CurateMode::Label,
                        "synth" | "synthetic" | "generate" => CurateMode::Synth,
                        "augment" | "contrastive" => CurateMode::Augment,
                        _ => bail!(
                            "line {}: unknown MODE '{}'. Expected: reformat, label, synth, augment",
                            line_num,
                            value
                        ),
                    };
                }
                "FORMAT" | "TASK" => {
                    let lower = unquoted.to_lowercase();
                    cf.format = match lower.as_str() {
                        "decision" | "choice" | "noul" | "score" => TargetDatasetFormat::Decision,
                        "classification" | "text" => TargetDatasetFormat::Classification,
                        "nli" | "pair" | "pairs" => TargetDatasetFormat::Nli,
                        _ => bail!(
                            "line {}: unknown FORMAT '{}'. Expected: decision, classification, nli",
                            line_num,
                            value
                        ),
                    };
                    if matches!(lower.as_str(), "choice" | "noul" | "score") {
                        cf.kind = lower;
                    }
                }
                "KIND" | "DECISION_TYPE" | "TYPE" => {
                    let lower = unquoted.to_lowercase();
                    match lower.as_str() {
                        "choice" | "noul" | "score" => cf.kind = lower,
                        _ => bail!(
                            "line {}: unknown KIND '{}'. Expected: choice, noul, score",
                            line_num,
                            value
                        ),
                    }
                }
                "INPUT" | "DATA" => {
                    if !unquoted.is_empty() {
                        cf.input = Some(resolve_path(base_dir, unquoted));
                    }
                }
                "SCHEMA" | "POLICY" => {
                    if !unquoted.is_empty() {
                        cf.schema = Some(resolve_path(base_dir, unquoted));
                    }
                }
                "OUTPUT" => {
                    if !unquoted.is_empty() {
                        cf.output = resolve_path(base_dir, unquoted);
                    }
                }
                "COUNT" | "LIMIT" => {
                    cf.count = unquoted.parse::<usize>().with_context(|| {
                        format!("line {}: invalid integer for COUNT '{}'", line_num, value)
                    })?;
                }
                "PARAMETER" => {
                    let (p_name, p_val) = value.split_once(char::is_whitespace).ok_or_else(|| {
                        anyhow::anyhow!("line {}: PARAMETER requires 'name value'", line_num)
                    })?;
                    match p_name.trim().to_lowercase().as_str() {
                        "batch_size" | "batch" => {
                            cf.batch_size = p_val.trim().parse::<usize>().with_context(|| {
                                format!("line {}: invalid integer for batch_size", line_num)
                            })?;
                        }
                        "min_confidence" | "confidence" => {
                            cf.min_confidence = p_val.trim().parse::<f64>().with_context(|| {
                                format!("line {}: invalid float for min_confidence", line_num)
                            })?;
                        }
                        "add_hard_negatives" | "hard_negatives" => {
                            cf.add_hard_negatives = match p_val.trim().to_lowercase().as_str() {
                                "true" | "1" | "yes" => true,
                                "false" | "0" | "no" => false,
                                _ => bail!(
                                    "line {}: invalid boolean for add_hard_negatives",
                                    line_num
                                ),
                            };
                        }
                        _ => {}
                    }
                }
                _ => {
                    bail!(
                        "line {}: unrecognized directive '{}' in Curatefile",
                        line_num,
                        directive
                    );
                }
            }
        }

        Ok(cf)
    }

    /// Load Curatefile from file
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read Curatefile at {:?}", path))?;
        let base_dir = path.parent().unwrap_or_else(|| Path::new("."));
        Self::parse(&content, base_dir)
    }

    /// Render this configuration as a clean declarative `.Curatefile`
    pub fn to_curatefile_string(&self) -> String {
        let mut lines = Vec::new();
        lines.push("# .Curatefile — Declarative Use-Case Dataset Curation & Teacher Distillation".to_string());
        lines.push("# Run with: trainaya curate (interactive) or trainaya curate --non-interactive".to_string());
        lines.push(String::new());
        if let Some(ref uc) = self.use_case {
            lines.push(format!("USE_CASE \"{}\"", uc.replace('"', "'")));
        }
        lines.push(format!("TEACHER {}", self.teacher));
        lines.push(format!("MODE {}", self.mode.as_str()));
        lines.push(format!(
            "FORMAT {}",
            match self.format {
                TargetDatasetFormat::Decision => "decision",
                TargetDatasetFormat::Classification => "classification",
                TargetDatasetFormat::Nli => "nli",
            }
        ));
        lines.push(format!("KIND {}", self.kind));
        if let Some(ref inp) = self.input {
            lines.push(format!("INPUT {}", inp.display()));
        }
        if let Some(ref sc) = self.schema {
            lines.push(format!("SCHEMA {}", sc.display()));
        }
        lines.push(format!("OUTPUT {}", self.output.display()));
        lines.push(format!("COUNT {}", self.count));
        lines.push(String::new());
        lines.push(format!("PARAMETER batch_size {}", self.batch_size));
        lines.push(format!("PARAMETER min_confidence {:.2}", self.min_confidence));
        lines.push(format!(
            "PARAMETER add_hard_negatives {}",
            self.add_hard_negatives
        ));
        lines.push(String::new());
        lines.join("\n")
    }

    /// Return the list of configured teacher engines (supports comma-separated multi-teacher consensus)
    pub fn teachers(&self) -> Vec<String> {
        let list: Vec<String> = self
            .teacher
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        if list.is_empty() {
            vec!["agy".to_string()]
        } else {
            list
        }
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
    fn test_parse_curatefile_basic() {
        let content = r#"
# Dataset curation configuration
USE_CASE "Triage incoming customer support tickets into refund, technical_help, or billing_question"
TEACHER agy
MODE label
FORMAT decision
KIND choice
INPUT ./data/raw.jsonl
SCHEMA ./data/schema.json
OUTPUT ./data/curated.jsonl
COUNT 100
PARAMETER batch_size 12
PARAMETER min_confidence 0.85
PARAMETER add_hard_negatives true
"#;
        let cf = Curatefile::parse(content, Path::new(".")).unwrap();
        assert_eq!(
            cf.use_case.as_deref(),
            Some("Triage incoming customer support tickets into refund, technical_help, or billing_question")
        );
        assert_eq!(cf.teacher, "agy");
        assert_eq!(cf.teachers(), vec!["agy".to_string()]);
        assert_eq!(cf.mode, CurateMode::Label);
        assert_eq!(cf.format, TargetDatasetFormat::Decision);
        assert_eq!(cf.kind, "choice");
        assert_eq!(cf.input, Some(PathBuf::from("./data/raw.jsonl")));
        assert_eq!(cf.schema, Some(PathBuf::from("./data/schema.json")));
        assert_eq!(cf.output, PathBuf::from("./data/curated.jsonl"));
        assert_eq!(cf.count, 100);
        assert_eq!(cf.batch_size, 12);
        assert_eq!(cf.min_confidence, 0.85);
        assert!(cf.add_hard_negatives);
    }

    #[test]
    fn test_parse_curatefile_multi_teacher_and_reformat() {
        let content = r#"
USE_CASE "Evaluate medical abstract claims against clinical trial evidence"
TEACHER agy, claude, ollama:qwen2.5:72b
MODE reformat
FORMAT decision
KIND noul
"#;
        let cf = Curatefile::parse(content, Path::new(".")).unwrap();
        assert_eq!(
            cf.teachers(),
            vec![
                "agy".to_string(),
                "claude".to_string(),
                "ollama:qwen2.5:72b".to_string()
            ]
        );
        assert_eq!(cf.mode, CurateMode::Label);
        assert_eq!(cf.kind, "noul");
    }
}
