//! Curatefile: Declarative recipe for dataset curation, labeling, and synthetic generation
//! using frontier models or CLI teacher engines (e.g., agy, claude, ollama).

use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurateMode {
    /// Annotate raw user text/events with gold labels and confidence
    Label,
    /// Generate synthetic examples directly from schema criteria
    Synth,
    /// Paraphrase and generate contrastive hard-negatives
    Augment,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetDatasetFormat {
    /// Decision dataset matching Ollaya (Winnow, Decider, Kev)
    Decision,
    /// Standard text classification (ModernBERT, BERT, DeBERTa)
    Classification,
    /// Natural Language Inference pairs (Premise, Hypothesis, Label)
    Nli,
}

#[derive(Debug, Clone)]
pub struct Curatefile {
    /// Teacher engine (TEACHER): 'agy', 'claude', 'ollama:<model>', 'openai:<model>'
    pub teacher: String,
    /// Curation mode (MODE): 'label', 'synth', 'augment'
    pub mode: CurateMode,
    /// Target dataset format (FORMAT): 'decision', 'classification', 'nli'
    pub format: TargetDatasetFormat,
    /// Path to raw input data file (INPUT / DATA)
    pub input: Option<PathBuf>,
    /// Path to question schema or policy criteria (SCHEMA)
    pub schema: Option<PathBuf>,
    /// Target output path for the curated dataset (OUTPUT)
    pub output: PathBuf,
    /// Number of examples to generate in synth/augment mode (COUNT / LIMIT)
    pub count: usize,
    /// Batch size for teacher queries (PARAMETER batch_size)
    pub batch_size: usize,
    /// Minimum confidence threshold to accept label without quarantine (PARAMETER min_confidence)
    pub min_confidence: f64,
    /// Generate hard-negative contrastive examples (PARAMETER add_hard_negatives)
    pub add_hard_negatives: bool,
}

impl Default for Curatefile {
    fn default() -> Self {
        Self {
            teacher: "agy".to_string(),
            mode: CurateMode::Label,
            format: TargetDatasetFormat::Decision,
            input: None,
            schema: None,
            output: PathBuf::from("./data/curated_train.jsonl"),
            count: 50,
            batch_size: 10,
            min_confidence: 0.80,
            add_hard_negatives: true,
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

            match directive.as_str() {
                "TEACHER" | "ENGINE" => {
                    cf.teacher = value.to_string();
                }
                "MODE" => {
                    cf.mode = match value.to_lowercase().as_str() {
                        "label" | "annotate" => CurateMode::Label,
                        "synth" | "synthetic" | "generate" => CurateMode::Synth,
                        "augment" | "contrastive" => CurateMode::Augment,
                        _ => bail!("line {}: unknown MODE '{}'. Expected: label, synth, augment", line_num, value),
                    };
                }
                "FORMAT" | "TASK" => {
                    cf.format = match value.to_lowercase().as_str() {
                        "decision" | "choice" | "noul" | "score" => TargetDatasetFormat::Decision,
                        "classification" | "text" => TargetDatasetFormat::Classification,
                        "nli" | "pair" | "pairs" => TargetDatasetFormat::Nli,
                        _ => bail!("line {}: unknown FORMAT '{}'. Expected: decision, classification, nli", line_num, value),
                    };
                }
                "INPUT" | "DATA" => {
                    if !value.is_empty() {
                        cf.input = Some(resolve_path(base_dir, value));
                    }
                }
                "SCHEMA" | "POLICY" => {
                    if !value.is_empty() {
                        cf.schema = Some(resolve_path(base_dir, value));
                    }
                }
                "OUTPUT" => {
                    if !value.is_empty() {
                        cf.output = resolve_path(base_dir, value);
                    }
                }
                "COUNT" | "LIMIT" => {
                    cf.count = value
                        .parse::<usize>()
                        .with_context(|| format!("line {}: invalid integer for COUNT '{}'", line_num, value))?;
                }
                "PARAMETER" => {
                    let (p_name, p_val) = value
                        .split_once(char::is_whitespace)
                        .ok_or_else(|| anyhow::anyhow!("line {}: PARAMETER requires 'name value'", line_num))?;
                    match p_name.trim().to_lowercase().as_str() {
                        "batch_size" | "batch" => {
                            cf.batch_size = p_val.trim().parse::<usize>()
                                .with_context(|| format!("line {}: invalid integer for batch_size", line_num))?;
                        }
                        "min_confidence" | "confidence" => {
                            cf.min_confidence = p_val.trim().parse::<f64>()
                                .with_context(|| format!("line {}: invalid float for min_confidence", line_num))?;
                        }
                        "add_hard_negatives" | "hard_negatives" => {
                            cf.add_hard_negatives = match p_val.trim().to_lowercase().as_str() {
                                "true" | "1" | "yes" => true,
                                "false" | "0" | "no" => false,
                                _ => bail!("line {}: invalid boolean for add_hard_negatives", line_num),
                            };
                        }
                        _ => {}
                    }
                }
                _ => {
                    bail!("line {}: unrecognized directive '{}' in Curatefile", line_num, directive);
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
TEACHER agy
MODE label
FORMAT decision
INPUT ./data/raw.jsonl
SCHEMA ./data/schema.json
OUTPUT ./data/curated.jsonl
COUNT 100
PARAMETER batch_size 12
PARAMETER min_confidence 0.85
PARAMETER add_hard_negatives true
"#;
        let cf = Curatefile::parse(content, Path::new(".")).unwrap();
        assert_eq!(cf.teacher, "agy");
        assert_eq!(cf.mode, CurateMode::Label);
        assert_eq!(cf.format, TargetDatasetFormat::Decision);
        assert_eq!(cf.input, Some(PathBuf::from("./data/raw.jsonl")));
        assert_eq!(cf.schema, Some(PathBuf::from("./data/schema.json")));
        assert_eq!(cf.output, PathBuf::from("./data/curated.jsonl"));
        assert_eq!(cf.count, 100);
        assert_eq!(cf.batch_size, 12);
        assert_eq!(cf.min_confidence, 0.85);
        assert!(cf.add_hard_negatives);
    }
}
