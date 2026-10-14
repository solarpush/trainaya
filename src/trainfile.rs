//! Trainfile parsing and configuration: defines model training recipes mirroring Ollaya's Modelfile.
//!
//! ```text
//! # Example .Trainfile
//! FROM winnow:e4b
//! DATA ./data/train_specific_decision.jsonl
//! SCHEMA ./data/questions_schema.json
//!
//! PARAMETER epochs 3
//! PARAMETER batch_size 8
//! PARAMETER lr 5e-5
//! PARAMETER lora_r 16
//! PARAMETER lora_alpha 32
//!
//! MERGE true
//! QUANT Q8_0
//! REGISTER winnow:custom
//! OUTPUT ./output_trained
//! ```
//!
//! Directives are case-insensitive. Values can be file paths (relative to the Trainfile)
//! or `"""`-quoted multi-line blocks. Lines starting with `#` are comments.

use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

use crate::config::{LoraConfig, MergeMode, TaskType, TrainingArgs};

#[derive(Debug, Clone)]
pub struct Trainfile {
    /// Base model tag, huggingface repo id, or local path (FROM)
    pub from: String,
    /// Path to dataset file (DATA)
    pub data: PathBuf,
    /// Optional path to questions schema JSON (SCHEMA)
    pub schema: Option<PathBuf>,
    /// Optional path to dedicated evaluation dataset (EVAL)
    pub eval: Option<PathBuf>,
    /// Fraction of training data for evaluation (EVAL_SPLIT)
    pub eval_split: f32,
    /// Task type (TASK: classification or nli)
    pub task: TaskType,
    /// Optional model family override (FAMILY)
    pub family: Option<String>,
    /// Directory where the output will be stored (OUTPUT)
    pub output_dir: PathBuf,
    /// Number of training epochs (PARAMETER epochs)
    pub epochs: usize,
    /// Training batch size (PARAMETER batch_size)
    pub batch_size: usize,
    /// Gradient accumulation steps (PARAMETER grad_accum)
    pub grad_accum_steps: usize,
    /// Learning rate (PARAMETER lr)
    pub lr: f64,
    /// Warmup ratio for LR scheduler (PARAMETER warmup_ratio)
    pub warmup_ratio: f32,
    /// Max gradient norm for clipping (PARAMETER max_grad_norm)
    pub max_grad_norm: f32,
    /// LR scheduler strategy: cosine, linear, constant (PARAMETER lr_scheduler)
    pub lr_scheduler: String,
    /// Maximum sequence length (PARAMETER max_length)
    pub max_length: usize,
    /// Optimizer weight decay (PARAMETER weight_decay)
    pub weight_decay: f32,
    /// Random seed (PARAMETER seed)
    pub seed: u64,
    /// Target compute device (PARAMETER device)
    pub device: String,
    /// 4-bit QLoRA enablement (PARAMETER qlora)
    pub qlora: bool,
    /// Mixed precision BF16 enablement (PARAMETER bf16)
    pub bf16: bool,
    /// Number of top transformer blocks to inject LoRA into for GGUF (PARAMETER lora_layers)
    pub lora_layers: usize,
    /// LoRA configuration (PARAMETER lora_r, lora_alpha, lora_dropout, target_modules)
    pub lora: LoraConfig,
    /// Merge strategy: auto, merge, no-merge (MERGE)
    pub merge_mode: MergeMode,
    /// Quantization format (QUANT)
    pub quant: Option<String>,
    /// Optional model name to register in Ollaya after training (REGISTER)
    pub register: Option<String>,
    /// Base template for Ollaya manifest inheritance (TEMPLATE)
    pub template: Option<String>,
    /// Custom Ollaya models store directory (MODELS_DIR)
    pub models_dir: Option<PathBuf>,
}

impl Default for Trainfile {
    fn default() -> Self {
        Self {
            from: String::new(),
            data: PathBuf::new(),
            schema: None,
            eval: None,
            eval_split: 0.1,
            task: TaskType::Classification,
            family: None,
            output_dir: PathBuf::from("./output_trained"),
            epochs: 3,
            batch_size: 8,
            grad_accum_steps: 1,
            lr: 5e-5,
            warmup_ratio: 0.1,
            max_grad_norm: 1.0,
            lr_scheduler: "cosine".to_string(),
            max_length: 128,
            weight_decay: 0.01,
            seed: 42,
            device: "auto".to_string(),
            qlora: false,
            bf16: false,
            lora_layers: 4,
            lora: LoraConfig::default(),
            merge_mode: MergeMode::Auto,
            quant: None,
            register: None,
            template: None,
            models_dir: None,
        }
    }
}

impl Trainfile {
    /// Read and parse a Trainfile from disk
    pub fn read(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read Trainfile at {}", path.display()))?;
        let base_dir = path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        Self::parse(&content, &base_dir)
    }

    /// Locate default Trainfile in directory (.Trainfile, Trainfile, trainfile, .trainfile)
    pub fn find_default(dir: &Path) -> Option<PathBuf> {
        let candidates = [".Trainfile", "Trainfile", "trainfile", ".trainfile"];
        for candidate in candidates {
            let p = dir.join(candidate);
            if p.is_file() {
                return Some(p);
            }
        }
        None
    }

    /// Parse Trainfile text with relative paths resolved against `base_dir`
    pub fn parse(text: &str, base_dir: &Path) -> Result<Self> {
        let mut tf = Self::default();
        let mut lines = text.lines().enumerate();

        while let Some((line_num, raw_line)) = lines.next() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
                continue;
            }

            let (directive, rest) = line.split_once(char::is_whitespace).unwrap_or((line, ""));
            let mut value = rest.trim().to_string();

            // Multi-line block: """ ... """
            if let Some(start) = value.strip_prefix("\"\"\"") {
                let mut block = start.to_string();
                if let Some(end) = block.find("\"\"\"") {
                    block.truncate(end);
                } else {
                    loop {
                        let Some((_, next)) = lines.next() else {
                            bail!("line {}: unterminated triple-quote block (\"\"\")", line_num + 1);
                        };
                        if let Some(end) = next.find("\"\"\"") {
                            block.push('\n');
                            block.push_str(&next[..end]);
                            break;
                        }
                        block.push('\n');
                        block.push_str(next);
                    }
                }
                value = block.trim().to_string();
            }

            if value.is_empty() && !directive.eq_ignore_ascii_case("MERGE") {
                bail!("line {}: directive '{}' requires a value", line_num + 1, directive);
            }

            apply_directive(&mut tf, directive, &value, base_dir, line_num + 1)?;
        }

        if tf.from.is_empty() {
            bail!("Trainfile requires a 'FROM <model>' directive");
        }
        if tf.data.as_os_str().is_empty() {
            bail!("Trainfile requires a 'DATA <path>' directive");
        }

        Ok(tf)
    }

    /// Convert into TrainingArgs for execution
    pub fn to_training_args(&self) -> TrainingArgs {
        TrainingArgs {
            base_model: self.from.clone(),
            family: self.family.clone(),
            data_path: self.data.clone(),
            eval_path: self.eval.clone(),
            eval_split: self.eval_split,
            output_dir: self.output_dir.clone(),
            task: self.task,
            epochs: self.epochs,
            batch_size: self.batch_size,
            grad_accum_steps: self.grad_accum_steps,
            lr: self.lr,
            warmup_ratio: self.warmup_ratio,
            max_grad_norm: self.max_grad_norm,
            lr_scheduler: self.lr_scheduler.clone(),
            max_length: self.max_length,
            weight_decay: self.weight_decay,
            seed: self.seed,
            qlora: self.qlora,
            bf16: self.bf16,
            lora_layers: self.lora_layers,
            device: self.device.clone(),
            lora: self.lora.clone(),
            merge_mode: self.merge_mode,
            quant: self.quant.clone(),
        }
    }
}

fn resolve_path(base_dir: &Path, p: &str) -> PathBuf {
    let path_str = if let Some(stripped) = p.strip_prefix("~/") {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        format!("{}/{}", home, stripped)
    } else {
        p.to_string()
    };

    let pb = PathBuf::from(path_str);
    if pb.is_absolute() {
        pb
    } else {
        base_dir.join(pb)
    }
}

fn apply_directive(
    tf: &mut Trainfile,
    directive: &str,
    value: &str,
    base_dir: &Path,
    line_num: usize,
) -> Result<()> {
    let dir_upper = directive.to_uppercase();
    match dir_upper.as_str() {
        "FROM" => {
            tf.from = value.to_string();
        }
        "DATA" => {
            // If inline JSON, write to temporary file
            if value.starts_with('{') || value.starts_with('[') {
                let tmp_dir = std::env::temp_dir().join("trainaya_trainfiles");
                let _ = std::fs::create_dir_all(&tmp_dir);
                let tmp_path = tmp_dir.join(format!("data_{}.jsonl", std::process::id()));
                std::fs::write(&tmp_path, value)?;
                tf.data = tmp_path;
            } else {
                tf.data = resolve_path(base_dir, value);
            }
        }
        "SCHEMA" => {
            if value.starts_with('{') {
                let tmp_dir = std::env::temp_dir().join("trainaya_trainfiles");
                let _ = std::fs::create_dir_all(&tmp_dir);
                let tmp_path = tmp_dir.join(format!("schema_{}.json", std::process::id()));
                std::fs::write(&tmp_path, value)?;
                tf.schema = Some(tmp_path);
            } else {
                tf.schema = Some(resolve_path(base_dir, value));
            }
        }
        "EVAL" => {
            tf.eval = Some(resolve_path(base_dir, value));
        }
        "EVAL_SPLIT" => {
            tf.eval_split = value
                .parse::<f32>()
                .with_context(|| format!("line {}: invalid float for EVAL_SPLIT", line_num))?;
        }
        "TASK" => match value.to_lowercase().as_str() {
            "classification" => tf.task = TaskType::Classification,
            "nli" => tf.task = TaskType::Nli,
            "decision" => tf.task = TaskType::Decision,
            _ => bail!("line {}: unknown TASK '{}'. Expected 'classification', 'nli', or 'decision'", line_num, value),
        },
        "FAMILY" => {
            tf.family = Some(value.to_string());
        }
        "OUTPUT" => {
            tf.output_dir = resolve_path(base_dir, value);
        }
        "PARAMETER" => {
            let (param_name, param_val) = value
                .split_once(char::is_whitespace)
                .ok_or_else(|| anyhow::anyhow!("line {}: PARAMETER requires 'name value'", line_num))?;
            apply_parameter(tf, param_name.trim(), param_val.trim(), line_num)?;
        }
        "MERGE" => {
            tf.merge_mode = match value.to_lowercase().as_str() {
                "true" | "1" | "yes" | "merge" => MergeMode::Merge,
                "false" | "0" | "no" | "no-merge" | "unmerged" => MergeMode::NoMerge,
                "auto" | "" => MergeMode::Auto,
                _ => bail!("line {}: invalid MERGE value '{}'. Use 'true', 'false', or 'auto'", line_num, value),
            };
        }
        "QUANT" | "QUANTIZE" => {
            tf.quant = Some(value.to_string());
        }
        "EXPORT" => {
            // Optional export format directive in Trainfile (gguf, safetensors, onnx)
        }
        "TAG" | "REGISTER" => {
            tf.register = Some(value.to_string());
        }
        "TEMPLATE" => {
            tf.template = Some(value.to_string());
        }
        "MODELS_DIR" | "MODELS-DIR" => {
            tf.models_dir = Some(resolve_path(base_dir, value));
        }
        _ => {
            bail!("line {}: unrecognized directive '{}'", line_num, directive);
        }
    }
    Ok(())
}

fn apply_parameter(tf: &mut Trainfile, name: &str, val: &str, line_num: usize) -> Result<()> {
    match name.to_lowercase().as_str() {
        "epochs" => {
            tf.epochs = val
                .parse::<usize>()
                .with_context(|| format!("line {}: invalid integer for PARAMETER epochs", line_num))?;
        }
        "batch_size" | "batch" => {
            tf.batch_size = val
                .parse::<usize>()
                .with_context(|| format!("line {}: invalid integer for PARAMETER batch_size", line_num))?;
        }
        "grad_accum" | "grad_accum_steps" | "gradient_accumulation_steps" => {
            tf.grad_accum_steps = val
                .parse::<usize>()
                .with_context(|| format!("line {}: invalid integer for PARAMETER grad_accum", line_num))?
                .max(1);
        }
        "lr" | "learning_rate" => {
            tf.lr = val
                .parse::<f64>()
                .with_context(|| format!("line {}: invalid float for PARAMETER lr", line_num))?;
        }
        "warmup_ratio" | "warmup" => {
            tf.warmup_ratio = val
                .parse::<f32>()
                .with_context(|| format!("line {}: invalid float for PARAMETER warmup_ratio", line_num))?;
        }
        "max_grad_norm" | "grad_clip" => {
            tf.max_grad_norm = val
                .parse::<f32>()
                .with_context(|| format!("line {}: invalid float for PARAMETER max_grad_norm", line_num))?;
        }
        "lr_scheduler" | "scheduler" => {
            tf.lr_scheduler = val.to_lowercase();
        }
        "max_length" | "seq_len" => {
            tf.max_length = val
                .parse::<usize>()
                .with_context(|| format!("line {}: invalid integer for PARAMETER max_length", line_num))?;
        }
        "weight_decay" => {
            tf.weight_decay = val
                .parse::<f32>()
                .with_context(|| format!("line {}: invalid float for PARAMETER weight_decay", line_num))?;
        }
        "seed" => {
            tf.seed = val
                .parse::<u64>()
                .with_context(|| format!("line {}: invalid integer for PARAMETER seed", line_num))?;
        }
        "device" => {
            tf.device = val.to_string();
        }
        "qlora" => {
            tf.qlora = val.parse::<bool>().unwrap_or(true);
        }
        "bf16" => {
            tf.bf16 = val.parse::<bool>().unwrap_or(true);
        }
        "lora_layers" | "num_lora_layers" => {
            tf.lora_layers = val
                .parse::<usize>()
                .with_context(|| format!("line {}: invalid integer for PARAMETER lora_layers", line_num))?;
        }
        "lora_r" | "r" => {
            tf.lora.r = val
                .parse::<usize>()
                .with_context(|| format!("line {}: invalid integer for PARAMETER lora_r", line_num))?;
        }
        "lora_alpha" | "alpha" => {
            tf.lora.alpha = val
                .parse::<f32>()
                .with_context(|| format!("line {}: invalid float for PARAMETER lora_alpha", line_num))?;
        }
        "lora_dropout" | "dropout" => {
            tf.lora.dropout = val
                .parse::<f32>()
                .with_context(|| format!("line {}: invalid float for PARAMETER lora_dropout", line_num))?;
        }
        "target_modules" | "targets" => {
            tf.lora.target_modules = val
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }
        _ => {
            bail!("line {}: unknown PARAMETER '{}'", line_num, name);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_trainfile_basic() {
        let content = r#"
# Fine-tune Winnow on DevOps policy
FROM winnow:e4b
DATA ./data/train_specific_decision.jsonl
SCHEMA ./data/questions_schema.json

PARAMETER epochs 4
PARAMETER batch_size 16
PARAMETER lr 2e-5
PARAMETER lora_r 32
PARAMETER lora_alpha 64
PARAMETER target_modules attn_output,ffn_down

MERGE true
QUANT Q8_0
REGISTER winnow:custom
OUTPUT ./output_winnow
"#;

        let base_dir = Path::new("/workspace");
        let tf = Trainfile::parse(content, base_dir).unwrap();

        assert_eq!(tf.from, "winnow:e4b");
        assert_eq!(tf.data, base_dir.join("./data/train_specific_decision.jsonl"));
        assert_eq!(tf.schema, Some(base_dir.join("./data/questions_schema.json")));
        assert_eq!(tf.epochs, 4);
        assert_eq!(tf.batch_size, 16);
        assert_eq!(tf.lr, 2e-5);
        assert_eq!(tf.lora.r, 32);
        assert_eq!(tf.lora.alpha, 64.0);
        assert_eq!(tf.lora.target_modules, vec!["attn_output", "ffn_down"]);
        assert_eq!(tf.merge_mode, MergeMode::Merge);
        assert_eq!(tf.quant, Some("Q8_0".to_string()));
        assert_eq!(tf.register, Some("winnow:custom".to_string()));
        assert_eq!(tf.output_dir, base_dir.join("./output_winnow"));
    }

    #[test]
    fn test_parse_trainfile_multiline_and_comments() {
        let content = r#"
// Comment style
FROM kev:0.8b
DATA ./data.jsonl
MERGE false
PARAMETER device cpu
"#;
        let tf = Trainfile::parse(content, Path::new(".")).unwrap();
        assert_eq!(tf.from, "kev:0.8b");
        assert_eq!(tf.merge_mode, MergeMode::NoMerge);
        assert_eq!(tf.device, "cpu");
    }
}
