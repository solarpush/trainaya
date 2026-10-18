use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum TaskType {
    Classification,
    Nli,
    Decision,
}

impl std::fmt::Display for TaskType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaskType::Classification => write!(f, "classification"),
            TaskType::Nli => write!(f, "nli"),
            TaskType::Decision => write!(f, "decision"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum ExportFormat {
    Safetensors,
    Gguf,
    Onnx,
}

impl std::fmt::Display for ExportFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExportFormat::Safetensors => write!(f, "safetensors"),
            ExportFormat::Gguf => write!(f, "gguf"),
            ExportFormat::Onnx => write!(f, "onnx"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraConfig {
    pub r: usize,
    pub alpha: f32,
    pub dropout: f32,
    pub target_modules: Vec<String>,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            r: 16,
            alpha: 32.0,
            dropout: 0.05,
            target_modules: vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
                "out_proj".to_string(),
                "query".to_string(),
                "key".to_string(),
                "value".to_string(),
                "dense".to_string(),
                "Wqkv".to_string(),
                "Wo".to_string(),
            ],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum, Default)]
#[serde(rename_all = "lowercase")]
pub enum MergeMode {
    #[default]
    Auto,
    Merge,
    NoMerge,
}

impl std::fmt::Display for MergeMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MergeMode::Auto => write!(f, "auto"),
            MergeMode::Merge => write!(f, "merge"),
            MergeMode::NoMerge => write!(f, "no-merge"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingArgs {
    pub base_model: String,
    pub family: Option<String>,
    pub data_path: PathBuf,
    pub eval_path: Option<PathBuf>,
    pub eval_split: f32,
    pub output_dir: PathBuf,
    pub task: TaskType,
    pub epochs: usize,
    pub batch_size: usize,
    pub grad_accum_steps: usize,
    pub lr: f64,
    pub warmup_ratio: f32,
    pub max_grad_norm: f32,
    pub lr_scheduler: String,
    pub max_length: usize,
    pub weight_decay: f32,
    pub seed: u64,
    pub qlora: bool,
    pub bf16: bool,
    pub lora_layers: usize,
    pub unfreeze_layers: usize,
    pub device: String,
    pub lora: LoraConfig,
    pub merge_mode: MergeMode,
    pub quant: Option<String>,
}

fn default_adapt_mode() -> String {
    "full".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptArgs {
    pub from: String,
    pub target: String,
    pub layout: String,
    #[serde(default = "default_adapt_mode")]
    pub mode: String,
    pub data_path: Option<PathBuf>,
    pub output_dir: PathBuf,
    pub epochs: usize,
    pub lr: f64,
    pub batch_size: usize,
    pub device: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeArgs {
    pub base_model: String,
    pub lora_path: PathBuf,
    pub output_dir: PathBuf,
    pub device: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportArgs {
    pub model_path: PathBuf,
    pub format: ExportFormat,
    pub quantize: Option<String>,
    pub output_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainayaMetadata {
    pub version: String,
    pub base_model: String,
    pub task: TaskType,
    pub num_labels: usize,
    pub id2label: std::collections::HashMap<usize, String>,
    pub label2id: std::collections::HashMap<String, usize>,
    pub lora_rank: usize,
    pub lora_alpha: f32,
    pub epochs: usize,
    pub train_samples: usize,
    pub eval_samples: usize,
    pub target_runtime: String,
}
