use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::config::MergeMode;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendType {
    CandleSafeTensors,
    Gguf,
    Onnx,
}

/// Standardized decision sample across all model families
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionSample {
    pub state: serde_json::Value,
    pub question: String,
    pub options: Vec<String>,
    pub target_index: usize,
}

/// Formatted prompt ready for tokenization or inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormattedPrompt {
    pub text: String,
    pub token_ids: Option<Vec<u32>>,
    pub target_index: usize,
    /// Option closing or mask positions (e.g. for pointer heads or masked tokens)
    pub option_markers: Vec<usize>,
    /// Question decide marker position (e.g. Kev <|fim_suffix|>)
    pub decide_marker: Option<usize>,
}

/// Formats decision samples into family-specific sequences and delimiter tokens
pub trait PromptBuilder: Send + Sync {
    fn format_sample(&self, sample: &DecisionSample) -> Result<FormattedPrompt>;
}

/// Strategy for forward pass scoring and loss computation
pub trait LossStrategy: Send + Sync {
    fn name(&self) -> &'static str;
}

/// Strategy for merging LoRA adapter weights with the base model
pub trait ModelMerger: Send + Sync {
    fn can_merge(&self) -> bool;

    fn merge(
        &self,
        base_path: &Path,
        adapter_path: &Path,
        output_path: &Path,
        quant_format: Option<&str>,
    ) -> Result<PathBuf>;
}

/// High-level model family definition
pub trait ModelFamily: Send + Sync {
    /// Family identifier (e.g. 'winnow-v1', 'kev-pointer-v1', 'nli-pairs-v1')
    fn identifier(&self) -> &'static str;

    /// Associated Ollaya inference layout
    fn default_layout(&self) -> &'static str;

    /// Primary execution backend
    fn backend_type(&self) -> BackendType;

    /// Default linear modules to target with LoRA
    fn default_lora_targets(&self) -> &[&'static str];

    /// Default merge mode when --merge is set to 'auto'
    fn default_merge_mode(&self) -> MergeMode;

    /// Factory for the family's prompt builder
    fn create_prompt_builder(&self) -> Box<dyn PromptBuilder>;

    /// Model weight merge pipeline
    fn merger(&self) -> Box<dyn ModelMerger>;
}
