use anyhow::Result;
use std::path::{Path, PathBuf};

use crate::config::MergeMode;
use crate::families::traits::{
    BackendType, DecisionSample, FormattedPrompt, ModelFamily, ModelMerger, PromptBuilder,
};

pub struct LayaPromptBuilder;

impl PromptBuilder for LayaPromptBuilder {
    fn format_sample(&self, sample: &DecisionSample) -> Result<FormattedPrompt> {
        let state_str = match &sample.state {
            serde_json::Value::String(s) => s.clone(),
            v => serde_json::to_string(v)?,
        };

        // Layout: [CLS] question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 [SEP] <state> [SEP]
        let mut opts_with_masks = String::new();
        for opt in &sample.options {
            opts_with_masks.push_str(&format!("[MASK] {} ", opt));
        }

        let prompt = format!(
            "question: {} [SEP] {}[SEP] {}",
            sample.question,
            opts_with_masks.trim_end(),
            state_str
        );

        Ok(FormattedPrompt {
            text: prompt,
            token_ids: None,
            target_index: sample.target_index,
            option_markers: Vec::new(),
            decide_marker: None,
        })
    }
}

pub struct LayaMerger;

impl ModelMerger for LayaMerger {
    fn can_merge(&self) -> bool {
        true
    }

    fn merge(
        &self,
        base_path: &Path,
        adapter_path: &Path,
        output_path: &Path,
        _quant_format: Option<&str>,
    ) -> Result<PathBuf> {
        let merge_args = crate::config::MergeArgs {
            base_model: base_path.to_string_lossy().to_string(),
            lora_path: adapter_path.to_path_buf(),
            output_dir: output_path.to_path_buf(),
            device: "cpu".to_string(),
        };
        let summary = crate::merger::run_merge(merge_args)?;
        Ok(summary.output_file)
    }
}

pub struct LayaFamily;

impl Default for LayaFamily {
    fn default() -> Self {
        Self
    }
}

impl ModelFamily for LayaFamily {
    fn identifier(&self) -> &'static str {
        "laya-markers-v1"
    }

    fn default_layout(&self) -> &'static str {
        "laya-markers-v1"
    }

    fn backend_type(&self) -> BackendType {
        BackendType::CandleSafeTensors
    }

    fn default_lora_targets(&self) -> &[&'static str] {
        &["query", "key", "value", "dense"]
    }

    fn default_merge_mode(&self) -> MergeMode {
        MergeMode::Merge
    }

    fn create_prompt_builder(&self) -> Box<dyn PromptBuilder> {
        Box::new(LayaPromptBuilder)
    }

    fn merger(&self) -> Box<dyn ModelMerger> {
        Box::new(LayaMerger)
    }
}
