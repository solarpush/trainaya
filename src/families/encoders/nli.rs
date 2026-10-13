use anyhow::Result;
use std::path::{Path, PathBuf};

use crate::config::MergeMode;
use crate::families::traits::{
    BackendType, DecisionSample, FormattedPrompt, ModelFamily, ModelMerger, PromptBuilder,
};

pub struct NliPromptBuilder {
    pub hypothesis_template: String,
}

impl Default for NliPromptBuilder {
    fn default() -> Self {
        Self {
            hypothesis_template: "This state implies: {}".to_string(),
        }
    }
}

impl PromptBuilder for NliPromptBuilder {
    fn format_sample(&self, sample: &DecisionSample) -> Result<FormattedPrompt> {
        let state_str = match &sample.state {
            serde_json::Value::String(s) => s.clone(),
            v => serde_json::to_string(v)?,
        };

        // For NLI pairs: [CLS] <state> [SEP] <hypothesis> [SEP]
        let target_option = sample
            .options
            .get(sample.target_index)
            .map(|s| s.as_str())
            .unwrap_or("none");
        let hypothesis = self.hypothesis_template.replace("{}", target_option);

        let text = format!("{} [SEP] {}", state_str, hypothesis);

        Ok(FormattedPrompt {
            text,
            token_ids: None,
            target_index: sample.target_index,
            option_markers: Vec::new(),
            decide_marker: None,
        })
    }
}

pub struct NliMerger;

impl ModelMerger for NliMerger {
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
        println!("  Fusing NLI LoRA adapter into base weights at {:?}...", output_path);
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

pub struct NliFamily;

impl Default for NliFamily {
    fn default() -> Self {
        Self
    }
}

impl ModelFamily for NliFamily {
    fn identifier(&self) -> &'static str {
        "nli-pairs-v1"
    }

    fn default_layout(&self) -> &'static str {
        "nli-pairs-v1"
    }

    fn backend_type(&self) -> BackendType {
        BackendType::CandleSafeTensors
    }

    fn default_lora_targets(&self) -> &[&'static str] {
        &["query", "key", "value", "dense", "Wqkv", "Wo"]
    }

    fn default_merge_mode(&self) -> MergeMode {
        MergeMode::Merge
    }

    fn create_prompt_builder(&self) -> Box<dyn PromptBuilder> {
        Box::new(NliPromptBuilder::default())
    }

    fn merger(&self) -> Box<dyn ModelMerger> {
        Box::new(NliMerger)
    }
}
