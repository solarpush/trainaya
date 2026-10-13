use anyhow::Result;
use std::path::{Path, PathBuf};

use crate::config::MergeMode;
use crate::families::traits::{
    BackendType, DecisionSample, FormattedPrompt, ModelFamily, ModelMerger, PromptBuilder,
};

pub struct KevPromptBuilder {
    pub max_state_tokens: usize,
}

impl Default for KevPromptBuilder {
    fn default() -> Self {
        Self {
            max_state_tokens: 2048,
        }
    }
}

impl PromptBuilder for KevPromptBuilder {
    fn format_sample(&self, sample: &DecisionSample) -> Result<FormattedPrompt> {
        let state_str = match &sample.state {
            serde_json::Value::String(s) => s.clone(),
            v => serde_json::to_string(v)?,
        };

        // Kev layout:
        // <|fim_prefix|> state <|fim_middle|> question (<|box_start|> opt_j <|box_end|>)* <|fim_suffix|>
        let mut prompt = format!(
            "<|fim_prefix|>{}<|fim_middle|>{}",
            state_str, sample.question
        );

        for opt in &sample.options {
            prompt.push_str(&format!("<|box_start|>{}<|box_end|>", opt));
        }
        prompt.push_str("<|fim_suffix|>");

        Ok(FormattedPrompt {
            text: prompt,
            token_ids: None,
            target_index: sample.target_index,
            option_markers: Vec::new(),
            decide_marker: None,
        })
    }
}

pub struct KevMerger;

impl ModelMerger for KevMerger {
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
        // SafeTensors linear weight fusion for Qwen base + LoRA adapter
        println!("  Fusing Kev LoRA adapter into base weights at {:?}...", output_path);
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

pub struct KevFamily;

impl Default for KevFamily {
    fn default() -> Self {
        Self
    }
}

impl ModelFamily for KevFamily {
    fn identifier(&self) -> &'static str {
        "kev-pointer-v1"
    }

    fn default_layout(&self) -> &'static str {
        "kev-pointer-v1"
    }

    fn backend_type(&self) -> BackendType {
        BackendType::CandleSafeTensors
    }

    fn default_lora_targets(&self) -> &[&'static str] {
        &[
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
        ]
    }

    /// For Kev, unmerged is the default matching upstream Ollaya runtime
    fn default_merge_mode(&self) -> MergeMode {
        MergeMode::NoMerge
    }

    fn create_prompt_builder(&self) -> Box<dyn PromptBuilder> {
        Box::new(KevPromptBuilder::default())
    }

    fn merger(&self) -> Box<dyn ModelMerger> {
        Box::new(KevMerger)
    }
}
