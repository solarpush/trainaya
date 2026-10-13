use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::config::MergeMode;
use crate::families::traits::{
    BackendType, DecisionSample, FormattedPrompt, ModelFamily, ModelMerger, PromptBuilder,
};

pub struct GenericLlmPromptBuilder;

impl PromptBuilder for GenericLlmPromptBuilder {
    fn format_sample(&self, sample: &DecisionSample) -> Result<FormattedPrompt> {
        let state_str = match &sample.state {
            serde_json::Value::String(s) => s.clone(),
            v => serde_json::to_string(v)?,
        };

        let mut opt_str = String::new();
        let letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"];
        for (i, opt) in sample.options.iter().enumerate() {
            let letter = if i < letters.len() { letters[i] } else { "?" };
            opt_str.push_str(&format!("{}: {}\n", letter, opt));
        }

        let prompt = format!(
            "Context: {}\n\nQuestion: {}\nOptions:\n{}Answer:\n",
            state_str, sample.question, opt_str
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

pub struct GenericLlmMerger;

impl ModelMerger for GenericLlmMerger {
    fn can_merge(&self) -> bool {
        true
    }

    fn merge(
        &self,
        base_path: &Path,
        adapter_path: &Path,
        output_path: &Path,
        quant_format: Option<&str>,
    ) -> Result<PathBuf> {
        let export_lora_bin = crate::utils::find_llama_tool("llama-export-lora");
        let quantize_bin = crate::utils::find_llama_tool("llama-quantize");

        let quant = quant_format.unwrap_or("Q8_0");

        if let (Some(export_lora), Some(quantize)) = (export_lora_bin, quantize_bin) {
            let tmp_f16 = output_path.with_extension("merged_f16.tmp");

            let status_merge = Command::new(&export_lora)
                .arg("-m")
                .arg(base_path)
                .arg("--lora")
                .arg(adapter_path)
                .arg("-o")
                .arg(&tmp_f16)
                .status()
                .context("Failed to execute llama-export-lora")?;

            if !status_merge.success() {
                bail!("llama-export-lora exited with code: {:?}", status_merge.code());
            }

            let status_quant = Command::new(&quantize)
                .arg(&tmp_f16)
                .arg(output_path)
                .arg(quant)
                .status()
                .context("Failed to execute llama-quantize")?;

            let _ = std::fs::remove_file(&tmp_f16);

            if !status_quant.success() {
                bail!("llama-quantize exited with code: {:?}", status_quant.code());
            }

            Ok(output_path.to_path_buf())
        } else {
            bail!("llama.cpp native merge binaries not found in vendor directory.");
        }
    }
}

pub struct LlmLogitsFamily;

impl Default for LlmLogitsFamily {
    fn default() -> Self {
        Self
    }
}

impl ModelFamily for LlmLogitsFamily {
    fn identifier(&self) -> &'static str {
        "llm-logits-v1"
    }

    fn default_layout(&self) -> &'static str {
        "llm-logits-v1"
    }

    fn backend_type(&self) -> BackendType {
        BackendType::Gguf
    }

    fn default_lora_targets(&self) -> &[&'static str] {
        &["attn_output", "ffn_down"]
    }

    fn default_merge_mode(&self) -> MergeMode {
        MergeMode::Merge
    }

    fn create_prompt_builder(&self) -> Box<dyn PromptBuilder> {
        Box::new(GenericLlmPromptBuilder)
    }

    fn merger(&self) -> Box<dyn ModelMerger> {
        Box::new(GenericLlmMerger)
    }
}
