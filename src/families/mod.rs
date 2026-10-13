pub mod decoders_head;
pub mod decoders_logit;
pub mod encoders;
pub mod traits;

use anyhow::{bail, Context, Result};
use std::collections::HashMap;

pub use decoders_head::KevFamily;
pub use decoders_logit::{LlmLogitsFamily, WinnowFamily};
pub use encoders::{LayaFamily, NliFamily};
pub use traits::{
    BackendType, DecisionSample, FormattedPrompt, LossStrategy, ModelFamily, ModelMerger,
    PromptBuilder,
};

pub struct FamilyRegistry {
    families: HashMap<String, Box<dyn ModelFamily>>,
}

impl Default for FamilyRegistry {
    fn default() -> Self {
        let mut registry = Self {
            families: HashMap::new(),
        };
        registry.register(Box::new(WinnowFamily));
        registry.register(Box::new(LlmLogitsFamily));
        registry.register(Box::new(KevFamily));
        registry.register(Box::new(NliFamily));
        registry.register(Box::new(LayaFamily));
        registry
    }
}

impl FamilyRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, family: Box<dyn ModelFamily>) {
        self.families
            .insert(family.identifier().to_string(), family);
    }

    pub fn get(&self, identifier: &str) -> Option<&dyn ModelFamily> {
        self.families.get(identifier).map(|f| f.as_ref())
    }

    pub fn list_families(&self) -> Vec<&'static str> {
        self.families.values().map(|f| f.identifier()).collect()
    }

    /// Automatically detect family from base model name, tag, file path, or Ollaya manifest
    pub fn detect_or_resolve(
        &self,
        base_model: &str,
        family_override: Option<&str>,
    ) -> Result<&dyn ModelFamily> {
        if let Some(name) = family_override {
            return self
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("Unknown model family '{}'. Available: {:?}", name, self.list_families()));
        }

        let base_lower = base_model.to_lowercase();

        // 1. Check Ollaya manifest if available
        if let Ok(manifest_family) = Self::inspect_ollaya_manifest(base_model) {
            if let Some(family) = self.get(&manifest_family) {
                return Ok(family);
            }
        }

        // 2. Pattern matching on name/path
        if base_lower.contains("winnow") || base_lower.contains("credence") {
            return self.get("winnow-v1").context("Winnow family not found");
        }
        if base_lower.contains("kev") {
            return self.get("kev-pointer-v1").context("Kev family not found");
        }
        if base_lower.contains("modernbert") || base_lower.contains("deberta") || base_lower.contains("nli") {
            return self.get("nli-pairs-v1").context("NLI family not found");
        }
        if base_lower.contains("laya") {
            return self.get("laya-markers-v1").context("Laya family not found");
        }
        if base_lower.ends_with(".gguf") {
            return self.get("llm-logits-v1").context("LLM Logits family not found");
        }

        // Default to NLI / ModernBERT encoder
        self.get("nli-pairs-v1").context("Default NLI family not found")
    }

    fn inspect_ollaya_manifest(base_model: &str) -> Result<String> {
        let parts: Vec<&str> = base_model.split(':').collect();
        let (family_part, tag_part) = if parts.len() == 2 {
            (parts[0], parts[1])
        } else {
            (base_model, "latest")
        };

        for models_dir in crate::registry::get_candidate_models_dirs() {
            let manifest_path = models_dir
                .join("manifests/ollaya.dev/library")
                .join(family_part)
                .join(tag_part);

            if manifest_path.exists() {
                let content = std::fs::read_to_string(&manifest_path)?;
                let val: serde_json::Value = serde_json::from_str(&content)?;

                // Inspect layout label in config/annotations
                if let Some(labels) = val.get("annotations").and_then(|a| a.as_object()) {
                    if let Some(layout) = labels.get("org.ollaya.layout").and_then(|l| l.as_str()) {
                        return Ok(layout.to_string());
                    }
                }

                // Or deduce from family tag
                if family_part == "winnow" {
                    return Ok("winnow-v1".to_string());
                }
                if family_part == "kev" {
                    return Ok("kev-pointer-v1".to_string());
                }
                if family_part == "nli" {
                    return Ok("nli-pairs-v1".to_string());
                }
            }
        }

        bail!("No manifest found for {}", base_model)
    }
}
