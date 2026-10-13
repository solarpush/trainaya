pub mod classifier;
pub mod gguf_lora;
pub mod lora;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use candle_transformers::models::modernbert::{Config as ModernBertConfig, ModernBert};
use hf_hub::api::sync::Api;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use crate::config::LoraConfig;
use classifier::{mean_pooling, ClassifierHead};
use lora::LinearLoRA;

pub enum ModelBackbone {
    ModernBert(ModernBert),
    Bert(BertModel),
}

pub struct SequenceClassifier {
    pub backbone: ModelBackbone,
    pub lora_layers: HashMap<String, LinearLoRA>,
    pub head: ClassifierHead,
    pub hidden_size: usize,
    pub num_labels: usize,
    pub device: Device,
}

impl SequenceClassifier {
    /// Load base model weights from local path or HuggingFace Hub
    pub fn load_or_fetch_weights(base_model: &str) -> Result<(PathBuf, PathBuf, PathBuf)> {
        let base_path = Path::new(base_model);

        if base_path.is_dir() {
            let config_path = base_path.join("config.json");
            let tokenizer_path = base_path.join("tokenizer.json");
            let weights_path = base_path.join("model.safetensors");

            if !config_path.exists() {
                bail!("Missing config.json in {}", base_path.display());
            }
            if !tokenizer_path.exists() {
                bail!("Missing tokenizer.json in {}", base_path.display());
            }
            if !weights_path.exists() {
                bail!("Missing model.safetensors in {}", base_path.display());
            }

            return Ok((config_path, tokenizer_path, weights_path));
        }

        // Otherwise fetch from HuggingFace Hub
        println!("  Downloading model files from HuggingFace Hub: '{}'...", base_model);
        let api = Api::new().context("Failed to initialize HuggingFace Hub API")?;
        let repo = api.model(base_model.to_string());

        let config_path = repo.get("config.json").context("Failed to download config.json")?;
        let tokenizer_path = repo
            .get("tokenizer.json")
            .context("Failed to download tokenizer.json")?;
        let weights_path = repo
            .get("model.safetensors")
            .context("Failed to download model.safetensors")?;

        Ok((config_path, tokenizer_path, weights_path))
    }

    /// Build a SequenceClassifier with LoRA attached to target modules
    pub fn build(
        base_model: &str,
        num_labels: usize,
        lora_cfg: &LoraConfig,
        device: &Device,
    ) -> Result<Self> {
        let (config_path, _, weights_path) = Self::load_or_fetch_weights(base_model)?;

        // Read config.json
        let config_file = File::open(&config_path)?;
        let config_json: Value = serde_json::from_reader(config_file)?;
        let model_type = config_json
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("bert")
            .to_lowercase();

        // Load safetensors weights
        let tensors = candle_core::safetensors::load(&weights_path, device)?;

        let mut lora_layers = HashMap::new();

        // Detect target linear layers to inject LoRA into
        for (name, tensor) in &tensors {
            let is_target = lora_cfg.target_modules.iter().any(|tm| name.contains(tm))
                && name.ends_with(".weight");

            if is_target {
                let shape = tensor.shape();
                if shape.dims().len() == 2 {
                    // Check if there is an associated bias
                    let bias_name = name.replace(".weight", ".bias");
                    let bias_tensor = tensors.get(&bias_name).cloned();

                    let lora_layer = LinearLoRA::new(
                        tensor.clone(),
                        bias_tensor,
                        lora_cfg.r,
                        lora_cfg.alpha,
                        device,
                    )?;
                    lora_layers.insert(name.to_string(), lora_layer);
                }
            }
        }

        // Build backbone
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path.clone()], DType::F32, device)?
        };

        let (backbone, hidden_size) = if model_type.contains("modernbert") {
            let config_content = std::fs::read_to_string(&config_path)?;
            let modernbert_cfg: ModernBertConfig = serde_json::from_str(&config_content)?;
            let h_size = modernbert_cfg.hidden_size;
            let model = ModernBert::load(vb.clone(), &modernbert_cfg)?;
            (ModelBackbone::ModernBert(model), h_size)
        } else {
            let config_content = std::fs::read_to_string(&config_path)?;
            let bert_cfg: BertConfig = serde_json::from_str(&config_content)?;
            let h_size = bert_cfg.hidden_size;
            let model = BertModel::load(vb, &bert_cfg)?;
            (ModelBackbone::Bert(model), h_size)
        };

        let head = ClassifierHead::new(hidden_size, num_labels, device)?;

        Ok(Self {
            backbone,
            lora_layers,
            head,
            hidden_size,
            num_labels,
            device: device.clone(),
        })
    }

    /// Forward pass: inputs -> backbone embeddings -> pooling -> head -> logits
    pub fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let hidden_states = match &self.backbone {
            ModelBackbone::ModernBert(mb) => mb.forward(input_ids, attention_mask)?,
            ModelBackbone::Bert(bert) => {
                let token_type_ids = Tensor::zeros_like(input_ids)?;
                bert.forward(input_ids, &token_type_ids, Some(attention_mask))?
            }
        };

        // Mean pooling over token embeddings
        let pooled = mean_pooling(&hidden_states, attention_mask)?;

        // Forward through classification head
        let logits = self.head.forward(&pooled)?;

        Ok(logits)
    }

    /// Collect all trainable parameters for the optimizer (LoRA vars + Head vars)
    pub fn trainable_vars(&self) -> Vec<Var> {
        let mut vars = Vec::new();
        for lora in self.lora_layers.values() {
            vars.extend(lora.vars());
        }
        vars.extend(self.head.vars());
        vars
    }

    /// Save adapter weights (LoRA A, LoRA B, and classification head) to safetensors
    pub fn save_adapter(&self, output_file: &Path) -> Result<()> {
        let mut tensor_map = HashMap::new();

        for (name, lora) in &self.lora_layers {
            lora.export_tensors(name, &mut tensor_map);
        }
        self.head.export_tensors("classifier", &mut tensor_map);

        let parent = output_file.parent().unwrap_or(Path::new("."));
        std::fs::create_dir_all(parent)?;

        candle_core::safetensors::save(&tensor_map, output_file)?;
        Ok(())
    }
}
