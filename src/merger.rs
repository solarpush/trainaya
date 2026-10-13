use anyhow::{bail, Context, Result};
use candle_core::{DType, Tensor};
use colored::Colorize;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;

use crate::quant::lora_merge_accumulate;
use crate::config::MergeArgs;
use crate::models::SequenceClassifier;
use crate::utils::{print_banner, print_status, print_success, select_device};

pub struct MergeSummary {
    pub total_base_tensors: usize,
    pub merged_layers_count: usize,
    pub output_file: PathBuf,
}

pub fn run_merge(args: MergeArgs) -> Result<MergeSummary> {
    print_banner("Permanent Weights Merging (Base + LoRA -> SafeTensors)");

    let device = select_device(&args.device)?;
    print_status("Base Model", &args.base_model);
    print_status("LoRA Adapter Dir", &args.lora_path.display().to_string());
    print_status("Output Dir", &args.output_dir.display().to_string());
    print_status("Compute Device", &format!("{:?}", device));

    let is_gguf = args.base_model.ends_with(".gguf")
        || args.base_model.starts_with("winnow")
        || args.lora_path.extension().map_or(false, |ext| ext == "gguf")
        || args.lora_path.join("adapter.gguf").exists();

    if is_gguf {
        let registry = crate::families::FamilyRegistry::default();
        let family = registry.detect_or_resolve(&args.base_model, None)?;
        let base_path = crate::models::gguf_lora::resolve_gguf_path(&args.base_model)?;
        let adapter_file = if args.lora_path.is_file() {
            args.lora_path.clone()
        } else {
            args.lora_path.join("adapter.gguf")
        };
        std::fs::create_dir_all(&args.output_dir)?;
        let out_file = args.output_dir.join("model.gguf");
        let merged_file = family.merger().merge(&base_path, &adapter_file, &out_file, None)?;
        print_success(&format!("Successfully merged GGUF model: {}", merged_file.display()));
        return Ok(MergeSummary {
            total_base_tensors: 1,
            merged_layers_count: 1,
            output_file: merged_file,
        });
    }

    // 1. Check LoRA files
    let adapter_weights_file = args.lora_path.join("adapter_model.safetensors");
    let adapter_config_file = args.lora_path.join("adapter_config.json");

    if !adapter_weights_file.exists() {
        bail!(
            "Missing adapter_model.safetensors in {}",
            args.lora_path.display()
        );
    }

    // Read adapter config for r and alpha
    let (lora_r, lora_alpha) = if adapter_config_file.exists() {
        let f = File::open(&adapter_config_file)?;
        let val: Value = serde_json::from_reader(f)?;
        let r = val.get("r").and_then(|v| v.as_u64()).unwrap_or(16) as usize;
        let alpha = val.get("lora_alpha").and_then(|v| v.as_f64()).unwrap_or(32.0) as f32;
        (r, alpha)
    } else {
        (16, 32.0)
    };
    let scale = lora_alpha / (lora_r as f32);

    print_status("LoRA Rank (r)", &lora_r.to_string());
    print_status("LoRA Alpha", &lora_alpha.to_string());
    print_status("Merge Scale", &format!("{:.4}", scale));

    // 2. Load Base Model Weights & Config
    let (config_path, tokenizer_path, base_weights_path) =
        SequenceClassifier::load_or_fetch_weights(&args.base_model)?;

    println!("\n  Loading base model safetensors...");
    let base_tensors = candle_core::safetensors::load(&base_weights_path, &device)?;

    println!("  Loading LoRA adapter safetensors...");
    let adapter_tensors = candle_core::safetensors::load(&adapter_weights_file, &device)?;

    // 3. Perform Merge
    let mut merged_tensors: HashMap<String, Tensor> = HashMap::new();
    let mut merged_count = 0usize;

    for (name, base_tensor) in base_tensors {
        let mut updated_tensor = base_tensor.to_dtype(DType::F32)?;

        // Check if there is an adapter for this layer
        let lora_a_name = format!("{}.lora_a.weight", name);
        let lora_b_name = format!("{}.lora_b.weight", name);

        if let (Some(a_tensor), Some(b_tensor)) = (adapter_tensors.get(&lora_a_name), adapter_tensors.get(&lora_b_name)) {
            let (out_features, in_features) = match updated_tensor.shape().dims() {
                [out_f, in_f] => (*out_f, *in_f),
                _ => (0, 0),
            };

            if in_features > 0 && out_features > 0 {
                let mut base_vec = updated_tensor.flatten_all()?.to_vec1::<f32>()?;
                let a_vec = a_tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                let b_vec = b_tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;

                lora_merge_accumulate(
                    &mut base_vec,
                    &a_vec,
                    &b_vec,
                    in_features,
                    out_features,
                    lora_r,
                    scale,
                )?;

                updated_tensor = Tensor::from_vec(base_vec, (out_features, in_features), &device)?;
            } else {
                // Fallback to Tensor matmul: W += scale * (B @ A)
                let delta = b_tensor.matmul(a_tensor)?;
                let delta_scaled = (delta * (scale as f64))?;
                updated_tensor = updated_tensor.broadcast_add(&delta_scaled)?;
            }

            merged_count += 1;
        }

        merged_tensors.insert(name, updated_tensor);
    }

    // Also import classifier head weights if present in adapter
    for (name, t) in &adapter_tensors {
        if name.starts_with("classifier.") {
            merged_tensors.insert(name.clone(), t.clone());
        }
    }

    // 4. Save Merged Model
    std::fs::create_dir_all(&args.output_dir)?;
    let output_weights_file = args.output_dir.join("model.safetensors");

    println!(
        "\n  Writing {} merged tensors to {}...",
        merged_tensors.len().to_string().cyan(),
        output_weights_file.display().to_string().green()
    );

    candle_core::safetensors::save(&merged_tensors, &output_weights_file)
        .context("Failed to serialize merged safetensors file")?;

    // Copy configs & tokenizer
    let out_config = args.output_dir.join("config.json");
    let out_tokenizer = args.output_dir.join("tokenizer.json");

    if config_path.exists() {
        std::fs::copy(&config_path, &out_config)?;
    }
    if tokenizer_path.exists() {
        std::fs::copy(&tokenizer_path, &out_tokenizer)?;
    }

    // If metadata exists, copy it as well
    let meta_src = args.lora_path.join("trainaya_metadata.json");
    if meta_src.exists() {
        std::fs::copy(&meta_src, args.output_dir.join("trainaya_metadata.json"))?;
    }

    print_success(&format!(
        "Merge completed: {} layers successfully updated into full-rank safetensors.",
        merged_count
    ));

    Ok(MergeSummary {
        total_base_tensors: merged_tensors.len(),
        merged_layers_count: merged_count,
        output_file: output_weights_file,
    })
}
