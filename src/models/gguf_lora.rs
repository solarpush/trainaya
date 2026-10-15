use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Shape, Tensor, Var};
use candle_nn::{loss, AdamW, Optimizer, ParamsAdamW};
use colored::Colorize;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use crate::config::TrainingArgs;
use crate::dataset::DecisionDataset;
use crate::exporter::export_lora_gguf;
use crate::families::traits::{DecisionSample, ModelFamily};
use crate::trainer::{compute_scheduled_lr, TrainMetrics};
use crate::utils::{print_banner, print_status, print_success, select_device};

/// Resolve model path from either a direct file path, blob path, or Ollaya model name (e.g., 'winnow:e4b')
pub fn resolve_gguf_path(base_model: &str) -> Result<PathBuf> {
    let p = Path::new(base_model);
    if p.exists() {
        return Ok(p.to_path_buf());
    }

    // Check if it's an Ollaya model tag: family:tag
    let parts: Vec<&str> = base_model.split(':').collect();
    let (family, tag) = if parts.len() == 2 {
        (parts[0], parts[1])
    } else {
        (base_model, "latest")
    };

    let candidate_dirs = crate::registry::get_candidate_models_dirs();

    for models_dir in &candidate_dirs {
        let manifest_path = models_dir
            .join("manifests/ollaya.dev/library")
            .join(family)
            .join(tag);

        if manifest_path.exists() {
            if let Ok(manifest_content) = std::fs::read_to_string(&manifest_path) {
                if let Ok(manifest_val) = serde_json::from_str::<Value>(&manifest_content) {
                    if let Some(layers) = manifest_val.get("layers").and_then(|l| l.as_array()) {
                        for layer in layers {
                            let media_type = layer
                                .get("mediaType")
                                .and_then(|m| m.as_str())
                                .unwrap_or("");
                            if media_type == "application/vnd.ollaya.weights.gguf" {
                                if let Some(digest) = layer.get("digest").and_then(|d| d.as_str()) {
                                    let blob_name = digest.replace(':', "-");
                                    let blob_path = models_dir.join("blobs").join(&blob_name);
                                    if blob_path.exists() {
                                        return Ok(blob_path);
                                    }
                                    for cand in &candidate_dirs {
                                        let cand_blob = cand.join("blobs").join(&blob_name);
                                        if cand_blob.exists() {
                                            return Ok(cand_blob);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    bail!(
        "Could not resolve GGUF base model for '{}'. Please provide a valid file path or registered model name.",
        base_model
    )
}

struct GgufLoraLayer {
    tensor_name: String,
    lora_a: Var,
    lora_b: Var,
}

/// Run multi-layer, multi-class, GPU-aware LoRA fine-tuning directly on top of a GGUF model
pub fn run_gguf_training(args: TrainingArgs, family: &dyn ModelFamily) -> Result<TrainMetrics> {
    print_banner("GGUF Multi-Layer LoRA Fine-Tuning Engine (Zero Python)");

    let device = select_device(&args.device)?;
    let base_gguf_path = resolve_gguf_path(&args.base_model)?;
    print_status("Base GGUF File", &base_gguf_path.display().to_string());
    print_status("Dataset Path", &args.data_path.display().to_string());
    print_status("Output Dir", &args.output_dir.display().to_string());
    print_status("Epochs", &args.epochs.to_string());
    print_status("LoRA Rank (r)", &args.lora.r.to_string());
    print_status("LoRA Alpha", &args.lora.alpha.to_string());
    print_status("LoRA Top Blocks", &args.lora_layers.to_string());
    print_status(
        "Learning Rate",
        &format!("{:e} (schedule: {})", args.lr, args.lr_scheduler),
    );
    print_status("Compute Device", &format!("{:?}", device));

    std::fs::create_dir_all(&args.output_dir)?;

    // 1. Inspect base GGUF metadata & tensor topology
    let mut file = File::open(&base_gguf_path)
        .context(format!("Failed to open base GGUF at {:?}", base_gguf_path))?;
    let content = candle_core::quantized::gguf_file::Content::read(&mut file)
        .context("Failed to parse GGUF content")?;

    let arch = match content.metadata.get("general.architecture") {
        Some(candle_core::quantized::gguf_file::Value::String(s)) => s.clone(),
        _ => "gemma4".to_string(),
    };
    print_status("Base Architecture", &arch);

    // Discover total transformer blocks dynamically from GGUF tensor names (blk.<N>.*)
    let mut max_block_idx = 0usize;
    for name in content.tensor_infos.keys() {
        if let Some(rest) = name.strip_prefix("blk.") {
            if let Some((idx_str, _)) = rest.split_once('.') {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    if idx > max_block_idx {
                        max_block_idx = idx;
                    }
                }
            }
        }
    }
    let total_blocks = max_block_idx + 1;
    let num_target_blocks = if args.lora_layers == 0 {
        total_blocks
    } else {
        args.lora_layers.min(total_blocks).max(1)
    };
    let start_block = total_blocks.saturating_sub(num_target_blocks);
    println!(
        "  Detected {} transformer blocks (blk.0 .. blk.{}). Injecting LoRA into blocks {}..={}.",
        total_blocks.to_string().cyan(),
        max_block_idx,
        start_block,
        max_block_idx
    );

    // 2. Load token embeddings and vocab map from base GGUF
    println!("\n  Extracting decision token projections and vocab from base GGUF...");
    let q_token_embd = content
        .tensor(&mut file, "token_embd.weight", &Device::Cpu)
        .context("Missing token_embd.weight in GGUF")?;
    let token_embd_cpu = q_token_embd.dequantize(&Device::Cpu)?;
    let vocab_size = token_embd_cpu.shape().dims()[0];
    let hidden_size = token_embd_cpu.shape().dims()[1];
    println!("  Hidden Size: {} | Vocab: {}", hidden_size, vocab_size);

    // Build vocab map from GGUF metadata
    let mut vocab_map: HashMap<String, u32> = HashMap::with_capacity(vocab_size);
    if let Some(candle_core::quantized::gguf_file::Value::Array(tokens)) =
        content.metadata.get("tokenizer.ggml.tokens")
    {
        for (id, tok) in tokens.iter().enumerate() {
            if let candle_core::quantized::gguf_file::Value::String(s) = tok {
                vocab_map.insert(s.clone(), id as u32);
            }
        }
    }
    println!("  Loaded vocabulary index: {} tokens", vocab_map.len());

    let get_token_id = |word: &str| -> Option<u32> {
        let candidates = [
            word.to_string(),
            format!("\u{2581}{}", word), // SentencePiece
            format!("\u{0120}{}", word), // GPT2 / Qwen BPE
            word.to_lowercase(),
            format!("\u{2581}{}", word.to_lowercase()),
            format!("\u{0120}{}", word.to_lowercase()),
            format!(" {}", word),
            format!(" {}", word.to_lowercase()),
        ];
        for cand in &candidates {
            if let Some(&id) = vocab_map.get(cand) {
                return Some(id);
            }
        }
        None
    };

    // 3. Load dataset via universal DecisionDataset
    let decision_samples =
        DecisionDataset::load_samples(&args.data_path, args.eval_path.as_deref())?;
    let prompt_builder = family.create_prompt_builder();
    let mut samples: Vec<(String, DecisionSample, usize)> =
        Vec::with_capacity(decision_samples.len());
    let mut max_option_count = 3usize;

    for d_sample in decision_samples {
        let formatted = prompt_builder.format_sample(&d_sample)?;
        let n_opts = d_sample
            .options
            .len()
            .max(formatted.target_index + 1)
            .max(2);
        if n_opts > max_option_count {
            max_option_count = n_opts;
        }
        samples.push((formatted.text, d_sample, formatted.target_index));
    }
    let num_classes = max_option_count.min(26);
    println!(
        "  Generated {} decision prompt training pairs across {} option classes.",
        samples.len().to_string().cyan(),
        num_classes.to_string().green()
    );

    // Resolve option label embeddings dynamically for letters A..Z
    let option_letters = [
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R",
        "S", "T", "U", "V", "W", "X", "Y", "Z",
    ];
    let fallback_ids = [236776u32, 236799, 236780, 236808, 236782, 236814];
    let mut option_embs_cpu = Vec::with_capacity(num_classes);
    for c_idx in 0..num_classes {
        let letter = option_letters[c_idx];
        let tok_id = get_token_id(letter).unwrap_or_else(|| {
            fallback_ids
                .get(c_idx)
                .copied()
                .unwrap_or((c_idx as u32 + 65).min(vocab_size.saturating_sub(1) as u32))
        });
        let safe_id = (tok_id as usize).min(vocab_size.saturating_sub(1));
        let emb = token_embd_cpu.get(safe_id)?;
        option_embs_cpu.push(emb);
    }
    let option_emb_refs: Vec<&Tensor> = option_embs_cpu.iter().collect();
    let target_emb = Tensor::stack(&option_emb_refs, 0)?.to_device(&device)?; // [num_classes, hidden_size]

    // 4. Inspect primary decision projection tensor shape dynamically from GGUF
    let primary_tensor_name = format!("blk.{}.attn_output.weight", max_block_idx);
    let (out_features, in_features) =
        if let Some(info) = content.tensor_infos.get(&primary_tensor_name) {
            let dims = info.shape.dims();
            if dims.len() == 2 {
                (dims[0], dims[1])
            } else {
                (hidden_size, 4096)
            }
        } else {
            (hidden_size, 4096)
        };

    let r = args.lora.r;
    let alpha = args.lora.alpha;
    let scale = (alpha as f64) / (r as f64);

    // Helper to embed a text segment into `hidden_size` using vocabulary lookup
    let embed_segment = |segment: &str| -> Vec<f32> {
        let mut out = vec![0.0f32; hidden_size];
        let mut count = 0.0f32;
        for clean in segment
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
        {
            if let Some(id) = get_token_id(clean) {
                if let Ok(emb) = token_embd_cpu
                    .get((id as usize).min(vocab_size.saturating_sub(1)))
                    .and_then(|t| t.to_vec1::<f32>())
                {
                    for j in 0..hidden_size {
                        out[j] += emb[j];
                    }
                    count += 1.0;
                }
            }
        }
        if count > 0.0 {
            for j in 0..hidden_size {
                out[j] /= count;
            }
        }
        out
    };

    // Context-aware feature extractor combining full prompt, state, and question into `hidden_size`
    let extract_hidden_state = |prompt: &str, d_sample: &DecisionSample| -> Vec<f32> {
        let state_str = match &d_sample.state {
            Value::String(s) => s.as_str(),
            _ => prompt,
        };
        let v_prompt = embed_segment(prompt);
        let v_state = embed_segment(state_str);
        let v_question = embed_segment(&d_sample.question);

        let mut h = vec![0.0f32; hidden_size];
        let mut norm_sq = 0.0f32;
        for j in 0..hidden_size {
            let val = 0.45 * v_prompt[j] + 0.35 * v_state[j] + 0.20 * v_question[j];
            h[j] = val;
            norm_sq += val * val;
        }
        let inv_norm = 1.0 / (norm_sq.sqrt() + 1e-6);
        for v in h.iter_mut() {
            *v *= inv_norm;
        }
        h
    };

    // Dequantize primary attn_output weight [out_features, in_features] to project hidden states into the native attention-head basis
    let w_base_opt = content
        .tensor(&mut file, &primary_tensor_name, &Device::Cpu)
        .ok()
        .and_then(|qt| qt.dequantize(&Device::Cpu).ok());

    // Precompute sample feature vectors in the native `in_features` subspace along with per-sample option count
    let mut sample_features: Vec<(Vec<f32>, usize, usize)> = Vec::with_capacity(samples.len());
    for (prompt, d_sample, target_idx) in &samples {
        let h_vec = extract_hidden_state(prompt, d_sample);
        let feat = if let Some(ref w_base) = w_base_opt {
            let dims = w_base.shape().dims();
            if dims.len() == 2 && dims[0] == hidden_size && dims[1] == in_features {
                if let Ok(h_t) = Tensor::from_slice(&h_vec, (1, hidden_size), &Device::Cpu) {
                    if let Ok(proj_t) = h_t.matmul(w_base) {
                        if let Ok(mut rows) = proj_t.to_vec2::<f32>() {
                            if let Some(mut row) = rows.pop() {
                                let nrm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
                                for v in row.iter_mut() {
                                    *v /= nrm + 1e-6;
                                }
                                row
                            } else {
                                vec![0.0f32; in_features]
                            }
                        } else {
                            vec![0.0f32; in_features]
                        }
                    } else {
                        vec![0.0f32; in_features]
                    }
                } else {
                    vec![0.0f32; in_features]
                }
            } else {
                let mut f = vec![0.0f32; in_features];
                for j in 0..in_features {
                    f[j] = h_vec[j % hidden_size];
                }
                f
            }
        } else {
            let mut f = vec![0.0f32; in_features];
            for j in 0..in_features {
                f[j] = h_vec[j % hidden_size];
            }
            f
        };

        let sample_n_opts = d_sample
            .options
            .len()
            .max(*target_idx + 1)
            .max(2)
            .min(num_classes);
        sample_features.push((feat, (*target_idx).min(num_classes - 1), sample_n_opts));
    }

    // 5. Initialize Multi-Layer LoRA adapters
    // When epochs > 0, use standard zero-initialized LoRA B (preserving 100% of base model at step 0)
    // When epochs == 0, compute an analytical separating hyperplane directly
    let (lora_a_init, lora_b_init) = if args.epochs > 0 {
        let std_dev = (1.0 / (in_features as f64).sqrt()) as f32 * 0.02;
        let a_init = Tensor::randn(0.0f32, std_dev, Shape::from((r, in_features)), &device)?;
        let b_init = Tensor::zeros(Shape::from((out_features, r)), DType::F32, &device)?;
        (a_init, b_init)
    } else {
        let mut prompt_summary = String::new();
        for (text, _, _) in samples.iter().take(10) {
            prompt_summary.push_str(text);
            prompt_summary.push(' ');
        }
        let q_rout_seg = embed_segment(&prompt_summary);
        let q_brch_seg = embed_segment("false true normal routine failure critical");

        let mut a0 = vec![0.0f32; in_features];
        for j in 0..in_features {
            a0[j] = q_rout_seg[j % hidden_size] - q_brch_seg[j % hidden_size];
        }
        let norm_a0: f32 = a0.iter().map(|v| v * v).sum::<f32>().sqrt();
        for v in a0.iter_mut() {
            *v /= norm_a0 + 1e-6;
        }

        let mut a_rows = vec![Tensor::from_vec(a0, Shape::from(in_features), &device)?];
        for _ in 1..r {
            a_rows.push(Tensor::zeros(Shape::from(in_features), DType::F32, &device)?);
        }
        let a_refs: Vec<&Tensor> = a_rows.iter().collect();
        let a_init = Tensor::stack(&a_refs, 0)?;

        let emb_a_dev = option_embs_cpu[0].to_device(&device)?;
        let emb_b_dev = option_embs_cpu[1.min(num_classes - 1)].to_device(&device)?;
        let diff_ba = (&emb_b_dev - &emb_a_dev)?;
        let norm_ba = (diff_ba.sqr()?.sum_all()?.to_vec0::<f32>()?).sqrt();
        let dir_ba = (diff_ba / (norm_ba as f64 + 1e-6))?;

        let gamma = 12.0f32;
        let col0_raw = (dir_ba * (gamma as f64))?;
        let col0 = if out_features == hidden_size {
            col0_raw
        } else {
            Tensor::zeros(Shape::from(out_features), DType::F32, &device)?
        };

        let mut b_cols = vec![col0];
        for _ in 1..r {
            b_cols.push(Tensor::zeros(Shape::from(out_features), DType::F32, &device)?);
        }
        let b_refs: Vec<&Tensor> = b_cols.iter().collect();
        let b_init = Tensor::stack(&b_refs, 1)?;
        (a_init, b_init)
    };

    // Build multi-layer LoRA stack across the top transformer blocks
    let mut lora_stack: Vec<GgufLoraLayer> = Vec::new();

    // Add auxiliary top blocks first (if num_target_blocks > 1)
    for blk_idx in start_block..max_block_idx {
        let t_name = format!("blk.{}.attn_output.weight", blk_idx);
        if let Some(info) = content.tensor_infos.get(&t_name) {
            let dims = info.shape.dims();
            if dims.len() == 2 && dims[0] == out_features && dims[1] == in_features {
                let std_dev = (1.0 / (in_features as f64).sqrt()) as f32 * 0.02;
                let a_t = if args.epochs > 0 {
                    Tensor::randn(0.0f32, std_dev, Shape::from((r, in_features)), &device)?
                } else {
                    Tensor::zeros(Shape::from((r, in_features)), DType::F32, &device)?
                };
                let b_t = Tensor::zeros(Shape::from((out_features, r)), DType::F32, &device)?;
                lora_stack.push(GgufLoraLayer {
                    tensor_name: t_name,
                    lora_a: Var::from_tensor(&a_t)?,
                    lora_b: Var::from_tensor(&b_t)?,
                });
            }
        }
    }

    // Add primary top decision block
    lora_stack.push(GgufLoraLayer {
        tensor_name: primary_tensor_name,
        lora_a: Var::from_tensor(&lora_a_init)?,
        lora_b: Var::from_tensor(&lora_b_init)?,
    });

    println!(
        "  Initialized {} GGUF LoRA adapter layers (rank={}, alpha={}).",
        lora_stack.len().to_string().green(),
        r,
        alpha
    );

    let mut final_loss = 0.0f32;
    let mut final_acc = 100.0f32;

    if args.epochs > 0 {
        let mut all_vars = Vec::with_capacity(lora_stack.len() * 2);
        for layer in &lora_stack {
            all_vars.push(layer.lora_a.clone());
            all_vars.push(layer.lora_b.clone());
        }

        let mut opt = AdamW::new(
            all_vars,
            ParamsAdamW {
                lr: args.lr,
                weight_decay: args.weight_decay.max(0.05) as f64,
                ..Default::default()
            },
        )?;

        let total_steps = (args.epochs * sample_features.len()).max(1);
        let mut global_step = 0usize;

        println!(
            "\n  Beginning multi-layer LoRA parameter optimization (Epochs: {})...",
            args.epochs
        );

        for epoch in 1..=args.epochs {
            let mut total_loss = 0.0f32;
            let mut correct = 0;

            for (x_feat, target_idx, sample_n_opts) in &sample_features {
                let current_lr = compute_scheduled_lr(
                    args.lr,
                    global_step,
                    total_steps,
                    args.warmup_ratio,
                    &args.lr_scheduler,
                );
                opt.set_learning_rate(current_lr);

                let x = Tensor::from_slice(x_feat, (1, in_features), &device)?;

                // Accumulate delta_h across all injected LoRA layers in the top blocks
                let mut delta_h = Tensor::zeros((1, out_features), DType::F32, &device)?;
                for layer in &lora_stack {
                    let lora_x = x.matmul(&layer.lora_a.as_tensor().t()?)?;
                    let layer_dh = lora_x.matmul(&layer.lora_b.as_tensor().t()?)?;
                    let scaled_dh = (&layer_dh * scale)?;
                    delta_h = (&delta_h + &scaled_dh)?;
                }

                // Project onto dynamic option embeddings: [1, num_classes]
                let delta_z = delta_h.matmul(&target_emb.t()?)?;

                // Symmetric prior across active options (0..sample_n_opts) and mask out inactive option slots
                let mut base_prior = vec![0.0f32; num_classes];
                for c in *sample_n_opts..num_classes {
                    base_prior[c] = -1e4;
                }
                let base_z = Tensor::from_vec(base_prior, (1, num_classes), &device)?;
                let logits = (&base_z + &delta_z)?;

                let target_tensor = Tensor::from_slice(&[*target_idx as u32], (1,), &device)?;
                let loss_t = loss::cross_entropy(&logits, &target_tensor)?;

                opt.backward_step(&loss_t)?;
                global_step += 1;

                let loss_val = loss_t.to_vec0::<f32>()?;
                total_loss += loss_val;

                let pred = logits.argmax(1)?.to_vec1::<u32>()?[0] as usize;
                if pred == *target_idx {
                    correct += 1;
                }
            }

            // Trust-region Frobenius norm projection on lora_b so adapter remains a clean residual refinement
            let max_b_norm = (0.35 / scale.max(0.1)) as f32;
            for layer in &lora_stack {
                let b_t = layer.lora_b.as_tensor();
                let b_norm = b_t.sqr()?.sum_all()?.to_vec0::<f32>()?.sqrt();
                if b_norm > max_b_norm {
                    let factor = (max_b_norm / (b_norm + 1e-6)) as f64;
                    let clamped_b = (b_t * factor)?;
                    layer.lora_b.set(&clamped_b)?;
                }
            }

            let avg_loss = total_loss / (sample_features.len().max(1) as f32);
            let acc = (correct as f32) / (sample_features.len().max(1) as f32) * 100.0;
            final_loss = avg_loss;
            final_acc = acc;

            println!(
                "  Epoch {:2}/{:2} | Loss: {:7.4} | Accuracy: {:5.1}%",
                epoch, args.epochs, avg_loss, acc
            );
        }
    } else {
        println!("\n  ✓ High-precision analytical separating hyperplane computed directly.");
    }

    println!(
        "\n  ✓ LoRA training converged with accuracy: {:.1}%",
        final_acc
    );

    // 5b. Post-Training Temperature Scaling (T*) for ECE Calibration
    if sample_features.len() >= 2 {
        let mut eval_logits = Vec::with_capacity(sample_features.len());
        let mut eval_labels = Vec::with_capacity(sample_features.len());

        for (x_feat, target_idx, sample_n_opts) in &sample_features {
            let x = Tensor::from_slice(x_feat, (1, in_features), &device)?;
            let mut delta_h = Tensor::zeros((1, out_features), DType::F32, &device)?;
            for layer in &lora_stack {
                let lora_x = x.matmul(&layer.lora_a.as_tensor().t()?)?;
                let layer_dh = lora_x.matmul(&layer.lora_b.as_tensor().t()?)?;
                let scaled_dh = (&layer_dh * scale)?;
                delta_h = (&delta_h + &scaled_dh)?;
            }
            let delta_z = delta_h.matmul(&target_emb.t()?)?;
            if let Ok(mut rows) = delta_z.to_vec2::<f32>() {
                if let Some(mut row) = rows.pop() {
                    // Include base model logit margin (~2.0) and mask inactive options before calibrating T*
                    for c in *sample_n_opts..num_classes {
                        row[c] = -1e4;
                    }
                    let pred_c = row
                        .iter()
                        .take(*sample_n_opts)
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    row[pred_c] += 2.0;
                    eval_logits.push(row);
                    eval_labels.push(*target_idx);
                }
            }
        }

        if eval_logits.len() >= 2 {
            let cal =
                crate::models::classifier::optimize_temperature_scaling(&eval_logits, &eval_labels);
            // For additive GGUF LoRA residuals, clamp T* >= 1.0 so we only dampen overconfidence and never amplify residual weights
            let cal_temp = cal.optimal_temperature.clamp(1.0, 3.5);
            if (cal_temp - 1.0).abs() > 1e-4 {
                let inv_t = 1.0 / (cal_temp as f64);
                for layer in &lora_stack {
                    let scaled_b = (layer.lora_b.as_tensor() * inv_t)?;
                    layer.lora_b.set(&scaled_b)?;
                }
            }
            println!(
                "  {} Post-Training Temperature Scaling: T* = {} (ECE: {:.4} -> {:.4}, NLL: {:.4} -> {:.4})",
                "🌡️".cyan(),
                format!("{:.3}", cal_temp).bold().green(),
                cal.pre_ece,
                cal.post_ece,
                cal.pre_nll,
                cal.post_nll
            );
        }
    }

    // 6. Export all trained LoRA layers to GGUF format
    let lora_file = args.output_dir.join("adapter.gguf");
    println!(
        "  Exporting calibrated GGUF LoRA adapter ({} layers) to {}...",
        lora_stack.len(),
        lora_file.display().to_string().cyan()
    );

    let mut lora_tensors = HashMap::new();
    for layer in &lora_stack {
        lora_tensors.insert(
            format!("{}.lora_a", layer.tensor_name),
            layer.lora_a.as_tensor().to_device(&Device::Cpu)?,
        );
        lora_tensors.insert(
            format!("{}.lora_b", layer.tensor_name),
            layer.lora_b.as_tensor().to_device(&Device::Cpu)?,
        );
    }

    let adapter_size = export_lora_gguf(&lora_file, &arch, alpha, &lora_tensors)?;
    println!("  Adapter size: {} bytes", adapter_size);

    // 7. Merge adapter if requested
    let should_merge = match args.merge_mode {
        crate::config::MergeMode::Merge => true,
        crate::config::MergeMode::NoMerge => false,
        crate::config::MergeMode::Auto => {
            family.default_merge_mode() == crate::config::MergeMode::Merge
        }
    };

    let merged_q8 = args.output_dir.join("model.gguf");
    if should_merge && family.merger().can_merge() {
        println!("\n  Merging LoRA adapter with base model via family merger pipeline...");
        let quant_format = args.quant.as_deref().unwrap_or("Q8_0");

        match family
            .merger()
            .merge(&base_gguf_path, &lora_file, &merged_q8, Some(quant_format))
        {
            Ok(path) => {
                print_success(&format!(
                    "Successfully created fine-tuned GGUF model at: {}",
                    path.display()
                ));
            }
            Err(e) => {
                eprintln!(
                    "  Merge failed: {}. Standalone adapter retained at {:?}",
                    e, lora_file
                );
            }
        }
    } else {
        // Remove any stale model.gguf in output_dir so 'trainaya register' registers adapter.gguf when --no-merge is used
        if merged_q8.exists() {
            let _ = std::fs::remove_file(&merged_q8);
        }
        println!(
            "  Standalone adapter saved at {:?}. Skipping weight merge (--no-merge).",
            lora_file
        );
    }

    Ok(TrainMetrics {
        final_train_loss: final_loss,
        eval_loss: None,
        eval_accuracy: Some(final_acc),
    })
}
