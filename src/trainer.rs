use anyhow::Result;
use candle_core::Tensor;
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use colored::Colorize;
use tokenizers::Tokenizer;

use crate::config::{TrainayaMetadata, TrainingArgs};
use crate::dataset::Dataset;
use crate::models::{classifier::compute_cross_entropy_loss, SequenceClassifier};
use crate::utils::{
    create_progress_bar, print_banner, print_status, print_success, select_device, select_dtype,
};

pub struct TrainMetrics {
    pub final_train_loss: f32,
    pub eval_loss: Option<f32>,
    pub eval_accuracy: Option<f32>,
}

/// Compute learning rate at a given optimization step using Warmup + Cosine / Linear / Constant schedule
pub fn compute_scheduled_lr(
    base_lr: f64,
    current_step: usize,
    total_steps: usize,
    warmup_ratio: f32,
    scheduler: &str,
) -> f64 {
    if total_steps <= 1 {
        return base_lr;
    }

    let warmup_steps = ((total_steps as f32) * warmup_ratio.clamp(0.0, 0.9)).round() as usize;

    // 1. Linear warmup phase (from 10% of base_lr up to 100% base_lr)
    if warmup_steps > 0 && current_step < warmup_steps {
        let progress = (current_step + 1) as f64 / (warmup_steps as f64);
        return base_lr * (0.1 + 0.9 * progress);
    }

    // 2. Post-warmup decay phase
    let decay_steps = total_steps.saturating_sub(warmup_steps).max(1);
    let post_warmup_step = current_step.saturating_sub(warmup_steps).min(decay_steps);
    let progress = (post_warmup_step as f64) / (decay_steps as f64);
    let min_lr = base_lr * 0.05;

    match scheduler.to_lowercase().as_str() {
        "constant" | "none" => base_lr,
        "linear" => base_lr - (base_lr - min_lr) * progress,
        _ => {
            // Default: Cosine annealing decay
            let cosine = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
            min_lr + (base_lr - min_lr) * cosine
        }
    }
}

pub fn run_training(args: TrainingArgs) -> Result<TrainMetrics> {
    print_banner("Fine-Tuning Loop (LoRA / Candle-Rust)");

    let device = select_device(&args.device)?;
    let dtype = select_dtype(&device, args.bf16);

    print_status("Base Model", &args.base_model);
    print_status("Dataset Path", &args.data_path.display().to_string());
    print_status("Task", &args.task.to_string());
    print_status("Output Dir", &args.output_dir.display().to_string());
    print_status("Epochs", &args.epochs.to_string());
    print_status(
        "Batch Size",
        &format!(
            "{} (grad_accum: {}, effective: {})",
            args.batch_size,
            args.grad_accum_steps.max(1),
            args.batch_size * args.grad_accum_steps.max(1)
        ),
    );
    print_status(
        "Learning Rate",
        &format!(
            "{:e} (schedule: {}, warmup: {:.0}%)",
            args.lr,
            args.lr_scheduler,
            args.warmup_ratio * 100.0
        ),
    );
    print_status("LoRA Rank (r)", &args.lora.r.to_string());
    print_status("LoRA Alpha", &args.lora.alpha.to_string());
    print_status("Compute Device", &format!("{:?} ({:?})", device, dtype));

    let registry = crate::families::FamilyRegistry::default();
    let family = registry.detect_or_resolve(&args.base_model, args.family.as_deref())?;

    print_status("Family Layout", family.identifier());
    print_status("Backend Type", &format!("{:?}", family.backend_type()));
    print_status("Merge Mode", &format!("{}", args.merge_mode));

    if family.backend_type() == crate::families::traits::BackendType::Gguf {
        return crate::models::gguf_lora::run_gguf_training(args, family);
    }

    // 1. Load Tokenizer
    let (_, tokenizer_path, _) = SequenceClassifier::load_or_fetch_weights(&args.base_model)?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer from {:?}: {}", tokenizer_path, e))?;

    // 2. Load Raw Samples (automatically using Family PromptBuilder for Kev / Laya / Decision datasets)
    let raw_samples = Dataset::load_raw_samples_with_family(&args.data_path, family)?;
    println!(
        "\n  Loaded {} raw samples from data file.",
        raw_samples.len().to_string().cyan()
    );

    let (train_raw, eval_raw) = if let Some(eval_path) = &args.eval_path {
        let eval_samples = Dataset::load_raw_samples_with_family(eval_path, family)?;
        (raw_samples, eval_samples)
    } else {
        Dataset::split_samples(raw_samples, args.eval_split, args.seed)
    };

    println!(
        "  Split: {} train samples, {} eval samples.",
        train_raw.len().to_string().green(),
        eval_raw.len().to_string().yellow()
    );

    // 3. Build Datasets (unpadded; dynamic per-batch padding is applied during batching)
    let train_dataset = Dataset::build(&train_raw, &tokenizer, args.max_length, None)?;
    let label2id = train_dataset.label2id.clone();
    let id2label = train_dataset.id2label.clone();
    let num_labels = train_dataset.num_labels;

    let eval_dataset = if !eval_raw.is_empty() {
        Some(Dataset::build(
            &eval_raw,
            &tokenizer,
            args.max_length,
            Some((label2id.clone(), id2label.clone())),
        )?)
    } else {
        None
    };

    println!(
        "  Detected {} classes: {:?}",
        num_labels.to_string().cyan(),
        label2id.keys().collect::<Vec<_>>()
    );

    // 4. Initialize Model with LoRA connected to forward graph
    println!("\n  Initializing model architecture with live LoRA graph injection...");
    let model = SequenceClassifier::build_with_dtype(
        &args.base_model,
        num_labels,
        &args.lora,
        dtype,
        &device,
    )?;
    let vars = model.trainable_vars();
    println!(
        "  LoRA injected into {} layers. Trainable variables: {}.",
        model.lora_layers.len().to_string().green(),
        vars.len().to_string().cyan()
    );

    // 5. Setup Optimizer & LR Schedule
    let grad_accum_steps = args.grad_accum_steps.max(1);
    let batches_per_epoch = (train_dataset.samples.len() + args.batch_size.max(1) - 1) / args.batch_size.max(1);
    let steps_per_epoch = (batches_per_epoch + grad_accum_steps - 1) / grad_accum_steps;
    let total_opt_steps = (steps_per_epoch * args.epochs).max(1);

    let params = ParamsAdamW {
        lr: args.lr,
        weight_decay: args.weight_decay as f64,
        ..Default::default()
    };
    let mut opt = AdamW::new(vars.clone(), params)?;

    std::fs::create_dir_all(&args.output_dir)?;
    let adapter_file = args.output_dir.join("adapter_model.safetensors");
    let best_checkpoint_file = args.output_dir.join("adapter_best.safetensors");

    // 6. Training Loop
    println!("\n  Starting training loop (dynamic batch padding + {} LR schedule)...\n", args.lr_scheduler);
    let mut last_train_loss = 0.0f32;
    let mut best_eval_loss: Option<f32> = None;
    let mut best_eval_accuracy: Option<f32> = None;
    let mut global_opt_step = 0usize;

    for epoch in 1..=args.epochs {
        let batches = train_dataset.create_batches(args.batch_size, true, &device)?;
        let num_batches = batches.len();
        let pb = create_progress_bar(num_batches as u64, "Training epoch");
        pb.set_message(format!("Epoch {}/{}", epoch, args.epochs));

        let mut epoch_loss_sum = 0.0f32;
        let mut accum_grads: Vec<Option<Tensor>> = vec![None; vars.len()];
        let mut accum_count = 0usize;

        for (batch_idx, batch) in batches.iter().enumerate() {
            // Forward pass with train=true (enables LoRA dropout & live graph injection)
            let logits = model.forward_t(&batch.input_ids, &batch.attention_mask, true)?;
            let loss = compute_cross_entropy_loss(&logits, &batch.labels)?;

            let loss_val = loss.to_scalar::<f32>()?;
            epoch_loss_sum += loss_val;

            // Scale loss for gradient accumulation and run backward pass
            let is_last_batch = batch_idx + 1 == num_batches;
            let current_window = if is_last_batch {
                (batch_idx % grad_accum_steps) + 1
            } else {
                grad_accum_steps
            };
            let scaled_loss = (&loss / (current_window as f64))?;
            let mut step_grads = scaled_loss.backward()?;

            // Accumulate gradients across micro-batches
            for (v_idx, var) in vars.iter().enumerate() {
                if let Some(g) = step_grads.get(var) {
                    let g_f32 = g.to_dtype(candle_core::DType::F32)?;
                    accum_grads[v_idx] = match &accum_grads[v_idx] {
                        Some(prev) => Some((prev + &g_f32)?),
                        None => Some(g_f32),
                    };
                }
            }
            accum_count += 1;

            if accum_count >= grad_accum_steps || is_last_batch {
                // Apply global L2 gradient norm clipping
                if args.max_grad_norm > 0.0 {
                    let mut total_norm_sq = 0.0f32;
                    for g_opt in accum_grads.iter().flatten() {
                        let n_sq = g_opt.sqr()?.sum_all()?.to_scalar::<f32>()?;
                        total_norm_sq += n_sq;
                    }
                    let total_norm = total_norm_sq.sqrt();
                    if total_norm > args.max_grad_norm {
                        let scale_factor = (args.max_grad_norm / (total_norm + 1e-6)) as f64;
                        for g_opt in accum_grads.iter_mut() {
                            if let Some(g) = g_opt {
                                *g = (g.clone() * scale_factor)?;
                            }
                        }
                    }
                }

                // Update learning rate according to schedule
                let current_lr = compute_scheduled_lr(
                    args.lr,
                    global_opt_step,
                    total_opt_steps,
                    args.warmup_ratio,
                    &args.lr_scheduler,
                );
                opt.set_learning_rate(current_lr);

                // Write accumulated & clipped gradients back into GradStore and step
                for (v_idx, var) in vars.iter().enumerate() {
                    if let Some(g) = accum_grads[v_idx].take() {
                        step_grads.insert(var, g.to_dtype(var.dtype())?);
                    }
                }
                opt.step(&step_grads)?;
                global_opt_step += 1;
                accum_count = 0;
            }

            pb.inc(1);
            pb.set_message(format!(
                "Epoch {}/{} - loss: {:.4}",
                epoch, args.epochs, loss_val
            ));
        }

        pb.finish_with_message(format!("Epoch {}/{} completed", epoch, args.epochs));
        last_train_loss = epoch_loss_sum / (num_batches.max(1) as f32);

        println!(
            "  Epoch {}/{}: average train loss = {}",
            epoch,
            args.epochs,
            format!("{:.4}", last_train_loss).bold().green()
        );

        // Run evaluation if eval dataset exists
        if let Some(eval_ds) = &eval_dataset {
            let eval_batches = eval_ds.create_batches(args.batch_size, false, &device)?;
            let mut eval_loss_sum = 0.0f32;
            let mut correct_predictions = 0usize;
            let mut total_predictions = 0usize;

            for e_batch in &eval_batches {
                let logits = model.forward_t(&e_batch.input_ids, &e_batch.attention_mask, false)?;
                let loss = compute_cross_entropy_loss(&logits, &e_batch.labels)?;
                eval_loss_sum += loss.to_scalar::<f32>()?;

                let preds = logits.argmax(1)?;
                let preds_vec = preds.to_vec1::<u32>()?;
                let labels_vec = e_batch.labels.to_vec1::<u32>()?;

                for (p, y) in preds_vec.iter().zip(labels_vec.iter()) {
                    if p == y {
                        correct_predictions += 1;
                    }
                    total_predictions += 1;
                }
            }

            let avg_eval_loss = eval_loss_sum / (eval_batches.len().max(1) as f32);
            let accuracy = (correct_predictions as f32) / (total_predictions.max(1) as f32);

            let is_best = best_eval_loss.map_or(true, |best| avg_eval_loss <= best);
            if is_best {
                best_eval_loss = Some(avg_eval_loss);
                best_eval_accuracy = Some(accuracy * 100.0);
                model.save_adapter(&best_checkpoint_file)?;
            }

            println!(
                "    ↳ Eval Loss: {}, Accuracy: {}{}",
                format!("{:.4}", avg_eval_loss).yellow(),
                format!("{:.2}%", accuracy * 100.0).bold().cyan(),
                if is_best { " ★ (best)".green().to_string() } else { String::new() }
            );
        }
    }

    // 7. Restore best validation checkpoint if available and save final adapter & metadata
    if best_checkpoint_file.exists() {
        let _ = model.load_adapter(&best_checkpoint_file);
        let _ = std::fs::remove_file(&best_checkpoint_file);
    }

    model.save_adapter(&adapter_file)?;
    println!(
        "\n  Saved adapter weights to: {}",
        adapter_file.display().to_string().cyan()
    );

    // Save adapter_config.json
    let adapter_config = serde_json::json!({
        "base_model_name_or_path": args.base_model,
        "r": args.lora.r,
        "lora_alpha": args.lora.alpha,
        "lora_dropout": args.lora.dropout,
        "target_modules": args.lora.target_modules,
        "peft_type": "LORA",
        "task_type": "SEQ_CLS"
    });
    let adapter_cfg_file = args.output_dir.join("adapter_config.json");
    std::fs::write(&adapter_cfg_file, serde_json::to_string_pretty(&adapter_config)?)?;

    // Copy tokenizer
    tokenizer
        .save(args.output_dir.join("tokenizer.json"), true)
        .map_err(|e| anyhow::anyhow!("Failed to save tokenizer: {}", e))?;

    // Save trainaya_metadata.json
    let metadata = TrainayaMetadata {
        version: env!("CARGO_PKG_VERSION").to_string(),
        base_model: args.base_model.clone(),
        task: args.task,
        num_labels,
        id2label,
        label2id,
        lora_rank: args.lora.r,
        lora_alpha: args.lora.alpha,
        epochs: args.epochs,
        train_samples: train_raw.len(),
        eval_samples: eval_raw.len(),
        target_runtime: "Ollaya".to_string(),
    };
    let metadata_file = args.output_dir.join("trainaya_metadata.json");
    std::fs::write(&metadata_file, serde_json::to_string_pretty(&metadata)?)?;

    print_success("Fine-tuning completed successfully! Adapter is ready for merging or serving.");

    // Merge if requested
    let should_merge = match args.merge_mode {
        crate::config::MergeMode::Merge => true,
        crate::config::MergeMode::NoMerge => false,
        crate::config::MergeMode::Auto => family.default_merge_mode() == crate::config::MergeMode::Merge,
    };

    if should_merge && family.merger().can_merge() {
        println!("\n  Merging LoRA adapter with base model via family merger pipeline...");
        let merged_dir = args.output_dir.join("merged");
        std::fs::create_dir_all(&merged_dir)?;
        let (_, _, base_weights_path) = SequenceClassifier::load_or_fetch_weights(&args.base_model)?;
        let merged_path = family.merger().merge(
            &base_weights_path,
            &args.output_dir,
            &merged_dir,
            args.quant.as_deref(),
        )?;
        print_success(&format!("Merged model saved to: {}", merged_path.display()));
    } else {
        println!("  Adapter retained in unmerged format at: {:?}", args.output_dir);
    }

    Ok(TrainMetrics {
        final_train_loss: last_train_loss,
        eval_loss: best_eval_loss,
        eval_accuracy: best_eval_accuracy,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lr_scheduler_warmup_and_cosine() {
        let base_lr = 1e-4;
        // Step 0 of 100 with 10% warmup (10 steps) -> should be < base_lr
        let lr_start = compute_scheduled_lr(base_lr, 0, 100, 0.1, "cosine");
        assert!(lr_start < base_lr && lr_start > 0.0);

        // Step 9 (end of warmup) -> reaches base_lr
        let lr_peak = compute_scheduled_lr(base_lr, 9, 100, 0.1, "cosine");
        assert!((lr_peak - base_lr).abs() < 1e-9);

        // Step 99 (end of cosine decay) -> approaches 5% of base_lr
        let lr_end = compute_scheduled_lr(base_lr, 99, 100, 0.1, "cosine");
        assert!(lr_end < base_lr * 0.1);
    }
}
