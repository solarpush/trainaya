use anyhow::Result;
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use colored::Colorize;
use tokenizers::Tokenizer;

use crate::config::{TrainayaMetadata, TrainingArgs};
use crate::dataset::Dataset;
use crate::models::{classifier::compute_cross_entropy_loss, SequenceClassifier};
use crate::utils::{create_progress_bar, print_banner, print_status, print_success, select_device};

pub struct TrainMetrics {
    pub final_train_loss: f32,
    pub eval_loss: Option<f32>,
    pub eval_accuracy: Option<f32>,
}

pub fn run_training(args: TrainingArgs) -> Result<TrainMetrics> {
    print_banner("Fine-Tuning Loop (LoRA / Candle-Rust)");

    let device = select_device(&args.device)?;
    print_status("Base Model", &args.base_model);
    print_status("Dataset Path", &args.data_path.display().to_string());
    print_status("Task", &args.task.to_string());
    print_status("Output Dir", &args.output_dir.display().to_string());
    print_status("Epochs", &args.epochs.to_string());
    print_status("Batch Size", &args.batch_size.to_string());
    print_status("Learning Rate", &format!("{:e}", args.lr));
    print_status("LoRA Rank (r)", &args.lora.r.to_string());
    print_status("LoRA Alpha", &args.lora.alpha.to_string());
    print_status("Compute Device", &format!("{:?}", device));

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

    // 2. Load Raw Samples
    let raw_samples = Dataset::load_raw_samples(&args.data_path)?;
    println!(
        "\n  Loaded {} raw samples from data file.",
        raw_samples.len().to_string().cyan()
    );

    let (train_raw, eval_raw) = if let Some(eval_path) = &args.eval_path {
        let eval_samples = Dataset::load_raw_samples(eval_path)?;
        (raw_samples, eval_samples)
    } else {
        Dataset::split_samples(raw_samples, args.eval_split, args.seed)
    };

    println!(
        "  Split: {} train samples, {} eval samples.",
        train_raw.len().to_string().green(),
        eval_raw.len().to_string().yellow()
    );

    // 3. Build Datasets
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

    // 4. Initialize Model with LoRA
    println!("\n  Initializing model architecture with LoRA layers...");
    let model = SequenceClassifier::build(&args.base_model, num_labels, &args.lora, &device)?;
    println!(
        "  LoRA injected into {} layers. Trainable variables: {}.",
        model.lora_layers.len().to_string().green(),
        model.trainable_vars().len().to_string().cyan()
    );

    // 5. Setup Optimizer
    let vars = model.trainable_vars();
    let params = ParamsAdamW {
        lr: args.lr,
        weight_decay: args.weight_decay as f64,
        ..Default::default()
    };
    let mut opt = AdamW::new(vars, params)?;

    // 6. Training Loop
    println!("\n  Starting training loop...\n");
    let mut last_train_loss = 0.0f32;

    for epoch in 1..=args.epochs {
        let batches = train_dataset.create_batches(args.batch_size, true, &device)?;
        let num_batches = batches.len();
        let pb = create_progress_bar(
            num_batches as u64,
            "Training epoch",
        );
        pb.set_message(format!("Epoch {}/{}", epoch, args.epochs));

        let mut epoch_loss_sum = 0.0f32;

        for batch in batches {
            // Forward pass
            let logits = model.forward(&batch.input_ids, &batch.attention_mask)?;
            let loss = compute_cross_entropy_loss(&logits, &batch.labels)?;

            let loss_val = loss.to_scalar::<f32>()?;
            epoch_loss_sum += loss_val;

            // Backward pass & optimizer step
            let grads = loss.backward()?;
            opt.step(&grads)?;

            pb.inc(1);
            pb.set_message(format!("Epoch {}/{} - loss: {:.4}", epoch, args.epochs, loss_val));
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
                let logits = model.forward(&e_batch.input_ids, &e_batch.attention_mask)?;
                let loss = compute_cross_entropy_loss(&logits, &e_batch.labels)?;
                eval_loss_sum += loss.to_scalar::<f32>()?;

                // Compute accuracy: argmax(logits) == labels
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

            println!(
                "    ↳ Eval Loss: {}, Accuracy: {}",
                format!("{:.4}", avg_eval_loss).yellow(),
                format!("{:.2}%", accuracy * 100.0).bold().cyan()
            );
        }
    }

    // 7. Save Checkpoint & Metadata
    std::fs::create_dir_all(&args.output_dir)?;
    let adapter_file = args.output_dir.join("adapter_model.safetensors");
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
    tokenizer.save(args.output_dir.join("tokenizer.json"), true)
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
        eval_loss: None,
        eval_accuracy: None,
    })
}
