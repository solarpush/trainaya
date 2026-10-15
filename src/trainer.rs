use anyhow::Result;
use colored::Colorize;
use serde_json::json;

use crate::config::{AdaptArgs, MergeMode, TrainingArgs};
use crate::families::FamilyRegistry;
use crate::py_bridge::run_py_job;
use crate::utils::{print_banner, print_status, print_success, resolve_device_name};

pub fn run_training(args: TrainingArgs) -> Result<()> {
    print_banner("PyTorch End-to-End Decision & Classification Training (uv + PyTorch)");

    let device = resolve_device_name(&args.device)?;

    let registry = FamilyRegistry::default();
    let family = registry.detect_or_resolve(&args.base_model, args.family.as_deref())?;
    let effective_merge_mode = match args.merge_mode {
        MergeMode::Auto => family.default_merge_mode(),
        other => other,
    };

    print_status("Base Model", &args.base_model);
    print_status("Model Family", family.identifier());
    print_status("Decision Layout", family.default_layout());
    print_status("Merge Strategy", &effective_merge_mode.to_string());
    print_status("Dataset Path", &args.data_path.display().to_string());
    print_status("Compute Device", &device);
    print_status("Epochs", &args.epochs.to_string());
    print_status("Batch Size", &args.batch_size.to_string());
    print_status("Learning Rate", &format!("{:.2e}", args.lr));
    if args.unfreeze_layers > 0 {
        print_status("Unfrozen Backbone", &format!("Top {} layers", args.unfreeze_layers));
    }
    println!();

    std::fs::create_dir_all(&args.output_dir)?;

    let job = json!({
        "base_model": args.base_model,
        "family": family.identifier(),
        "data_path": args.data_path,
        "eval_path": args.eval_path,
        "eval_split": args.eval_split,
        "output_dir": args.output_dir,
        "task": args.task.to_string(),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "lr": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": args.max_grad_norm,
        "lr_scheduler": args.lr_scheduler,
        "max_length": args.max_length,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "bf16": args.bf16,
        "lora_layers": args.lora_layers,
        "unfreeze_layers": args.unfreeze_layers,
        "device": device,
        "lora": {
            "r": args.lora.r,
            "alpha": args.lora.alpha,
            "dropout": args.lora.dropout,
            "target_modules": args.lora.target_modules,
        },
        "merge_mode": effective_merge_mode.to_string(),
        "quant": args.quant,
    });

    let res = run_py_job("train", &job)?;

    if let Some(temps) = res.get("temperature") {
        let ece_before = res.get("ece_before").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let ece_after = res.get("ece_after").and_then(|v| v.as_f64()).unwrap_or(0.0);
        println!(
            "\n  {} Post-Hoc Temperature Calibration: T = {} (ECE {:.4} -> {:.4})",
            "🌡️".cyan(),
            temps.to_string().green().bold(),
            ece_before,
            ece_after
        );
    }

    print_success(&format!(
        "PyTorch training & calibration completed! Artifacts saved to: {}",
        args.output_dir.display()
    ));

    Ok(())
}

pub fn run_adapt(args: AdaptArgs) -> Result<()> {
    print_banner("Generic Model Adaptation for Ollaya Decision Runtime");

    let device = resolve_device_name(&args.device)?;
    print_status("Source Model", &args.from);
    print_status("Target Tag", &args.target);
    print_status("Target Layout", &args.layout);
    if let Some(ref d) = args.data_path {
        print_status("Alignment Data", &d.display().to_string());
    }
    print_status("Output Directory", &args.output_dir.display().to_string());
    println!();

    std::fs::create_dir_all(&args.output_dir)?;

    let job = json!({
        "source": args.from,
        "target_tag": args.target,
        "layout": args.layout,
        "output_dir": args.output_dir,
        "data_path": args.data_path,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "device": device,
    });

    let res = run_py_job("adapt", &job)?;
    let resolved_layout = res
        .get("layout")
        .and_then(|v| v.as_str())
        .unwrap_or(&args.layout);

    print_success(&format!(
        "Successfully adapted '{}' into Ollaya '{}' bundle at {}",
        args.from,
        resolved_layout,
        args.output_dir.display()
    ));

    Ok(())
}
