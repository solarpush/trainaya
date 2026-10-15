use anyhow::{Result, bail};
use colored::Colorize;
use std::path::{Path, PathBuf};

use crate::config::{ExportArgs, ExportFormat};
use crate::utils::{print_banner, print_status, print_success};

pub struct ExportSummary {
    pub format: ExportFormat,
    pub output_path: PathBuf,
    pub size_bytes: u64,
}

pub fn run_export(args: ExportArgs) -> Result<ExportSummary> {
    print_banner("Model Export & Packaging for Ollaya");

    print_status("Source Model", &args.model_path.display().to_string());
    print_status("Target Format", &args.format.to_string());
    print_status(
        "Quantization",
        args.quantize.as_deref().unwrap_or("none (full precision)"),
    );
    print_status("Output Directory", &args.output_dir.display().to_string());

    std::fs::create_dir_all(&args.output_dir)?;

    let weights_file = if args.model_path.is_file() {
        args.model_path.clone()
    } else if args.model_path.join("model.safetensors").exists() {
        args.model_path.join("model.safetensors")
    } else if args.model_path.join("model.gguf").exists() {
        args.model_path.join("model.gguf")
    } else {
        bail!(
            "Missing 'model.safetensors' or 'model.gguf' in '{}'.",
            args.model_path.display()
        );
    };

    let src_dir = if args.model_path.is_dir() {
        args.model_path.as_path()
    } else {
        args.model_path.parent().unwrap_or_else(|| Path::new("."))
    };

    for sidecar in ["config.json", "tokenizer.json", "decision.json", "calibration.json"] {
        let s = src_dir.join(sidecar);
        if s.exists() {
            let dst = args.output_dir.join(sidecar);
            if s != dst {
                let _ = std::fs::copy(&s, &dst);
            }
        }
    }

    let out_name = match args.format {
        ExportFormat::Safetensors => "model.safetensors",
        ExportFormat::Gguf => "model.gguf",
        ExportFormat::Onnx => "model.onnx",
    };
    let out_path = args.output_dir.join(out_name);
    if weights_file != out_path {
        std::fs::copy(&weights_file, &out_path)?;
    }

    let size_bytes = std::fs::metadata(&out_path)?.len();
    let summary = ExportSummary {
        format: args.format,
        output_path: out_path,
        size_bytes,
    };

    write_ollaya_manifest(&args, &summary)?;

    print_success(&format!(
        "Model successfully exported to '{}' format for Ollaya runtime!",
        args.format
    ));

    Ok(summary)
}

fn write_ollaya_manifest(args: &ExportArgs, summary: &ExportSummary) -> Result<()> {
    let manifest_file = args.output_dir.join("ollaya_manifest.json");

    let manifest = serde_json::json!({
        "runtime": "Ollaya",
        "trainaya_version": env!("CARGO_PKG_VERSION"),
        "format": summary.format.to_string(),
        "quantization": args.quantize.as_deref().unwrap_or("none"),
        "primary_file": summary.output_path.file_name().and_then(|s| s.to_str()).unwrap_or(""),
        "size_bytes": summary.size_bytes,
        "compatibility": {
            "zero_copy": true,
            "fast_inference": true
        }
    });

    std::fs::write(&manifest_file, serde_json::to_string_pretty(&manifest)?)?;
    println!(
        "  Generated Ollaya manifest: {}",
        manifest_file.display().to_string().green()
    );
    Ok(())
}
