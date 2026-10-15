use anyhow::Result;
use serde_json::json;
use std::path::PathBuf;

use crate::config::MergeArgs;
use crate::py_bridge::run_py_job;
use crate::utils::{print_banner, print_status, print_success, resolve_device_name};

pub struct MergeSummary {
    pub total_base_tensors: usize,
    pub merged_layers_count: usize,
    pub output_file: PathBuf,
}

pub fn run_merge(args: MergeArgs) -> Result<MergeSummary> {
    print_banner("Permanent Weights Merging (Base + LoRA -> SafeTensors / GGUF)");

    let device = resolve_device_name(&args.device)?;
    print_status("Base Model", &args.base_model);
    print_status("LoRA Adapter Dir", &args.lora_path.display().to_string());
    print_status("Output Dir", &args.output_dir.display().to_string());
    print_status("Compute Device", &device);

    let job = json!({
        "base_model": args.base_model,
        "lora_path": args.lora_path,
        "output_dir": args.output_dir,
        "device": device,
    });

    let res = run_py_job("merge", &job)?;
    let total = res
        .get("total_base_tensors")
        .and_then(|v| v.as_u64())
        .unwrap_or(1) as usize;
    let merged = res
        .get("merged_layers_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(1) as usize;
    let out_file = res
        .get("output_file")
        .and_then(|v| v.as_str())
        .map(PathBuf::from)
        .unwrap_or_else(|| args.output_dir.join("model.safetensors"));

    print_success(&format!(
        "Merge completed via PyTorch engine: {} layers merged into {}",
        merged,
        out_file.display()
    ));

    Ok(MergeSummary {
        total_base_tensors: total,
        merged_layers_count: merged,
        output_file: out_file,
    })
}
