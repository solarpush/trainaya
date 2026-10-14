use anyhow::Result;
use candle_core::{DType, Device};
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::Path;

/// Choose the best compute device available or specified by user
pub fn select_device(device_str: &str) -> Result<Device> {
    let hw = crate::hardware::SystemHardware::detect();
    hw.apply_environment(device_str);

    match device_str.to_lowercase().as_str() {
        "cpu" => Ok(Device::Cpu),
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                use anyhow::Context;
                Device::new_cuda(0).context("Failed to initialize CUDA device 0")
            }
            #[cfg(not(feature = "cuda"))]
            {
                tracing::warn!("CUDA feature is not enabled during build, falling back to CPU.");
                println!(
                    "{} CUDA feature not enabled in build. Using CPU SIMD instead.",
                    "⚠️ [WARNING]".yellow().bold()
                );
                Ok(Device::Cpu)
            }
        }
        "metal" | "mps" => {
            #[cfg(feature = "metal")]
            {
                use anyhow::Context;
                Device::new_metal(0).context("Failed to initialize Apple Metal device 0")
            }
            #[cfg(not(feature = "metal"))]
            {
                tracing::warn!("Metal feature is not enabled during build, falling back to CPU.");
                println!(
                    "{} Metal feature not enabled in build (compile with --features metal). Using CPU SIMD instead.",
                    "⚠️ [WARNING]".yellow().bold()
                );
                Ok(Device::Cpu)
            }
        }
        "auto" | "" => {
            #[cfg(feature = "cuda")]
            {
                if let Ok(dev) = Device::new_cuda(0) {
                    return Ok(dev);
                }
            }
            #[cfg(feature = "metal")]
            {
                if let Ok(dev) = Device::new_metal(0) {
                    return Ok(dev);
                }
            }
            Ok(Device::Cpu)
        }
        "rocm" | "hip" | "vulkan" => {
            #[cfg(feature = "cuda")]
            {
                if let Ok(dev) = Device::new_cuda(0) {
                    return Ok(dev);
                }
            }
            if let crate::hardware::GpuKind::Rocm { name, arch } = &hw.gpu {
                println!(
                    "  {} Active AMD GPU: {} (HSA_OVERRIDE_GFX_VERSION={})",
                    "⚡ [ROCm/Vulkan]".cyan().bold(),
                    name.bright_white().bold(),
                    arch.green()
                );
            }
            Ok(Device::Cpu)
        }
        other => {
            if let Some(stripped) = other.strip_prefix("cuda:") {
                #[cfg(feature = "cuda")]
                {
                    let id: usize = stripped.parse().map_err(|e| anyhow::anyhow!("Invalid CUDA device ID: {}", e))?;
                    return Device::new_cuda(id).map_err(|e| anyhow::anyhow!("Failed to initialize CUDA device: {}", e));
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = stripped;
                    println!(
                        "{} CUDA/GPU not compiled in this binary, fallback to CPU.",
                        "⚠️ [WARNING]".yellow().bold()
                    );
                    return Ok(Device::Cpu);
                }
            }
            if let Some(stripped) = other.strip_prefix("metal:") {
                #[cfg(feature = "metal")]
                {
                    let id: usize = stripped.parse().map_err(|e| anyhow::anyhow!("Invalid Metal device ID: {}", e))?;
                    return Device::new_metal(id).map_err(|e| anyhow::anyhow!("Failed to initialize Metal device: {}", e));
                }
                #[cfg(not(feature = "metal"))]
                {
                    let _ = stripped;
                    return Ok(Device::Cpu);
                }
            }
            anyhow::bail!("Unsupported device: '{}'. Use 'cpu', 'cuda', 'metal', 'rocm', 'vulkan', or 'auto'.", device_str);
        }
    }
}

/// Select optimal tensor precision (BF16 on supported GPUs or when explicitly requested, F32 otherwise)
pub fn select_dtype(device: &Device, bf16_requested: bool) -> DType {
    if bf16_requested {
        match device {
            Device::Cuda(_) | Device::Metal(_) => DType::BF16,
            Device::Cpu => DType::F32, // Keep F32 on CPU as Candle CPU matmul does not natively support BF16 backward
        }
    } else {
        DType::F32
    }
}

/// Create an ergonomic progress bar
pub fn create_progress_bar(len: u64, message: &'static str) -> ProgressBar {
    let pb = ProgressBar::new(len);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
            .unwrap()
            .progress_chars("#>-")
    );
    pb.set_message(message);
    pb
}

/// Print section header banner
pub fn print_banner(title: &str) {
    let line = "=".repeat(60);
    println!("\n{}", line.cyan());
    println!("  {} {}", "🚀 Trainaya:".bold().white(), title.bold().cyan());
    println!("{}\n", line.cyan());
}

/// Format byte size into human readable string
pub fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GiB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MiB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KiB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Print formatted key-value status
pub fn print_status(key: &str, value: impl AsRef<str>) {
    println!("  {:<18} : {}", key.dimmed(), value.as_ref().bold().bright_white());
}

/// Print success message
pub fn print_success(msg: &str) {
    println!("\n{} {}\n", "✓".green().bold(), msg.green());
}

/// Check file existence with friendly error
pub fn ensure_file_exists(path: &Path, desc: &str) -> Result<()> {
    if !path.exists() {
        anyhow::bail!("{} not found at '{}'", desc, path.display());
    }
    Ok(())
}

use std::path::PathBuf;

/// Locate a llama.cpp binary (e.g. llama-export-lora, llama-quantize)
pub fn find_llama_tool(bin_name: &str) -> Option<PathBuf> {
    // 1. Explicit env var override (e.g. LLAMA_EXPORT_LORA, LLAMA_QUANTIZE)
    let env_var_name = bin_name.to_uppercase().replace('-', "_");
    if let Ok(path_str) = std::env::var(&env_var_name) {
        let p = PathBuf::from(path_str);
        if p.exists() {
            return Some(p);
        }
    }
    if let Ok(dir_str) = std::env::var("LLAMA_CPP_DIR") {
        let p = PathBuf::from(&dir_str).join("build/bin").join(bin_name);
        if p.exists() {
            return Some(p);
        }
        let p2 = PathBuf::from(dir_str).join("bin").join(bin_name);
        if p2.exists() {
            return Some(p2);
        }
    }

    // 2. Relative to current project or executable
    let candidates = [
        PathBuf::from("vendor/llama.cpp/build/bin").join(bin_name),
        PathBuf::from("../vendor/llama.cpp/build/bin").join(bin_name),
        PathBuf::from("/usr/local/bin").join(bin_name),
        PathBuf::from("/usr/bin").join(bin_name),
    ];

    for cand in &candidates {
        if cand.exists() {
            return Some(cand.clone());
        }
    }

    // 3. Search in system PATH via `which`
    if let Ok(output) = std::process::Command::new("which").arg(bin_name).output() {
        if output.status.success() {
            let p_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let p = PathBuf::from(p_str);
            if p.exists() {
                return Some(p);
            }
        }
    }

    None
}
