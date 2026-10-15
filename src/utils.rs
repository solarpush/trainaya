use anyhow::Result;
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::{Path, PathBuf};

/// Normalize compute device string and apply hardware environment variables
pub fn resolve_device_name(device_str: &str) -> Result<String> {
    let hw = crate::hardware::SystemHardware::detect();
    hw.apply_environment(device_str);

    let lower = device_str.trim().to_lowercase();
    match lower.as_str() {
        "" | "auto" => Ok("auto".to_string()),
        "cpu" | "cuda" | "metal" | "mps" | "rocm" | "hip" | "vulkan" => Ok(lower),
        other if other.starts_with("cuda:") || other.starts_with("metal:") => Ok(other.to_string()),
        other => anyhow::bail!(
            "Unsupported device: '{}'. Use 'cpu', 'cuda', 'metal', 'rocm', 'vulkan', or 'auto'.",
            other
        ),
    }
}

/// Create an ergonomic progress bar
pub fn create_progress_bar(len: u64, message: &'static str) -> ProgressBar {
    let pb = ProgressBar::new(len);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
            .unwrap()
            .progress_chars("#>-"),
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

/// Locate a llama.cpp binary (e.g. llama-export-lora, llama-quantize)
pub fn find_llama_tool(bin_name: &str) -> Option<PathBuf> {
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
