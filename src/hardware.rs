use colored::Colorize;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuKind {
    Rocm { name: String, arch: String },
    Cuda { name: String },
    Metal { name: String },
    None,
}

#[derive(Debug, Clone)]
pub struct SystemHardware {
    pub os: String,
    pub arch: String,
    pub cpu_simd: Vec<String>,
    pub gpu: GpuKind,
}

impl SystemHardware {
    pub fn detect() -> Self {
        let os = std::env::consts::OS.to_string();
        let arch = std::env::consts::ARCH.to_string();

        let mut cpu_simd = Vec::new();
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                cpu_simd.push("AVX2".to_string());
            }
            if is_x86_feature_detected!("fma") {
                cpu_simd.push("FMA".to_string());
            }
            if is_x86_feature_detected!("avx512f") {
                cpu_simd.push("AVX-512F".to_string());
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            cpu_simd.push("NEON".to_string());
        }

        let gpu = Self::detect_gpu();

        Self {
            os,
            arch,
            cpu_simd,
            gpu,
        }
    }

    fn detect_gpu() -> GpuKind {
        // 1. Check for Apple Silicon Metal
        if cfg!(target_os = "macos") {
            return GpuKind::Metal {
                name: "Apple Silicon (MPS Metal)".to_string(),
            };
        }

        // 2. Check for AMD ROCm / HIP
        if Path::new("/dev/kfd").exists() {
            let mut gpu_name = "AMD Radeon GPU".to_string();
            let mut arch_override = "12.0.0".to_string();

            // Try reading PCI devices or DRM
            if let Ok(output) = std::process::Command::new("lspci").output() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                for line in stdout.lines() {
                    if (line.contains("VGA") || line.contains("Display")) && line.contains("AMD") {
                        if line.contains("9070") || line.contains("Navi 48") {
                            gpu_name = "AMD Radeon RX 9070 / Navi 48 (RDNA 4)".to_string();
                            arch_override = "12.0.0".to_string();
                        } else if line.contains("7900") || line.contains("Navi 31") {
                            gpu_name = "AMD Radeon RX 7900 / Navi 31 (RDNA 3)".to_string();
                            arch_override = "11.0.0".to_string();
                        } else {
                            gpu_name = line.to_string();
                        }
                        break;
                    }
                }
            }

            return GpuKind::Rocm {
                name: gpu_name,
                arch: arch_override,
            };
        }

        // 3. Check for NVIDIA CUDA
        if Path::new("/dev/nvidia0").exists() || Self::has_command("nvidia-smi") {
            let mut name = "NVIDIA GeForce / Data Center GPU".to_string();
            if let Ok(output) = std::process::Command::new("nvidia-smi")
                .arg("--query-gpu=name")
                .arg("--format=csv,noheader")
                .output()
            {
                let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !stdout.is_empty() {
                    name = stdout;
                }
            }
            return GpuKind::Cuda { name };
        }

        GpuKind::None
    }

    fn has_command(cmd: &str) -> bool {
        std::process::Command::new("which")
            .arg(cmd)
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Return specific environment variables needed by the active GPU runtime
    pub fn get_environment_for_device(&self, device_str: &str) -> HashMap<String, String> {
        let mut envs = HashMap::new();
        let target = device_str.to_lowercase();

        match &self.gpu {
            GpuKind::Rocm { arch, .. } if target == "rocm" || target == "hip" || target == "vulkan" || target == "auto" => {
                envs.insert("HSA_OVERRIDE_GFX_VERSION".to_string(), arch.clone());
                envs.insert("HIP_VISIBLE_DEVICES".to_string(), "0".to_string());
                envs.insert("ROCM_PATH".to_string(), "/usr".to_string());
            }
            GpuKind::Cuda { .. } if target == "cuda" || target == "auto" => {
                envs.insert("CUDA_VISIBLE_DEVICES".to_string(), "0".to_string());
            }
            GpuKind::Metal { .. } if target == "metal" || target == "mps" || target == "auto" => {
                envs.insert("PYTORCH_ENABLE_MPS_FALLBACK".to_string(), "1".to_string());
            }
            _ => {}
        }

        envs
    }

    /// Apply GPU runtime environment variables to current process if not already set
    pub fn apply_environment(&self, device_str: &str) {
        for (k, v) in self.get_environment_for_device(device_str) {
            if std::env::var_os(&k).is_none() {
                unsafe {
                    std::env::set_var(k, v);
                }
            }
        }
    }

    /// Resolve the optimal `UV_TORCH_BACKEND` string (`rocm7.1`, `rocm6.3`, `auto`, etc.) for `uv`
    pub fn uv_torch_backend(&self) -> String {
        if let Ok(explicit) = std::env::var("UV_TORCH_BACKEND") {
            if !explicit.trim().is_empty() {
                return explicit;
            }
        }
        match &self.gpu {
            GpuKind::Rocm { .. } => {
                if let Ok(out) = std::process::Command::new("hipconfig").arg("--version").output() {
                    let ver = String::from_utf8_lossy(&out.stdout).trim().to_string();
                    let parts: Vec<&str> = ver.split('.').collect();
                    if parts.len() >= 2 {
                        let major_minor = format!("{}.{}", parts[0], parts[1]);
                        return match major_minor.as_str() {
                            "7.2" => "rocm7.2".to_string(),
                            "7.1" => "rocm7.1".to_string(),
                            "7.0" => "rocm7.0".to_string(),
                            "6.4" => "rocm6.4".to_string(),
                            "6.3" => "rocm6.3".to_string(),
                            "6.2" => "rocm6.2.4".to_string(),
                            _ => "rocm7.1".to_string(),
                        };
                    }
                }
                "rocm7.1".to_string()
            }
            GpuKind::Cuda { .. } => "auto".to_string(),
            GpuKind::Metal { .. } | GpuKind::None => "auto".to_string(),
        }
    }

    /// Print formatted diagnostics
    pub fn print_diagnostics(&self) {
        println!("  {:<24} : {}", "Target OS".dimmed(), self.os.cyan());
        println!("  {:<24} : {}", "Target Architecture".dimmed(), self.arch.cyan());

        print!("  {:<24} : ", "CPU SIMD Acceleration".dimmed());
        if self.cpu_simd.is_empty() {
            println!("{}", "Standard generic CPU".yellow());
        } else {
            println!("{}", self.cpu_simd.join(", ").green().bold());
        }

        println!("\n  {} Cross-Device Hardware:", "🖥️".cyan().bold());
        match &self.gpu {
            GpuKind::Rocm { name, arch } => {
                println!("    • Platform : {}", "AMD ROCm / HIP (Active)".green().bold());
                println!("    • Device   : {}", name.bright_white().bold());
                println!("    • Target   : GFX override = {}", arch.cyan());
            }
            GpuKind::Cuda { name } => {
                println!("    • Platform : {}", "NVIDIA CUDA (Active)".green().bold());
                println!("    • Device   : {}", name.bright_white().bold());
            }
            GpuKind::Metal { name } => {
                println!("    • Platform : {}", "Apple Silicon Metal (Active)".green().bold());
                println!("    • Device   : {}", name.bright_white().bold());
            }
            GpuKind::None => {
                println!("    • Platform : {}", "CPU Only (No discrete GPU detected)".yellow());
            }
        }
    }
}
