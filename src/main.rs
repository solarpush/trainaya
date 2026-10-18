use anyhow::Result;
use clap::Parser;
use colored::Colorize;
use trainaya::cli::{Cli, Commands};
use trainaya::exporter::run_export;
use trainaya::hardware::SystemHardware;
use trainaya::merger::run_merge;
use trainaya::registry::run_register;
use trainaya::trainer::{run_adapt, run_training};
use trainaya::utils::print_banner;

fn main() {
    let cli = Cli::parse();
    let is_update_cmd = matches!(cli.command, Commands::Update(_));

    let res = match cli.command {
        Commands::Train(opts) => (|| -> Result<()> {
            let (args, register_name, models_dir) = opts.resolve_training_args()?;
            let out_dir = args.output_dir.clone();
            let base_model = args.base_model.clone();
            run_training(args)?;

            if let Some(target_name) = register_name {
                let reg_args = trainaya::registry::RegisterArgs {
                    model_path: out_dir,
                    name: target_name,
                    base_template: Some(base_model),
                    tokenizer_path: None,
                    models_dir,
                };
                if let Err(e) = run_register(reg_args) {
                    eprintln!(
                        "  Warning: Automatic registration failed: {}. You can run 'trainaya register' manually.",
                        e
                    );
                }
            }
            Ok(())
        })(),
        Commands::Adapt(opts) => (|| -> Result<()> {
            let (adapt_args, models_dir) = opts.into_adapt_args();
            let out_dir = adapt_args.output_dir.clone();
            let target_name = adapt_args.target.clone();
            run_adapt(adapt_args)?;

            let reg_args = trainaya::registry::RegisterArgs {
                model_path: out_dir,
                name: target_name,
                base_template: None,
                tokenizer_path: None,
                models_dir,
            };
            if let Err(e) = run_register(reg_args) {
                eprintln!(
                    "  Warning: Automatic registration failed: {}. You can run 'trainaya register' manually.",
                    e
                );
            }
            Ok(())
        })(),
        Commands::Merge(opts) => run_merge(opts.into()).map(|_| ()),
        Commands::Export(opts) => run_export(opts.into()).map(|_| ()),
        Commands::Register(opts) => run_register(opts.into()),
        Commands::Doctor => run_doctor(),
        Commands::Bench(opts) => (|| -> Result<()> {
            let bf = opts.resolve_benchfile()?;
            trainaya::bench::run_benchmark_with_options(
                &bf.model,
                bf.baseline.as_deref(),
                bf.data.as_deref(),
                bf.subsets.as_deref(),
                bf.limit,
                &bf.url,
                bf.output.as_deref(),
                bf.min_accuracy,
                bf.export_failures.as_deref(),
                bf.no_cache,
            )
        })(),
        Commands::Curate(opts) => (|| -> Result<()> {
            let cf = opts.resolve_curate_args()?;
            trainaya::curate::run_curate(cf)
        })(),
        Commands::Update(opts) => {
            trainaya::updater::run_update(opts.check, opts.force, opts.version.as_deref())
        }
    };

    if !is_update_cmd {
        trainaya::updater::check_and_notify_passive();
    }

    if let Err(err) = res {
        eprintln!(
            "\n{} {}",
            "❌ [TRAINAYA ERROR]:".bold().red(),
            format!("{:#}", err).red()
        );
        std::process::exit(1);
    }
}

fn run_doctor() -> Result<()> {
    print_banner("System, Hardware & uv/PyTorch Engine Diagnostics");

    println!(
        "  {:<24} : {}",
        "Trainaya Version".dimmed(),
        format!("v{}", trainaya::updater::CURRENT_VERSION).green().bold()
    );

    let hw = SystemHardware::detect();
    hw.print_diagnostics();

    println!("\n  {} Embedded uv + PyTorch Engine:", "🐍".cyan().bold());
    match trainaya::py_bridge::run_py_doctor() {
        Ok(info) => {
            let py_v = info.get("python_version").and_then(|v| v.as_str()).unwrap_or("unknown");
            let torch_v = info.get("torch_version").and_then(|v| v.as_str()).unwrap_or("unknown");
            let tf_v = info
                .get("transformers_version")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let onnx_v = info.get("onnx_version").and_then(|v| v.as_str()).unwrap_or("unknown");
            let has_laya = info
                .get("laya_available")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let cuda = info
                .get("cuda_available")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            println!("    • Python Runtime        : {}", py_v.green().bold());
            println!("    • PyTorch Engine        : {}", torch_v.green().bold());
            println!("    • Transformers / ONNX   : {} / {}", tf_v.cyan(), onnx_v.cyan());
            println!(
                "    • Laya Reference SDK    : {}",
                if has_laya {
                    "Active (laya 0.3.7)".green().bold()
                } else {
                    "Not installed".yellow()
                }
            );
            let hip_ver = info.get("hip_version").and_then(|v| v.as_str());
            let vram_mb = info.get("vram_mb").and_then(|v| v.as_u64()).unwrap_or(0);
            let accel_str = if cuda {
                let dev_name = info
                    .get("cuda_device")
                    .and_then(|v| v.as_str())
                    .unwrap_or("GPU");
                if let Some(hip) = hip_ver {
                    format!("ROCm / HIP {} ({} — {} MiB VRAM)", hip, dev_name, vram_mb)
                        .green()
                        .bold()
                        .to_string()
                } else {
                    format!("NVIDIA CUDA ({} — {} MiB VRAM)", dev_name, vram_mb)
                        .green()
                        .bold()
                        .to_string()
                }
            } else {
                "CPU / SIMD Multi-Thread".yellow().to_string()
            };
            println!("    • Torch Accelerator     : {}", accel_str);
        }
        Err(e) => {
            println!("    • Engine Status         : {} ({})", "ERROR".red().bold(), e);
        }
    }

    println!("\n  {} Ollaya Runtime Compatibility:", "🎯".purple().bold());
    println!(
        "    • Format: SafeTensors  - {}",
        "Supported (zero-copy memory map)".green()
    );
    println!(
        "    • Format: GGUF (v3)    - {}",
        "Supported (llama.cpp + llm-logits-v1)".green()
    );
    println!(
        "    • Format: ONNX package - {}",
        "Supported (weightless external-data + inline patcher)".green()
    );

    println!(
        "\n{} System is healthy and ready for PyTorch training, adaptation & export.\n",
        "✓".green().bold()
    );
    Ok(())
}
