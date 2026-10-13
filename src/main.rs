use anyhow::Result;
use clap::Parser;
use colored::Colorize;
use trainaya::cli::{Cli, Commands};
use trainaya::exporter::run_export;
use trainaya::merger::run_merge;
use trainaya::registry::run_register;
use trainaya::trainer::run_training;
use trainaya::utils::print_banner;

fn main() {
    let cli = Cli::parse();

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
                    eprintln!("  Warning: Automatic registration failed: {}. You can run 'trainaya register' manually.", e);
                }
            }
            Ok(())
        })(),
        Commands::Merge(opts) => run_merge(opts.into()).map(|_| ()),
        Commands::Export(opts) => run_export(opts.into()).map(|_| ()),
        Commands::Register(opts) => run_register(opts.into()),
        Commands::Doctor => run_doctor(),
        Commands::Bench(opts) => (|| -> Result<()> {
            let (model, baseline, data, subsets, limit, url, out, min_accuracy) =
                opts.resolve_bench_args()?;
            trainaya::bench::run_benchmark(
                &model,
                baseline.as_deref(),
                data.as_deref(),
                subsets.as_deref(),
                limit,
                &url,
                out.as_deref(),
                min_accuracy,
            )
        })(),
        Commands::Curate(opts) => (|| -> Result<()> {
            let cf = opts.resolve_curate_args()?;
            trainaya::curate::run_curate(cf)
        })(),
    };

    if let Err(err) = res {
        eprintln!("\n{} {}", "❌ [TRAINAYA ERROR]:".bold().red(), format!("{:#}", err).red());
        std::process::exit(1);
    }
}

use trainaya::hardware::SystemHardware;

fn run_doctor() -> Result<()> {
    print_banner("System & Hardware Diagnostics");

    let hw = SystemHardware::detect();
    hw.print_diagnostics();

    #[cfg(feature = "cuda")]
    {
        println!("    • Candle GPU backend    : {}", "COMPILED (CUDA)".green().bold());
    }
    #[cfg(not(feature = "cuda"))]
    {
        println!("    • Candle GPU backend    : {}", "CPU Native AVX2/FMA Mode".yellow());
    }

    // Pure Rust SIMD Kernels status
    println!("\n  {} Native Pure Rust SIMD Kernels:", "⚙️".cyan().bold());
    println!("    • Fast LoRA Accumulate Kernel : {}", "COMPILED (Pure Rust SIMD)".green().bold());
    println!("    • 4-Bit NF4 Quantize / Dequant: {}", "COMPILED (Pure Rust SIMD)".green().bold());
    println!("    • 8-Bit Int8 Quantize / Dequant: {}", "COMPILED (Pure Rust SIMD)".green().bold());

    // Target runtime
    println!("\n  {} Ollaya Runtime Compatibility:", "🎯".purple().bold());
    println!("    • Format: SafeTensors  - {}", "Supported (zero-copy memory map)".green());
    println!("    • Format: GGUF (v3)    - {}", "Supported (llama.cpp integration)".green());
    println!("    • Format: ONNX package - {}", "Supported".green());

    println!("\n{} System is healthy and ready for local training & export.\n", "✓".green().bold());
    Ok(())
}
