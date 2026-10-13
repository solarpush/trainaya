use clap::Parser;
use std::path::PathBuf;
use trainaya::cli::{Cli, Commands};
use trainaya::config::TaskType;

#[test]
fn test_cli_train_command_parsing() {
    let args = vec![
        "trainaya",
        "train",
        "--base",
        "answerdotai/ModernBERT-base",
        "--data",
        "examples/sample_classification.jsonl",
        "--task",
        "classification",
        "--epochs",
        "5",
        "--batch-size",
        "16",
        "--lr",
        "0.0001",
        "--lora-r",
        "8",
        "--lora-alpha",
        "16.0",
        "--output-dir",
        "./custom_output",
    ];

    let cli = Cli::try_parse_from(args).expect("Failed to parse CLI args");
    match cli.command {
        Commands::Train(opts) => {
            assert_eq!(opts.base, "answerdotai/ModernBERT-base");
            assert_eq!(opts.data, Some(PathBuf::from("examples/sample_classification.jsonl")));
            assert_eq!(opts.task, TaskType::Classification);
            assert_eq!(opts.epochs, Some(5));
            assert_eq!(opts.batch_size, Some(16));
            assert_eq!(opts.lr, Some(0.0001));
            assert_eq!(opts.lora_r, Some(8));
            assert_eq!(opts.lora_alpha, Some(16.0));
            assert_eq!(opts.output_dir, Some(PathBuf::from("./custom_output")));

            let (training_args, _, _) = opts.resolve_training_args().unwrap();
            assert_eq!(training_args.epochs, 5);
            assert_eq!(training_args.batch_size, 16);
            assert_eq!(training_args.lr, 0.0001);
            assert_eq!(training_args.lora.r, 8);
        }
        _ => panic!("Expected Train command"),
    }
}

#[test]
fn test_cli_merge_command_parsing() {
    let args = vec![
        "trainaya",
        "merge",
        "--base",
        "answerdotai/ModernBERT-base",
        "--lora",
        "./custom_output",
        "--output-dir",
        "./merged_output",
        "--device",
        "cpu",
    ];

    let cli = Cli::try_parse_from(args).expect("Failed to parse CLI args");
    match cli.command {
        Commands::Merge(opts) => {
            assert_eq!(opts.base, "answerdotai/ModernBERT-base");
            assert_eq!(opts.lora, PathBuf::from("./custom_output"));
            assert_eq!(opts.output_dir, PathBuf::from("./merged_output"));
            assert_eq!(opts.device, "cpu");
        }
        _ => panic!("Expected Merge command"),
    }
}

#[test]
fn test_cli_export_command_parsing() {
    let args = vec![
        "trainaya",
        "export",
        "--model",
        "./merged_output",
        "--format",
        "gguf",
        "--quantize",
        "int8",
        "--output-dir",
        "./ollaya_export",
    ];

    let cli = Cli::try_parse_from(args).expect("Failed to parse CLI args");
    match cli.command {
        Commands::Export(opts) => {
            assert_eq!(opts.model, PathBuf::from("./merged_output"));
            assert_eq!(opts.format, trainaya::config::ExportFormat::Gguf);
            assert_eq!(opts.quantize, Some("int8".to_string()));
            assert_eq!(opts.output_dir, PathBuf::from("./ollaya_export"));
        }
        _ => panic!("Expected Export command"),
    }
}

#[test]
fn test_cli_train_with_trainfile() {
    let dir = tempfile::tempdir().unwrap();
    let trainfile_path = dir.path().join(".Trainfile");
    let content = r#"
FROM winnow:e4b
DATA ./data.jsonl
PARAMETER epochs 7
PARAMETER lr 3e-5
MERGE true
REGISTER winnow:custom_triage
"#;
    std::fs::write(&trainfile_path, content).unwrap();

    let args = vec![
        "trainaya",
        "train",
        "-f",
        trainfile_path.to_str().unwrap(),
        "--epochs",
        "10", // override
    ];

    let cli = Cli::try_parse_from(args).expect("Failed to parse CLI args");
    match cli.command {
        Commands::Train(opts) => {
            assert_eq!(opts.file, Some(trainfile_path));
            assert_eq!(opts.epochs, Some(10));

            let (training_args, register, _) = opts.resolve_training_args().unwrap();
            assert_eq!(training_args.base_model, "winnow:e4b");
            assert_eq!(training_args.epochs, 10); // Overridden from 7 to 10
            assert_eq!(training_args.lr, 3e-5);
            assert_eq!(training_args.merge_mode, trainaya::config::MergeMode::Merge);
            assert_eq!(register, Some("winnow:custom_triage".to_string()));
        }
        _ => panic!("Expected Train command"),
    }
}

#[test]
fn test_dynamic_tag_and_output_expansion() {
    let tmp = tempfile::tempdir().unwrap();
    let trainfile_path = tmp.path().join("Trainfile");

    let content = r#"
FROM winnow:e4b
DATA examples/sample_classification.jsonl
OUTPUT ./runs/{tag}
TAG winnow:v{auto}
"#;
    std::fs::write(&trainfile_path, content).unwrap();

    let args = vec![
        "trainaya",
        "train",
        "-f",
        trainfile_path.to_str().unwrap(),
        "--models-dir",
        tmp.path().to_str().unwrap(),
    ];

    let cli = Cli::try_parse_from(args).expect("Failed to parse CLI args");
    match cli.command {
        Commands::Train(opts) => {
            let (training_args, tag, _) = opts.resolve_training_args().unwrap();
            // First run: winnow:v1
            assert_eq!(tag, Some("winnow:v1".to_string()));
            assert!(training_args.output_dir.to_str().unwrap().ends_with("winnow_v1"));
        }
        _ => panic!("Expected Train command"),
    }
}

