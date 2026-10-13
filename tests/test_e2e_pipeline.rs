use candle_core::{DType, Device, Shape, Tensor};
use std::collections::HashMap;
use trainaya::config::{ExportArgs, ExportFormat, MergeArgs};
use trainaya::exporter::run_export;
use trainaya::merger::run_merge;

#[test]
fn test_e2e_merge_and_export_pipeline() {
    let temp_dir = tempfile::tempdir().unwrap();
    let base_dir = temp_dir.path().join("base_model");
    let lora_dir = temp_dir.path().join("lora_adapter");
    let merged_dir = temp_dir.path().join("merged_model");
    let export_dir = temp_dir.path().join("ollaya_export");

    std::fs::create_dir_all(&base_dir).unwrap();
    std::fs::create_dir_all(&lora_dir).unwrap();

    let device = Device::Cpu;

    // 1. Create dummy base model
    let mut base_tensors = HashMap::new();
    let weight_tensor = Tensor::ones(Shape::from((4, 8)), DType::F32, &device).unwrap();
    base_tensors.insert("model.layer.0.attention.query.weight".to_string(), weight_tensor);
    candle_core::safetensors::save(&base_tensors, base_dir.join("model.safetensors")).unwrap();

    // Base config
    let base_config = serde_json::json!({
        "model_type": "modernbert",
        "hidden_size": 8,
        "num_attention_heads": 2
    });
    std::fs::write(
        base_dir.join("config.json"),
        serde_json::to_string(&base_config).unwrap(),
    )
    .unwrap();

    // Dummy tokenizer.json
    let dummy_tok = serde_json::json!({"version": "1.0"});
    std::fs::write(
        base_dir.join("tokenizer.json"),
        serde_json::to_string(&dummy_tok).unwrap(),
    )
    .unwrap();

    // 2. Create dummy LoRA adapter
    let mut lora_tensors = HashMap::new();
    // LoRA A: [rank=2, in_features=8]
    let lora_a = Tensor::ones(Shape::from((2, 8)), DType::F32, &device).unwrap();
    // LoRA B: [out_features=4, rank=2]
    let lora_b = Tensor::ones(Shape::from((4, 2)), DType::F32, &device).unwrap();

    lora_tensors.insert(
        "model.layer.0.attention.query.weight.lora_a.weight".to_string(),
        lora_a,
    );
    lora_tensors.insert(
        "model.layer.0.attention.query.weight.lora_b.weight".to_string(),
        lora_b,
    );

    // Classification head
    let head_w = Tensor::ones(Shape::from((2, 8)), DType::F32, &device).unwrap();
    let head_b = Tensor::zeros(Shape::from(2), DType::F32, &device).unwrap();
    lora_tensors.insert("classifier.weight".to_string(), head_w);
    lora_tensors.insert("classifier.bias".to_string(), head_b);

    candle_core::safetensors::save(&lora_tensors, lora_dir.join("adapter_model.safetensors")).unwrap();

    let adapter_config = serde_json::json!({
        "r": 2,
        "lora_alpha": 4.0,
        "base_model_name_or_path": base_dir.to_str().unwrap()
    });
    std::fs::write(
        lora_dir.join("adapter_config.json"),
        serde_json::to_string(&adapter_config).unwrap(),
    )
    .unwrap();

    // 3. Test Merge
    let merge_args = MergeArgs {
        base_model: base_dir.to_str().unwrap().to_string(),
        lora_path: lora_dir.clone(),
        output_dir: merged_dir.clone(),
        device: "cpu".to_string(),
    };

    let merge_summary = run_merge(merge_args).expect("Merge failed");
    assert_eq!(merge_summary.merged_layers_count, 1);
    assert!(merged_dir.join("model.safetensors").exists());
    assert!(merged_dir.join("config.json").exists());
    assert!(merged_dir.join("tokenizer.json").exists());

    // Verify merged weights mathematically:
    // W_base = 1.0 everywhere (4x8)
    // lora_b = 1.0 (4x2), lora_a = 1.0 (2x8) -> B @ A = 2.0 everywhere (4x8)
    // scale = 4.0 / 2 = 2.0
    // delta = 2.0 * 2.0 = 4.0
    // W_merged = 1.0 + 4.0 = 5.0
    let merged_tensors =
        candle_core::safetensors::load(merged_dir.join("model.safetensors"), &device).unwrap();
    let merged_q = merged_tensors
        .get("model.layer.0.attention.query.weight")
        .unwrap();
    let q_vec = merged_q.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    for &val in &q_vec {
        assert!((val - 5.0).abs() < 1e-5, "Expected 5.0, got {}", val);
    }

    // 4. Test Export to SafeTensors + INT8
    let export_args = ExportArgs {
        model_path: merged_dir.clone(),
        format: ExportFormat::Safetensors,
        quantize: Some("int8".to_string()),
        output_dir: export_dir.clone(),
    };

    let export_summary = run_export(export_args).expect("Export failed");
    assert_eq!(export_summary.format, ExportFormat::Safetensors);
    assert!(export_dir.join("model.safetensors").exists());
    assert!(export_dir.join("int8_scales.json").exists());
    assert!(export_dir.join("ollaya_manifest.json").exists());

    // 5. Test Export to GGUF
    let gguf_export_dir = temp_dir.path().join("gguf_export");
    let gguf_args = ExportArgs {
        model_path: merged_dir.clone(),
        format: ExportFormat::Gguf,
        quantize: None,
        output_dir: gguf_export_dir.clone(),
    };

    let gguf_summary = run_export(gguf_args).expect("GGUF export failed");
    assert_eq!(gguf_summary.format, ExportFormat::Gguf);
    assert!(gguf_export_dir.join("model.gguf").exists());
    assert!(gguf_export_dir.join("ollaya_manifest.json").exists());
}
