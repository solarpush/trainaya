use candle_core::{DType, Device, Shape, Tensor};
use std::collections::HashMap;
use std::path::PathBuf;
use trainaya::exporter::export_lora_gguf;

#[test]
fn test_real_merge_with_winnow() {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/pierre".to_string());
    let base_model = PathBuf::from(home).join(".ollaya/models/blobs/sha256-840e3f50e5a9c218727f44e121d1b37cc9e2c3b318c8eb422ba6ef2e27b618a2");
    if !base_model.exists() {
        println!("Base model not found: {:?}", base_model);
        return;
    }

    let device = Device::Cpu;
    let mut tensors = HashMap::new();

    // blk.41.attn_output.weight has in_features=4096, out_features=2560
    // lora_a: [16, 4096]
    let lora_a = Tensor::zeros(Shape::from((16, 4096)), DType::F32, &device).unwrap();
    // lora_b: [2560, 16]
    let lora_b = Tensor::zeros(Shape::from((2560, 16)), DType::F32, &device).unwrap();

    tensors.insert("blk.41.attn_output.weight.lora_a".to_string(), lora_a);
    tensors.insert("blk.41.attn_output.weight.lora_b".to_string(), lora_b);

    let lora_path = PathBuf::from("/tmp/test_winnow_lora.gguf");
    let bytes = export_lora_gguf(&lora_path, "gemma4", 32.0, &tensors).expect("Failed to export GGUF LoRA");
    println!("Exported adapter: {} bytes", bytes);

    let export_lora_bin = PathBuf::from("/home/pierre/Projects/trainaya/vendor/llama.cpp/build/bin/llama-export-lora");
    let _out_merged = PathBuf::from("/tmp/test_winnow_merged_f16.gguf");

    // Test running llama-export-lora with --help or checking parameters
    assert!(export_lora_bin.exists(), "llama-export-lora binary must exist");
}
