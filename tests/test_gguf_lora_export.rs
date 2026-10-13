use candle_core::{DType, Device, Shape, Tensor};
use candle_core::quantized::gguf_file::Value;
use std::collections::HashMap;
use trainaya::exporter::export_lora_gguf;

#[test]
fn test_export_lora_gguf_format() {
    let device = Device::Cpu;
    let mut tensors = HashMap::new();

    // blk.41.attn_q.weight has in_features=2560, out_features=4096
    // lora_a: [16, 2560]
    let lora_a = Tensor::zeros(Shape::from((16, 2560)), DType::F32, &device).unwrap();
    // lora_b: [4096, 16]
    let lora_b = Tensor::zeros(Shape::from((4096, 16)), DType::F32, &device).unwrap();

    tensors.insert("blk.41.attn_q.weight.lora_a".to_string(), lora_a);
    tensors.insert("blk.41.attn_q.weight.lora_b".to_string(), lora_b);

    let temp_dir = tempfile::tempdir().unwrap();
    let lora_path = temp_dir.path().join("adapter.gguf");

    let bytes = export_lora_gguf(&lora_path, "gemma4", 32.0, &tensors).expect("Failed to export GGUF LoRA");
    assert!(bytes > 0);

    // Verify it parses back with candle
    let mut f = std::fs::File::open(&lora_path).unwrap();
    let content = candle_core::quantized::gguf_file::Content::read(&mut f).unwrap();

    match content.metadata.get("general.type") {
        Some(Value::String(s)) => assert_eq!(s, "adapter"),
        other => panic!("Expected general.type = adapter, got {:?}", other),
    }
    match content.metadata.get("adapter.type") {
        Some(Value::String(s)) => assert_eq!(s, "lora"),
        other => panic!("Expected adapter.type = lora, got {:?}", other),
    }
    match content.metadata.get("general.architecture") {
        Some(Value::String(s)) => assert_eq!(s, "gemma4"),
        other => panic!("Expected general.architecture = gemma4, got {:?}", other),
    }
    match content.metadata.get("adapter.lora.alpha") {
        Some(Value::F32(v)) => assert_eq!(*v, 32.0),
        other => panic!("Expected adapter.lora.alpha = 32.0, got {:?}", other),
    }

    assert!(content.tensor_infos.contains_key("blk.41.attn_q.weight.lora_a"));
    assert!(content.tensor_infos.contains_key("blk.41.attn_q.weight.lora_b"));
    println!("GGUF LoRA adapter successfully verified: {} bytes", bytes);
}
