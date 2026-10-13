use std::fs::File;
use std::path::PathBuf;

#[test]
fn test_inspect_winnow_gguf() {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/pierre".to_string());
    let path = PathBuf::from(home).join(".ollaya/models/blobs/sha256-840e3f50e5a9c218727f44e121d1b37cc9e2c3b318c8eb422ba6ef2e27b618a2");
    if !path.exists() {
        println!("File not found: {:?}", path);
        return;
    }
    let mut file = File::open(&path).expect("Failed to open GGUF");
    let content = candle_core::quantized::gguf_file::Content::read(&mut file).expect("Failed to parse GGUF content");
    
    println!("=== GGUF Metadata ===");
    for (k, v) in &content.metadata {
        if k.starts_with("general.") || k.contains("architecture") || k.contains("layer_count") || k.contains("context_length") {
            println!("  {} = {:?}", k, v);
        }
    }
    
    println!("=== GGUF Tensors (Total: {}) ===", content.tensor_infos.len());
    let mut sample_count = 0;
    for (name, info) in &content.tensor_infos {
        if sample_count < 15 || name.contains("output") {
            println!("  Tensor: {} -> shape: {:?}, ggml_dtype: {:?}", name, info.shape, info.ggml_dtype);
            sample_count += 1;
        }
    }
}
