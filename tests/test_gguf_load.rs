use candle_core::{DType, Device};
use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;

#[test]
fn test_dequantize_gguf_layer() {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/pierre".to_string());
    let path = PathBuf::from(home).join(".ollaya/models/blobs/sha256-840e3f50e5a9c218727f44e121d1b37cc9e2c3b318c8eb422ba6ef2e27b618a2");
    if !path.exists() {
        println!("File not found: {:?}", path);
        return;
    }
    let mut file = File::open(&path).expect("Failed to open GGUF");
    let content = candle_core::quantized::gguf_file::Content::read(&mut file).expect("Failed to read GGUF");

    let device = Device::Cpu;
    let t_name = "blk.41.attn_q.weight";
    
    let _start = Instant::now();
    let qtensor = content.tensor(&mut file, t_name, &device).expect("Failed to read tensor");
    let dequant_start = Instant::now();
    let tensor_f32 = qtensor.dequantize(&device).expect("Failed to dequantize");
    let elapsed = dequant_start.elapsed();
    
    println!("Dequantized {} in {:?}", t_name, elapsed);
    println!("  Shape: {:?}", tensor_f32.shape());
    println!("  DType: {:?}", tensor_f32.dtype());
    assert_eq!(tensor_f32.dtype(), DType::F32);
}
