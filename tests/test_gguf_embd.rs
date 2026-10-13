use candle_core::Device;
use std::fs::File;
use std::path::PathBuf;

#[test]
fn test_inspect_token_embd() {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/pierre".to_string());
    let path = PathBuf::from(home).join(".ollaya/models/blobs/sha256-840e3f50e5a9c218727f44e121d1b37cc9e2c3b318c8eb422ba6ef2e27b618a2");
    if !path.exists() {
        println!("File not found: {:?}", path);
        return;
    }
    let mut file = File::open(&path).expect("Failed to open GGUF");
    let content = candle_core::quantized::gguf_file::Content::read(&mut file).expect("Failed to read GGUF");

    let device = Device::Cpu;
    let t_name = "token_embd.weight";
    let qtensor = content.tensor(&mut file, t_name, &device).expect("Failed to load tensor");
    
    // Winnow label IDs:
    // A: 236776, B: 236799, C: 236780
    // Let's dequantize full embedding or test
    let dequant_start = std::time::Instant::now();
    let embd_f32 = qtensor.dequantize(&device).expect("Failed to dequantize");
    println!("Dequantized full 262k x 2560 embedding table in {:?}", dequant_start.elapsed());
    println!("Shape: {:?}", embd_f32.shape());
    
    // Slice tokens A, B, C
    let emb_a = embd_f32.get(236776).unwrap();
    let emb_b = embd_f32.get(236799).unwrap();
    let emb_c = embd_f32.get(236780).unwrap();
    println!("Embedding vector A shape: {:?}", emb_a.shape());
    println!("Embedding vector B shape: {:?}", emb_b.shape());
    println!("Embedding vector C shape: {:?}", emb_c.shape());
}
