use candle_core::{DType, Device};
use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;

#[test]
fn test_dequantize_gguf_layer() {
    let path = match trainaya::models::gguf_lora::resolve_gguf_path("winnow:e4b") {
        Ok(p) => p,
        Err(_) => {
            println!("winnow:e4b not found, skipping");
            return;
        }
    };
    let mut file = File::open(&path).expect("Failed to open GGUF");
    let content = candle_core::quantized::gguf_file::Content::read(&mut file).expect("Failed to read GGUF");

    let device = Device::Cpu;
    for t_name in [
        "blk.40.attn_output.weight",
        "blk.41.attn_output.weight",
        "blk.41.attn_v.weight",
        "blk.41.post_attention_norm.weight",
        "blk.41.layer_output_scale.weight",
        "output_norm.weight",
    ] {
        let qtensor = content.tensor(&mut file, t_name, &device).expect("Failed to read tensor");
        let dequant_start = Instant::now();
        let tensor_f32 = qtensor.dequantize(&device).expect("Failed to dequantize");
        let elapsed = dequant_start.elapsed();
        let norm = tensor_f32.sqr().unwrap().sum_all().unwrap().to_vec0::<f32>().unwrap().sqrt();
        println!(
            "Dequantized {:35} in {:8?}: shape={:?}, frob_norm={:.4}",
            t_name,
            elapsed,
            tensor_f32.shape(),
            norm
        );
        assert_eq!(tensor_f32.dtype(), DType::F32);
    }
}
