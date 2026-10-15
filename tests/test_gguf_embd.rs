use candle_core::Device;
use std::fs::File;
use std::path::PathBuf;

#[test]
fn test_inspect_token_embd() {
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
    let qtensor = content.tensor(&mut file, "token_embd.weight", &device).expect("Failed to load tensor");
    let out_norm = content.tensor(&mut file, "output_norm.weight", &device).unwrap().dequantize(&device).unwrap();
    let post_attn_norm = content.tensor(&mut file, "blk.41.post_attention_norm.weight", &device).unwrap().dequantize(&device).unwrap();

    let dequant_start = std::time::Instant::now();
    let embd_f32 = qtensor.dequantize(&device).expect("Failed to dequantize");
    println!("Dequantized full 262k x 2560 embedding table in {:?}", dequant_start.elapsed());

    for (label, id) in [("A", 236776usize), ("B", 236799), ("C", 236780), ("D", 236796), ("E", 236788)] {
        let emb = embd_f32.get(id).unwrap();
        let raw_norm = emb.sqr().unwrap().sum_all().unwrap().to_vec0::<f32>().unwrap().sqrt();
        let eff_emb = (&(&emb * &out_norm).unwrap() * &post_attn_norm).unwrap();
        let eff_norm = eff_emb.sqr().unwrap().sum_all().unwrap().to_vec0::<f32>().unwrap().sqrt();
        println!("Option {} (id={}): raw_norm={:.4}, effective_norm(with output_norm*post_attn_norm)={:.4}", label, id, raw_norm, eff_norm);
    }
}
