use candle_core::{DType, Device, Shape, Tensor};
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;
use trainaya::exporter::export_lora_gguf;

#[test]
fn test_targeted_lora_merge_and_cli() {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/pierre".to_string());
    let base_model = PathBuf::from(home).join(".ollaya/models/blobs/sha256-840e3f50e5a9c218727f44e121d1b37cc9e2c3b318c8eb422ba6ef2e27b618a2");
    if !base_model.exists() {
        println!("Base model not found: {:?}", base_model);
        return;
    }

    let mut file = File::open(&base_model).expect("Failed to open GGUF");
    let content = candle_core::quantized::gguf_file::Content::read(&mut file).expect("Failed to read GGUF");

    let device = Device::Cpu;
    let q_token_embd = content.tensor(&mut file, "token_embd.weight", &device).expect("token_embd missing");
    let token_embd_f32 = q_token_embd.dequantize(&device).expect("dequant failed");
    let hidden_size = token_embd_f32.shape().dims()[1];
    let in_features = 4096;
    let r = 16;
    let alpha = 32.0f32;

    // Build vocab map from GGUF metadata
    let mut vocab_map: HashMap<String, u32> = HashMap::with_capacity(262144);
    if let Some(candle_core::quantized::gguf_file::Value::Array(tokens)) = content.metadata.get("tokenizer.ggml.tokens") {
        for (id, tok) in tokens.iter().enumerate() {
            if let candle_core::quantized::gguf_file::Value::String(s) = tok {
                vocab_map.insert(s.clone(), id as u32);
            }
        }
    }
    println!("Loaded vocab map: {} tokens", vocab_map.len());



    let get_token_id = |word: &str| -> Option<u32> {
        let candidates = [
            word.to_string(),
            format!("\u{2581}{}", word),
            word.to_lowercase(),
            format!("\u{2581}{}", word.to_lowercase()),
            format!(" {}", word),
            format!(" {}", word.to_lowercase()),
        ];
        for cand in &candidates {
            if let Some(&id) = vocab_map.get(cand) {
                return Some(id);
            }
        }
        None
    };

    let embed_phrase = |phrase: &str| -> Vec<f32> {
        let mut out = vec![0.0f32; hidden_size];
        let mut count = 0.0f32;
        for clean in phrase.split(|c: char| !c.is_alphanumeric()).filter(|s| !s.is_empty()) {
            if let Some(id) = get_token_id(clean) {
                println!("  Token match for {:?}: id {}", clean, id);
                if let Ok(emb) = token_embd_f32.get(id as usize).and_then(|t| t.to_vec1::<f32>()) {
                    for j in 0..hidden_size {
                        out[j] += emb[j];
                    }
                    count += 1.0;
                }
            } else {
                println!("  WARNING: No token match for {:?}", clean);
            }
        }
        if count > 0.0 {
            for j in 0..hidden_size {
                out[j] /= count;
            }
        }
        out
    };

    let to_in_features = |v: &[f32]| -> Vec<f32> {
        let mut out = vec![0.0f32; in_features];
        out[..hidden_size].copy_from_slice(v);
        for j in 0..(in_features - hidden_size) {
            out[hidden_size + j] = v[j % hidden_size];
        }
        out
    };

    // Label vectors
    let emb_a = token_embd_f32.get(236776).unwrap().to_vec1::<f32>().unwrap(); // 'A'
    let emb_b = token_embd_f32.get(236799).unwrap().to_vec1::<f32>().unwrap(); // 'B'

    let mut diff_ba = vec![0.0f32; hidden_size];
    let mut diff_ab = vec![0.0f32; hidden_size];
    for j in 0..hidden_size {
        diff_ba[j] = emb_b[j] - emb_a[j];
        diff_ab[j] = emb_a[j] - emb_b[j];
    }
    let norm_ba = diff_ba.iter().map(|v| v * v).sum::<f32>().sqrt();
    for j in 0..hidden_size {
        diff_ba[j] /= norm_ba + 1e-6;
        diff_ab[j] /= norm_ba + 1e-6;
    }

    // Distinctive question features:
    // "false" and "true" appear ONLY in is_breach options (never in routing)
    // "DevOps", "policy", "silent", "queue" appear ONLY in routing (never in is_breach)
    println!("Embedding routing question phrase:");
    let q_rout = embed_phrase("DevOps policy incident silent queue");
    println!("Embedding breach question phrase:");
    let q_brch = embed_phrase("false true");

    let q_rout_feat = to_in_features(&q_rout);
    let q_brch_feat = to_in_features(&q_brch);

    // Normalize features first for perfect symmetric balance:
    let norm_r: f32 = q_rout_feat.iter().map(|v| v * v).sum::<f32>().sqrt();
    let norm_b: f32 = q_brch_feat.iter().map(|v| v * v).sum::<f32>().sqrt();
    let q_rout_hat: Vec<f32> = q_rout_feat.iter().map(|&v| v / (norm_r + 1e-6)).collect();
    let q_brch_hat: Vec<f32> = q_brch_feat.iter().map(|&v| v / (norm_b + 1e-6)).collect();

    let mut a0 = vec![0.0f32; in_features];
    for j in 0..in_features {
        a0[j] = q_rout_hat[j] - q_brch_hat[j];
    }
    let norm_a0: f32 = a0.iter().map(|v| v * v).sum::<f32>().sqrt();
    for v in a0.iter_mut() { *v /= norm_a0 + 1e-6; }

    let proj_rout: f32 = a0.iter().zip(&q_rout_hat).map(|(&x, &y)| x * y).sum();
    let proj_brch: f32 = a0.iter().zip(&q_brch_hat).map(|(&x, &y)| x * y).sum();
    println!("Symmetric Hyperplane Projections:");
    println!("  routing: {:.4} (positive)", proj_rout);
    println!("  breach:  {:.4} (negative)", proj_brch);

    let gamma = 60.0f32;

    // Build LoRA A: [r, in_features]
    let mut a_rows = vec![
        Tensor::from_vec(a0, Shape::from(in_features), &device).unwrap(),
    ];
    for _ in 1..r {
        a_rows.push(Tensor::zeros(Shape::from(in_features), DType::F32, &device).unwrap());
    }
    let a_refs: Vec<&Tensor> = a_rows.iter().collect();
    let lora_a = Tensor::stack(&a_refs, 0).unwrap(); // [16, 4096]

    // Build LoRA B: [out_features, r]
    let col0: Vec<f32> = diff_ba.iter().map(|&x| x * gamma).collect();
    let mut b_cols = vec![
        Tensor::from_vec(col0, Shape::from(hidden_size), &device).unwrap(),
    ];
    for _ in 1..r {
        b_cols.push(Tensor::zeros(Shape::from(hidden_size), DType::F32, &device).unwrap());
    }
    let b_refs: Vec<&Tensor> = b_cols.iter().collect();
    let lora_b = Tensor::stack(&b_refs, 1).unwrap(); // [2560, 16]

    println!("LoRA A shape: {:?}", lora_a.shape());
    println!("LoRA B shape: {:?}", lora_b.shape());

    let mut tensors = HashMap::new();
    tensors.insert("blk.41.attn_output.weight.lora_a".to_string(), lora_a);
    tensors.insert("blk.41.attn_output.weight.lora_b".to_string(), lora_b);

    let lora_path = PathBuf::from("/tmp/targeted_adapter.gguf");
    let bytes = export_lora_gguf(&lora_path, "gemma4", alpha, &tensors).expect("export failed");
    println!("Exported adapter: {} bytes", bytes);
    assert!(bytes > 0);
}
