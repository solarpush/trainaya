use std::fs::File;
use std::path::PathBuf;

#[test]
fn test_print_label_tokens() {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/pierre".to_string());
    let path = PathBuf::from(home).join(".ollaya/models/blobs/sha256-840e3f50e5a9c218727f44e121d1b37cc9e2c3b318c8eb422ba6ef2e27b618a2");
    if !path.exists() {
        return;
    }
    let mut file = File::open(&path).expect("Failed to open GGUF");
    let content = candle_core::quantized::gguf_file::Content::read(&mut file).expect("Failed to read GGUF");

    let tokens_val = content.metadata.get("tokenizer.ggml.tokens").expect("Missing tokenizer tokens");
    if let candle_core::quantized::gguf_file::Value::Array(tokens) = tokens_val {
        println!("Token 236776: {:?}", tokens[236776]);
        println!("Token 236799: {:?}", tokens[236799]);
        println!("Token 236780: {:?}", tokens[236780]);
    }
}
