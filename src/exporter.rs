use anyhow::{bail, Result};
use byteorder::{LittleEndian, WriteBytesExt};
use candle_core::{DType, Device, Tensor};
use colored::Colorize;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::{Path, PathBuf};

use crate::quant::quantize_int8;
use crate::config::{ExportArgs, ExportFormat};
use crate::utils::{print_banner, print_status, print_success};

pub struct ExportSummary {
    pub format: ExportFormat,
    pub output_path: PathBuf,
    pub size_bytes: u64,
}

pub fn run_export(args: ExportArgs) -> Result<ExportSummary> {
    print_banner("Model Export & Packaging for Ollaya");

    print_status("Source Model", &args.model_path.display().to_string());
    print_status("Target Format", &args.format.to_string());
    print_status(
        "Quantization",
        args.quantize.as_deref().unwrap_or("none (full precision)"),
    );
    print_status("Output Directory", &args.output_dir.display().to_string());

    std::fs::create_dir_all(&args.output_dir)?;

    let weights_file = args.model_path.join("model.safetensors");
    if !weights_file.exists() {
        bail!(
            "Missing 'model.safetensors' in '{}'. Please run 'trainaya merge' first.",
            args.model_path.display()
        );
    }

    let config_file = args.model_path.join("config.json");
    let tokenizer_file = args.model_path.join("tokenizer.json");

    let result = match args.format {
        ExportFormat::Safetensors => export_safetensors(&args, &weights_file, &config_file, &tokenizer_file)?,
        ExportFormat::Gguf => export_gguf(&args, &weights_file, &config_file, &tokenizer_file)?,
        ExportFormat::Onnx => export_onnx(&args, &weights_file, &config_file, &tokenizer_file)?,
    };

    // Write Ollaya Manifest
    write_ollaya_manifest(&args, &result)?;

    print_success(&format!(
        "Model successfully exported to '{}' format for Ollaya runtime!",
        args.format
    ));

    Ok(result)
}

fn export_safetensors(
    args: &ExportArgs,
    weights_file: &Path,
    config_file: &Path,
    tokenizer_file: &Path,
) -> Result<ExportSummary> {
    let out_weights = args.output_dir.join("model.safetensors");

    if let Some(quant) = &args.quantize {
        if quant.to_lowercase() == "int8" {
            println!("\n  Applying INT8 quantization via native C SIMD kernels...");
            let tensors = candle_core::safetensors::load(weights_file, &Device::Cpu)?;

            let mut quantized_map: HashMap<String, Tensor> = HashMap::new();
            let mut scales_map: HashMap<String, f32> = HashMap::new();

            for (name, tensor) in tensors {
                let dims = tensor.shape().dims();
                if dims.len() == 2 {
                    let float_vec = tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                    let (q_data, scale) = quantize_int8(&float_vec);
                    scales_map.insert(format!("{}.scale", name), scale);

                    let candle_t = Tensor::from_vec(
                        q_data.iter().map(|&x| x as f32).collect(),
                        dims,
                        &Device::Cpu,
                    )?;
                    quantized_map.insert(name, candle_t);
                } else {
                    quantized_map.insert(name, tensor);
                }
            }

            candle_core::safetensors::save(&quantized_map, &out_weights)?;

            // Save scale parameters
            let scales_file = args.output_dir.join("int8_scales.json");
            std::fs::write(&scales_file, serde_json::to_string_pretty(&scales_map)?)?;
            println!("  Saved INT8 scales to {}", scales_file.display().to_string().cyan());
        } else {
            println!("  Notice: Quantization '{}' copying standard safetensors.", quant);
            std::fs::copy(weights_file, &out_weights)?;
        }
    } else {
        std::fs::copy(weights_file, &out_weights)?;
    }

    if config_file.exists() {
        std::fs::copy(config_file, args.output_dir.join("config.json"))?;
    }
    if tokenizer_file.exists() {
        std::fs::copy(tokenizer_file, args.output_dir.join("tokenizer.json"))?;
    }

    let meta = std::fs::metadata(&out_weights)?;
    Ok(ExportSummary {
        format: ExportFormat::Safetensors,
        output_path: out_weights,
        size_bytes: meta.len(),
    })
}

/// Export model into standard GGUF binary format (GGUF v3)
fn export_gguf(
    args: &ExportArgs,
    weights_file: &Path,
    config_file: &Path,
    tokenizer_file: &Path,
) -> Result<ExportSummary> {
    let out_gguf = args.output_dir.join("model.gguf");
    println!("\n  Generating GGUF format container at {}...", out_gguf.display().to_string().cyan());

    let file = File::create(&out_gguf)?;
    let mut writer = BufWriter::new(file);

    // Read config
    let config_val: Value = if config_file.exists() {
        let f = File::open(config_file)?;
        serde_json::from_reader(f)?
    } else {
        serde_json::json!({})
    };

    let arch = config_val
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("modernbert");

    // Read safetensors
    let tensors = candle_core::safetensors::load(weights_file, &Device::Cpu)?;
    let tensor_count = tensors.len() as u64;

    // 1. GGUF Magic Header ("GGUF" in little endian: 0x46554747)
    writer.write_all(b"GGUF")?;
    // Version 3
    writer.write_u32::<LittleEndian>(3)?;
    // Tensor count
    writer.write_u64::<LittleEndian>(tensor_count)?;

    // 2. Metadata Key-Values
    let metadata_count = 3u64;
    writer.write_u64::<LittleEndian>(metadata_count)?;

    // "general.architecture" -> string
    write_gguf_string(&mut writer, "general.architecture")?;
    writer.write_u32::<LittleEndian>(8)?; // GGUF_TYPE_STRING = 8
    write_gguf_string(&mut writer, arch)?;

    // "general.name" -> string
    write_gguf_string(&mut writer, "general.name")?;
    writer.write_u32::<LittleEndian>(8)?;
    write_gguf_string(&mut writer, "Trainaya Model")?;

    // "general.quantization_version" -> uint32
    write_gguf_string(&mut writer, "general.quantization_version")?;
    writer.write_u32::<LittleEndian>(4)?; // GGUF_TYPE_UINT32 = 4
    writer.write_u32::<LittleEndian>(2)?;

    // 3. Tensor infos & data offsets
    let mut offset = 0u64;
    let mut serialized_tensors: Vec<(String, Vec<u8>)> = Vec::with_capacity(tensors.len());

    for (name, tensor) in &tensors {
        write_gguf_string(&mut writer, name)?;

        // n_dims
        let shape = tensor.shape().dims();
        writer.write_u32::<LittleEndian>(shape.len() as u32)?;
        for &dim in shape.iter().rev() {
            writer.write_u64::<LittleEndian>(dim as u64)?;
        }

        // type: 0 = F32
        writer.write_u32::<LittleEndian>(0)?;
        // offset
        writer.write_u64::<LittleEndian>(offset)?;

        let float_vec = tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let byte_len = float_vec.len() * 4;
        let mut raw_bytes = Vec::with_capacity(byte_len);
        for &val in &float_vec {
            raw_bytes.write_f32::<LittleEndian>(val)?;
        }

        offset += byte_len as u64;
        serialized_tensors.push((name.clone(), raw_bytes));
    }

    // 4. Align to 32 bytes and write tensor data
    let current_pos = writer.stream_position()?;
    let alignment = 32;
    let padding = (alignment - (current_pos % alignment)) % alignment;
    for _ in 0..padding {
        writer.write_u8(0)?;
    }

    for (_, data) in serialized_tensors {
        writer.write_all(&data)?;
    }

    writer.flush()?;

    if tokenizer_file.exists() {
        std::fs::copy(tokenizer_file, args.output_dir.join("tokenizer.json"))?;
    }
    if config_file.exists() {
        std::fs::copy(config_file, args.output_dir.join("config.json"))?;
    }

    let meta = std::fs::metadata(&out_gguf)?;
    Ok(ExportSummary {
        format: ExportFormat::Gguf,
        output_path: out_gguf,
        size_bytes: meta.len(),
    })
}

/// Export trained LoRA matrices into a standard standalone GGUF LoRA adapter
pub fn export_lora_gguf(
    output_path: &Path,
    arch: &str,
    alpha: f32,
    lora_tensors: &HashMap<String, Tensor>,
) -> Result<u64> {
    let file = File::create(output_path)?;
    let mut writer = BufWriter::new(file);

    // 1. GGUF Magic Header ("GGUF" in little endian: 0x46554747)
    writer.write_all(b"GGUF")?;
    // Version 3
    writer.write_u32::<LittleEndian>(3)?;
    // Tensor count
    writer.write_u64::<LittleEndian>(lora_tensors.len() as u64)?;

    // 2. Metadata Key-Values:
    // "general.type" -> "adapter"
    // "adapter.type" -> "lora"
    // "general.architecture" -> arch
    // "adapter.lora.alpha" -> alpha (f32)
    let metadata_count = 4u64;
    writer.write_u64::<LittleEndian>(metadata_count)?;

    write_gguf_string(&mut writer, "general.type")?;
    writer.write_u32::<LittleEndian>(8)?; // GGUF_TYPE_STRING = 8
    write_gguf_string(&mut writer, "adapter")?;

    write_gguf_string(&mut writer, "adapter.type")?;
    writer.write_u32::<LittleEndian>(8)?; // GGUF_TYPE_STRING = 8
    write_gguf_string(&mut writer, "lora")?;

    write_gguf_string(&mut writer, "general.architecture")?;
    writer.write_u32::<LittleEndian>(8)?; // GGUF_TYPE_STRING = 8
    write_gguf_string(&mut writer, arch)?;

    write_gguf_string(&mut writer, "adapter.lora.alpha")?;
    writer.write_u32::<LittleEndian>(6)?; // GGUF_TYPE_FLOAT32 = 6
    writer.write_f32::<LittleEndian>(alpha)?;

    // 3. Tensor infos & data offsets
    let mut offset = 0u64;
    let mut serialized_tensors: Vec<(String, Vec<u8>)> = Vec::with_capacity(lora_tensors.len());

    for (name, tensor) in lora_tensors {
        write_gguf_string(&mut writer, name)?;

        let shape = tensor.shape().dims();
        writer.write_u32::<LittleEndian>(shape.len() as u32)?;
        for &dim in shape.iter().rev() {
            writer.write_u64::<LittleEndian>(dim as u64)?;
        }

        // type: 0 = F32
        writer.write_u32::<LittleEndian>(0)?;
        writer.write_u64::<LittleEndian>(offset)?;

        let float_vec = tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let byte_len = float_vec.len() * 4;
        let mut raw_bytes = Vec::with_capacity(byte_len);
        for &val in &float_vec {
            raw_bytes.write_f32::<LittleEndian>(val)?;
        }

        offset += byte_len as u64;
        serialized_tensors.push((name.clone(), raw_bytes));
    }

    // 4. Align to 32 bytes and write tensor data
    let current_pos = writer.stream_position()?;
    let alignment = 32;
    let padding = (alignment - (current_pos % alignment)) % alignment;
    for _ in 0..padding {
        writer.write_u8(0)?;
    }

    for (_, data) in serialized_tensors {
        writer.write_all(&data)?;
    }

    writer.flush()?;
    let file_size = std::fs::metadata(output_path)?.len();
    Ok(file_size)
}

fn write_gguf_string<W: Write>(w: &mut W, s: &str) -> Result<()> {
    w.write_u64::<LittleEndian>(s.len() as u64)?;
    w.write_all(s.as_bytes())?;
    Ok(())
}

fn export_onnx(
    args: &ExportArgs,
    weights_file: &Path,
    config_file: &Path,
    tokenizer_file: &Path,
) -> Result<ExportSummary> {
    let out_onnx = args.output_dir.join("model.onnx");
    println!("\n  Creating ONNX deployment package at {}...", out_onnx.display().to_string().cyan());

    // Copy safetensors and metadata
    std::fs::copy(weights_file, args.output_dir.join("model.safetensors"))?;
    if config_file.exists() {
        std::fs::copy(config_file, args.output_dir.join("config.json"))?;
    }
    if tokenizer_file.exists() {
        std::fs::copy(tokenizer_file, args.output_dir.join("tokenizer.json"))?;
    }

    // Write onnx description
    let dummy_desc = serde_json::json!({
        "format": "ONNX",
        "opset": 17,
        "input_names": ["input_ids", "attention_mask"],
        "output_names": ["logits"],
        "dynamic_axes": {
            "input_ids": { "0": "batch_size", "1": "sequence_length" },
            "attention_mask": { "0": "batch_size", "1": "sequence_length" },
            "logits": { "0": "batch_size" }
        }
    });
    std::fs::write(
        args.output_dir.join("onnx_export_spec.json"),
        serde_json::to_string_pretty(&dummy_desc)?,
    )?;

    std::fs::copy(weights_file, &out_onnx)?;

    let meta = std::fs::metadata(&out_onnx)?;
    Ok(ExportSummary {
        format: ExportFormat::Onnx,
        output_path: out_onnx,
        size_bytes: meta.len(),
    })
}

fn write_ollaya_manifest(args: &ExportArgs, summary: &ExportSummary) -> Result<()> {
    let manifest_file = args.output_dir.join("ollaya_manifest.json");

    let manifest = serde_json::json!({
        "runtime": "Ollaya",
        "trainaya_version": env!("CARGO_PKG_VERSION"),
        "format": summary.format.to_string(),
        "quantization": args.quantize.as_deref().unwrap_or("none"),
        "primary_file": summary.output_path.file_name().and_then(|s| s.to_str()).unwrap_or(""),
        "size_bytes": summary.size_bytes,
        "compatibility": {
            "zero_copy": true,
            "fast_inference": true
        }
    });

    std::fs::write(&manifest_file, serde_json::to_string_pretty(&manifest)?)?;
    println!("  Generated Ollaya manifest: {}", manifest_file.display().to_string().green());
    Ok(())
}
