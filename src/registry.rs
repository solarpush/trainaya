use anyhow::{Context, Result, bail};
use colored::Colorize;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::utils::{format_size, print_banner, print_status};

#[derive(Debug, Clone)]
pub struct RegisterArgs {
    /// Path to model file (model.safetensors or model.gguf) or directory
    pub model_path: PathBuf,
    /// Target Ollaya model name (e.g. 'laya:custom', 'winnow:custom', 'shieldstral:v1')
    pub name: String,
    /// Base template to inherit non-weight layers from (e.g. 'laya:en', 'winnow:e4b')
    pub base_template: Option<String>,
    /// Optional path to custom tokenizer.json (for SafeTensors models)
    pub tokenizer_path: Option<PathBuf>,
    /// Custom path to Ollaya models directory (default: ~/.ollaya/models or $OLLAYA_MODELS)
    pub models_dir: Option<PathBuf>,
}

/// Compute SHA256 hex digest and size of a file
pub fn hash_file(path: &Path) -> Result<(String, u64)> {
    let mut file = fs::File::open(path)
        .with_context(|| format!("Failed to open file for hashing: {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    let mut total_bytes = 0u64;

    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
        total_bytes += n as u64;
    }

    let hash_bytes = hasher.finalize();
    let hex_digest = hash_bytes
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect::<String>();

    Ok((hex_digest, total_bytes))
}

fn store_blob(file_path: &Path, blobs_dir: &Path) -> Result<(String, u64)> {
    let (hash, size) = hash_file(file_path)?;
    let blob_name = format!("sha256-{}", hash);
    let target = blobs_dir.join(&blob_name);
    if !target.exists() {
        if fs::hard_link(file_path, &target).is_err() {
            fs::copy(file_path, &target)
                .with_context(|| format!("Failed to copy blob to {}", target.display()))?;
        }
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(&target, fs::Permissions::from_mode(0o644));
    }
    Ok((hash, size))
}

fn is_directory_writable(dir: &Path) -> bool {
    let mut current = dir;
    while !current.exists() {
        match current.parent() {
            Some(p) => current = p,
            None => return false,
        }
    }
    let test_file = current.join(format!(".trainaya_wtest_{}", std::process::id()));
    match fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&test_file)
    {
        Ok(_) => {
            let _ = fs::remove_file(test_file);
            true
        }
        Err(_) => false,
    }
}

pub fn get_candidate_models_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(env_path) = std::env::var("OLLAYA_MODELS") {
        let trimmed = env_path.trim();
        if !trimmed.is_empty() {
            dirs.push(PathBuf::from(trimmed));
        }
    }
    dirs.push(PathBuf::from("/usr/share/ollaya/.ollaya/models"));
    if let Ok(home) = std::env::var("HOME") {
        dirs.push(PathBuf::from(home).join(".ollaya").join("models"));
    }
    dirs.retain(|d| d.exists());
    dirs.dedup();
    dirs
}

pub fn get_ollaya_models_dir(custom_path: Option<&Path>) -> Result<PathBuf> {
    if let Some(p) = custom_path {
        return Ok(p.to_path_buf());
    }

    if let Ok(env_path) = std::env::var("OLLAYA_MODELS") {
        let trimmed = env_path.trim();
        if !trimmed.is_empty() {
            return Ok(PathBuf::from(trimmed));
        }
    }

    let system_dir = PathBuf::from("/usr/share/ollaya/.ollaya/models");
    if system_dir.exists() && is_directory_writable(&system_dir) {
        return Ok(system_dir);
    }

    if let Ok(home) = std::env::var("HOME") {
        let user_dir = PathBuf::from(&home).join(".ollaya").join("models");
        if user_dir.exists() {
            return Ok(user_dir);
        }
    }

    if system_dir.exists() {
        return Ok(system_dir);
    }

    let home = std::env::var("HOME").context("Environment variable $HOME is not set")?;
    Ok(PathBuf::from(home).join(".ollaya").join("models"))
}

pub fn parse_model_name(name_str: &str) -> (String, String) {
    let trimmed = name_str.trim();
    if let Some((model, tag)) = trimmed.split_once(':') {
        (model.to_string(), tag.to_string())
    } else {
        (trimmed.to_string(), "latest".to_string())
    }
}

pub fn resolve_dynamic_tag(tag_spec: &str, models_dir: &Path) -> String {
    let (model, tag) = parse_model_name(tag_spec);
    let resolved_tag = if tag.contains("{auto}") || tag == "auto" {
        let manifest_dir = models_dir.join("manifests/ollaya.dev/library").join(&model);
        let mut v = 1;
        loop {
            let cand = if tag == "auto" {
                format!("v{}", v)
            } else {
                tag.replace("{auto}", &v.to_string())
            };
            if !manifest_dir.join(&cand).exists() {
                break cand;
            }
            v += 1;
        }
    } else if tag.contains("{timestamp}") {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        tag.replace("{timestamp}", &ts.to_string())
    } else {
        tag
    };

    format!("{}:{}", model, resolved_tag)
}

pub fn run_register(args: RegisterArgs) -> Result<()> {
    print_banner("Ollaya Model Registration");

    let models_dir = get_ollaya_models_dir(args.models_dir.as_deref())?;
    let blobs_dir = models_dir.join("blobs");
    let manifests_base = models_dir.join("manifests").join("ollaya.dev").join("library");

    fs::create_dir_all(&blobs_dir)
        .with_context(|| format!("Failed to create blobs directory: {}", blobs_dir.display()))?;
    fs::create_dir_all(&manifests_base).with_context(|| {
        format!(
            "Failed to create manifests directory: {}",
            manifests_base.display()
        )
    })?;

    let (weight_file, detected_tokenizer, is_gguf) = resolve_files(&args.model_path)?;
    let src_dir = if args.model_path.is_dir() {
        args.model_path.clone()
    } else {
        weight_file
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf()
    };

    let tokenizer_file = args.tokenizer_path.or(detected_tokenizer);

    print_status("Source Weights", weight_file.display().to_string());
    print_status(
        "Model Format",
        if is_gguf {
            "GGUF v3 (Decoder / Logits)"
        } else {
            "SafeTensors / ONNX (Encoder)"
        },
    );

    println!("  Computing SHA-256 digest of weights...");
    let (weight_hash, weight_size) = store_blob(&weight_file, &blobs_dir)?;
    print_status("Weight Digest", format!("sha256:{}", weight_hash));
    print_status("Weight Size", format_size(weight_size));

    let mut tok_info: Option<(String, u64)> = None;
    if let Some(tok_path) = &tokenizer_file {
        if tok_path.exists() {
            let info = store_blob(tok_path, &blobs_dir)?;
            print_status("Tokenizer Digest", format!("sha256:{}", info.0));
            tok_info = Some(info);
        }
    }

    let cal_path = src_dir.join("calibration.json");
    let cal_info = if cal_path.exists() {
        let info = store_blob(&cal_path, &blobs_dir)?;
        print_status("Calibration Blob", format!("sha256:{}", info.0));
        Some(info)
    } else {
        None
    };

    let dec_path = src_dir.join("decision.json");
    let dec_info = if dec_path.exists() {
        let info = store_blob(&dec_path, &blobs_dir)?;
        print_status("Decision Blob", format!("sha256:{}", info.0));
        Some(info)
    } else {
        None
    };

    // Check for PyTorch-patched ONNX graphs (model_fp32.onnx and model_fp16.onnx)
    let onnx_fp32_path = src_dir.join("model_fp32.onnx");
    let onnx_fp32_info = if onnx_fp32_path.exists() {
        let info = store_blob(&onnx_fp32_path, &blobs_dir)?;
        print_status("ONNX FP32 Graph", format!("sha256:{}", info.0));
        Some(info)
    } else {
        None
    };

    let onnx_fp16_path = src_dir.join("model_fp16.onnx");
    let onnx_fp16_info = if onnx_fp16_path.exists() {
        let info = store_blob(&onnx_fp16_path, &blobs_dir)?;
        print_status("ONNX FP16 Graph", format!("sha256:{}", info.0));
        Some(info)
    } else {
        None
    };

    // Resolve template manifest if available
    let default_template = if is_gguf {
        "winnow:e4b"
    } else {
        "laya:multilingual"
    };
    let raw_template = args.base_template.as_deref().unwrap_or(default_template);
    let template_name = if raw_template.eq_ignore_ascii_case("laya:multimodal") {
        "laya:multilingual"
    } else {
        raw_template
    };
    let (tmpl_model, tmpl_tag) = parse_model_name(template_name);

    let mut template_manifest_path = manifests_base.join(&tmpl_model).join(&tmpl_tag);
    if !template_manifest_path.exists() {
        for cand in get_candidate_models_dirs() {
            let cand_path = cand
                .join("manifests/ollaya.dev/library")
                .join(&tmpl_model)
                .join(&tmpl_tag);
            if cand_path.exists() {
                template_manifest_path = cand_path;
                break;
            }
        }
    }

    let mut manifest_val: serde_json::Value = if template_manifest_path.exists() {
        print_status("Base Template", template_name);
        let template_str = fs::read_to_string(&template_manifest_path)?;
        serde_json::from_str(&template_str)?
    } else if dec_info.is_some() {
        // Standalone adapted model (e.g. from `trainaya adapt`)
        let cfg_path = src_dir.join("ollaya_config.json");
        let (cfg_hash, cfg_size) = if cfg_path.exists() {
            store_blob(&cfg_path, &blobs_dir)?
        } else {
            let fallback_cfg = serde_json::json!({
                "model_format": if is_gguf { "gguf" } else { "onnx" },
                "family": if is_gguf { "llm-logits" } else { "laya" },
                "context_length": 8192,
                "languages": ["multilingual"],
            });
            let tmp_cfg = src_dir.join("ollaya_config.json");
            fs::write(&tmp_cfg, serde_json::to_string_pretty(&fallback_cfg)?)?;
            store_blob(&tmp_cfg, &blobs_dir)?
        };
        serde_json::json!({
            "schemaVersion": 2,
            "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
            "config": {
                "mediaType": "application/vnd.ollaya.config.v1+json",
                "digest": format!("sha256:{}", cfg_hash),
                "size": cfg_size,
            },
            "layers": [
                {
                    "mediaType": if is_gguf { "application/vnd.ollaya.weights.gguf" } else { "application/vnd.ollaya.weights" },
                    "digest": format!("sha256:{}", weight_hash),
                    "size": weight_size
                },
                {
                    "mediaType": "application/vnd.ollaya.decision",
                    "digest": format!("sha256:{}", dec_info.as_ref().unwrap().0),
                    "size": dec_info.as_ref().unwrap().1
                }
            ]
        })
    } else {
        bail!(
            "Base template manifest not found for '{}'. Looked in {}.",
            template_name,
            manifests_base.display()
        );
    };

    let layers = manifest_val
        .get_mut("layers")
        .and_then(|l| l.as_array_mut())
        .context("Template manifest missing 'layers' array")?;

    let is_adapter = weight_file
        .file_name()
        .is_some_and(|f| f == "adapter.gguf" || f == "adapter_model.safetensors");

    if is_adapter && is_gguf {
        let mut adapter_layer_found = false;
        for layer in layers.iter_mut() {
            let mtype = layer.get("mediaType").and_then(|m| m.as_str()).unwrap_or("");
            if mtype == "application/vnd.ollaya.adapter.gguf" {
                layer["digest"] = serde_json::Value::String(format!("sha256:{}", weight_hash));
                layer["size"] = serde_json::Value::Number(weight_size.into());
                adapter_layer_found = true;
                break;
            }
        }
        if !adapter_layer_found {
            layers.push(serde_json::json!({
                "mediaType": "application/vnd.ollaya.adapter.gguf",
                "digest": format!("sha256:{}", weight_hash),
                "size": weight_size,
                "annotations": {
                    "org.ollaya.creator": "trainaya"
                }
            }));
        }
    } else {
        let target_media_type = if is_gguf {
            "application/vnd.ollaya.weights.gguf"
        } else {
            "application/vnd.ollaya.weights"
        };

        let mut weight_layer_updated = false;
        for layer in layers.iter_mut() {
            let mtype = layer.get("mediaType").and_then(|m| m.as_str()).unwrap_or("");
            if mtype == target_media_type || (mtype.contains("weights") && !weight_layer_updated) {
                layer["digest"] = serde_json::Value::String(format!("sha256:{}", weight_hash));
                layer["size"] = serde_json::Value::Number(weight_size.into());
                if let Some(obj) = layer.as_object_mut() {
                    obj.remove("urls");
                    let mut annotations = serde_json::Map::new();
                    annotations.insert(
                        "org.ollaya.creator".to_string(),
                        serde_json::Value::String("trainaya".to_string()),
                    );
                    obj.insert("annotations".to_string(), serde_json::Value::Object(annotations));
                }
                weight_layer_updated = true;
                break;
            }
        }

        if !weight_layer_updated {
            bail!("Could not find a matching weights layer in template manifest");
        }

        // Update ONNX graph layers (`fp32` and `fp16`) if patched ONNX graphs were generated by PyTorch engine!
        for layer in layers.iter_mut() {
            let mtype = layer.get("mediaType").and_then(|m| m.as_str()).unwrap_or("");
            if mtype == "application/vnd.ollaya.graph.onnx" {
                let prec = layer
                    .get("annotations")
                    .and_then(|a| a.get("org.ollaya.precision"))
                    .and_then(|p| p.as_str())
                    .unwrap_or("fp32");
                let replacement = if prec == "fp16" {
                    onnx_fp16_info.as_ref().or(onnx_fp32_info.as_ref())
                } else {
                    onnx_fp32_info.as_ref()
                };
                if let Some((onnx_hash, onnx_size)) = replacement {
                    layer["digest"] = serde_json::Value::String(format!("sha256:{}", onnx_hash));
                    layer["size"] = serde_json::Value::Number((*onnx_size).into());
                    if let Some(obj) = layer.as_object_mut() {
                        obj.remove("urls");
                    }
                }
            }
        }
    }

    if let Some((tok_hash, tok_size)) = tok_info {
        for layer in layers.iter_mut() {
            let mtype = layer.get("mediaType").and_then(|m| m.as_str()).unwrap_or("");
            if mtype == "application/vnd.ollaya.tokenizer" {
                layer["digest"] = serde_json::Value::String(format!("sha256:{}", tok_hash));
                layer["size"] = serde_json::Value::Number(tok_size.into());
                if let Some(obj) = layer.as_object_mut() {
                    obj.remove("urls");
                }
                break;
            }
        }
    }

    if let Some((dec_hash, dec_size)) = dec_info {
        for layer in layers.iter_mut() {
            let mtype = layer.get("mediaType").and_then(|m| m.as_str()).unwrap_or("");
            if mtype == "application/vnd.ollaya.decision" {
                layer["digest"] = serde_json::Value::String(format!("sha256:{}", dec_hash));
                layer["size"] = serde_json::Value::Number(dec_size.into());
                if let Some(obj) = layer.as_object_mut() {
                    obj.remove("urls");
                }
                break;
            }
        }
    }

    if let Some((cal_hash, cal_size)) = cal_info {
        let mut cal_layer_found = false;
        for layer in layers.iter_mut() {
            let mtype = layer.get("mediaType").and_then(|m| m.as_str()).unwrap_or("");
            if mtype == "application/vnd.ollaya.calibration" {
                layer["digest"] = serde_json::Value::String(format!("sha256:{}", cal_hash));
                layer["size"] = serde_json::Value::Number(cal_size.into());
                if let Some(obj) = layer.as_object_mut() {
                    obj.remove("urls");
                    let mut annotations = serde_json::Map::new();
                    annotations.insert(
                        "org.ollaya.creator".to_string(),
                        serde_json::Value::String("trainaya".to_string()),
                    );
                    obj.insert("annotations".to_string(), serde_json::Value::Object(annotations));
                }
                cal_layer_found = true;
                break;
            }
        }
        if !cal_layer_found {
            layers.push(serde_json::json!({
                "mediaType": "application/vnd.ollaya.calibration",
                "digest": format!("sha256:{}", cal_hash),
                "size": cal_size,
                "annotations": {
                    "org.ollaya.creator": "trainaya"
                }
            }));
        }
    }

    let (target_model, target_tag) = parse_model_name(&args.name);
    let target_manifest_dir = manifests_base.join(&target_model);
    fs::create_dir_all(&target_manifest_dir)?;

    let target_manifest_path = target_manifest_dir.join(&target_tag);
    if target_manifest_path.exists() {
        println!(
            "  ℹ️  [NOTICE]: Ollaya tag '{}:{}' already exists and will be updated.",
            target_model, target_tag
        );
    }
    let formatted_manifest = serde_json::to_string_pretty(&manifest_val)?;
    fs::write(&target_manifest_path, formatted_manifest)
        .with_context(|| format!("Failed to write manifest to {}", target_manifest_path.display()))?;

    print_status(
        "Registered Model",
        format!("{}:{}", target_model, target_tag).green().bold().to_string(),
    );
    print_status("Manifest Path", target_manifest_path.display().to_string());

    println!("\n{} Model successfully registered in Ollaya runtime!", "✓".green().bold());
    Ok(())
}

fn resolve_files(path: &Path) -> Result<(PathBuf, Option<PathBuf>, bool)> {
    if path.is_file() {
        let is_gguf = path.extension().is_some_and(|e| e == "gguf");
        let parent = path.parent();
        let tok = parent.map(|p| p.join("tokenizer.json")).filter(|p| p.exists());
        return Ok((path.to_path_buf(), tok, is_gguf));
    }

    if path.is_dir() {
        let model_gguf = path.join("model.gguf");
        if model_gguf.exists() {
            return Ok((model_gguf, None, true));
        }

        let safetensors = path.join("model.safetensors");
        if safetensors.exists() {
            let tok = path.join("tokenizer.json");
            return Ok((safetensors, if tok.exists() { Some(tok) } else { None }, false));
        }

        let adapter_gguf = path.join("adapter.gguf");
        if adapter_gguf.exists() {
            return Ok((adapter_gguf, None, true));
        }

        if let Ok(entries) = fs::read_dir(path) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_file() && p.extension().is_some_and(|e| e == "gguf") {
                    return Ok((p, None, true));
                }
            }
        }

        if let Ok(entries) = fs::read_dir(path) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_file() && p.extension().is_some_and(|e| e == "safetensors") {
                    let tok = path.join("tokenizer.json");
                    return Ok((p, if tok.exists() { Some(tok) } else { None }, false));
                }
            }
        }
    }

    bail!("Could not find model weights (.safetensors or .gguf) at '{}'", path.display())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_parse_model_name() {
        assert_eq!(
            parse_model_name("laya:custom"),
            ("laya".to_string(), "custom".to_string())
        );
        assert_eq!(
            parse_model_name("my-model"),
            ("my-model".to_string(), "latest".to_string())
        );
        assert_eq!(
            parse_model_name("winnow:e4b"),
            ("winnow".to_string(), "e4b".to_string())
        );
    }

    #[test]
    fn test_hash_file() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        fs::write(&file_path, b"hello trainaya").unwrap();

        let (hash, size) = hash_file(&file_path).unwrap();
        assert_eq!(size, 14);
        assert_eq!(hash, "991a2c21955bf24592629813dcb080259dfb62b75a417468f17abd92e0fb499a");
    }

    #[test]
    fn test_register_flow_mock() {
        let dir = tempdir().unwrap();
        let root = dir.path();

        let manifest_dir = root.join("manifests/ollaya.dev/library/laya");
        fs::create_dir_all(&manifest_dir).unwrap();
        let mock_template = serde_json::json!({
            "schemaVersion": 2,
            "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
            "layers": [
                {
                    "mediaType": "application/vnd.ollaya.weights",
                    "digest": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
                    "size": 100
                },
                {
                    "mediaType": "application/vnd.ollaya.decision",
                    "digest": "sha256:1111111111111111111111111111111111111111111111111111111111111111",
                    "size": 50
                }
            ]
        });
        fs::write(manifest_dir.join("en"), serde_json::to_string(&mock_template).unwrap()).unwrap();

        let model_file = root.join("model.safetensors");
        fs::write(&model_file, b"MOCK_WEIGHTS_CONTENT").unwrap();

        let args = RegisterArgs {
            model_path: model_file,
            name: "laya:test-custom".to_string(),
            base_template: Some("laya:en".to_string()),
            tokenizer_path: None,
            models_dir: Some(root.to_path_buf()),
        };
        run_register(args).unwrap();

        let (expected_hash, expected_size) = hash_file(&root.join("model.safetensors")).unwrap();
        let blob_path = root.join("blobs").join(format!("sha256-{}", expected_hash));
        assert!(blob_path.exists());

        let registered_manifest_path = root.join("manifests/ollaya.dev/library/laya/test-custom");
        assert!(registered_manifest_path.exists());

        let reg_content: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(registered_manifest_path).unwrap()).unwrap();
        let layers = reg_content["layers"].as_array().unwrap();
        let weights_layer = &layers[0];
        assert_eq!(weights_layer["mediaType"], "application/vnd.ollaya.weights");
        assert_eq!(weights_layer["digest"], format!("sha256:{}", expected_hash));
        assert_eq!(weights_layer["size"], expected_size);
    }
}
