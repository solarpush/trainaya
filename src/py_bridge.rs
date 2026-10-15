use anyhow::{Context, Result, bail};
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use serde_json::Value;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const EMBED_PYPROJECT: &str = include_str!("../py_engine/pyproject.toml");
const EMBED_INIT: &str = include_str!("../py_engine/trainaya_py/__init__.py");
const EMBED_MAIN: &str = include_str!("../py_engine/trainaya_py/__main__.py");
const EMBED_CALIBRATION: &str = include_str!("../py_engine/trainaya_py/calibration.py");
const EMBED_ONNX_PATCH: &str = include_str!("../py_engine/trainaya_py/onnx_patch.py");
const EMBED_DATASET_LOADER: &str = include_str!("../py_engine/trainaya_py/dataset_loader.py");
const EMBED_LAYA_TRAINER: &str = include_str!("../py_engine/trainaya_py/laya_trainer.py");
const EMBED_ADAPT_ENGINE: &str = include_str!("../py_engine/trainaya_py/adapt_engine.py");
const EMBED_HF_TRAINER: &str = include_str!("../py_engine/trainaya_py/hf_trainer.py");

/// Ensure the embedded Python package is materialized either in `./py_engine` (dev workspace)
/// or in `~/.cache/trainaya/py_engine/` (standalone binary mode).
pub fn ensure_py_engine_dir() -> Result<PathBuf> {
    let local_engine = PathBuf::from("py_engine");
    if local_engine.join("trainaya_py/__main__.py").exists() {
        return Ok(fs::canonicalize(&local_engine).unwrap_or(local_engine));
    }

    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let cache_engine = PathBuf::from(home).join(".cache/trainaya/py_engine");
    let pkg_dir = cache_engine.join("trainaya_py");
    fs::create_dir_all(&pkg_dir)
        .with_context(|| format!("Failed to create {}", pkg_dir.display()))?;

    let files = [
        (cache_engine.join("pyproject.toml"), EMBED_PYPROJECT),
        (pkg_dir.join("__init__.py"), EMBED_INIT),
        (pkg_dir.join("__main__.py"), EMBED_MAIN),
        (pkg_dir.join("calibration.py"), EMBED_CALIBRATION),
        (pkg_dir.join("onnx_patch.py"), EMBED_ONNX_PATCH),
        (pkg_dir.join("dataset_loader.py"), EMBED_DATASET_LOADER),
        (pkg_dir.join("laya_trainer.py"), EMBED_LAYA_TRAINER),
        (pkg_dir.join("adapt_engine.py"), EMBED_ADAPT_ENGINE),
        (pkg_dir.join("hf_trainer.py"), EMBED_HF_TRAINER),
    ];

    for (path, content) in files {
        let needs_write = match fs::read_to_string(&path) {
            Ok(existing) => existing != content,
            Err(_) => true,
        };
        if needs_write {
            fs::write(&path, content)?;
        }
    }

    Ok(cache_engine)
}

/// Resolve the Python command (`./.venv/bin/python` if present, otherwise `uv run --project <dir> python`).
fn build_python_command(engine_dir: &Path, subcmd: &str, cfg_arg: Option<&Path>) -> Result<Command> {
    let hw = crate::hardware::SystemHardware::detect();
    hw.apply_environment("auto");
    let torch_backend = hw.uv_torch_backend();

    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cwd_venv = cwd.join(".venv/bin/python");
    let engine_venv = engine_dir.join(".venv/bin/python");
    let parent_venv = engine_dir
        .parent()
        .map(|p| p.join(".venv/bin/python"))
        .unwrap_or_default();

    let direct_python = if cwd_venv.exists() {
        Some(cwd_venv)
    } else if engine_venv.exists() {
        Some(engine_venv)
    } else if parent_venv.exists() {
        Some(parent_venv)
    } else {
        None
    };

    if let Some(py_bin) = direct_python {
        let mut cmd = Command::new(py_bin);
        cmd.env("PYTHONPATH", engine_dir);
        for (k, v) in hw.get_environment_for_device("auto") {
            cmd.env(k, v);
        }
        cmd.arg("-m").arg("trainaya_py").arg(subcmd);
        if let Some(cfg) = cfg_arg {
            cmd.arg("--config").arg(cfg);
        }
        return Ok(cmd);
    }

    let mut cmd = Command::new("uv");
    cmd.arg("run")
        .arg("--project")
        .arg(engine_dir)
        .arg("--torch-backend")
        .arg(&torch_backend)
        .arg("python")
        .arg("-m")
        .arg("trainaya_py")
        .arg(subcmd);
    if let Some(cfg) = cfg_arg {
        cmd.arg("--config").arg(cfg);
    }
    cmd.env("PYTHONPATH", engine_dir);
    cmd.env("UV_TORCH_BACKEND", &torch_backend);
    for (k, v) in hw.get_environment_for_device("auto") {
        cmd.env(k, v);
    }
    Ok(cmd)
}

/// Execute a Python engine job (`train`, `adapt`, `merge`) with live JSONL event streaming.
pub fn run_py_job(subcmd: &str, job_config: &Value) -> Result<Value> {
    let engine_dir = ensure_py_engine_dir()?;
    let tmp_dir = std::env::temp_dir();
    let cfg_path = tmp_dir.join(format!("trainaya_job_{}_{}.json", subcmd, std::process::id()));
    fs::write(&cfg_path, serde_json::to_string_pretty(job_config)?)?;

    let mut cmd = build_python_command(&engine_dir, subcmd, Some(&cfg_path))?;
    cmd.stdout(Stdio::piped()).stderr(Stdio::inherit());

    let mut child = cmd
        .spawn()
        .context("Failed to launch Python/uv engine. Ensure `uv` is installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`).")?;

    let stdout = child.stdout.take().context("Failed to capture Python engine stdout")?;
    let reader = BufReader::new(stdout);

    let mut pb: Option<ProgressBar> = None;
    let mut complete_payload: Option<Value> = None;
    let mut error_msg: Option<String> = None;

    for line in reader.lines() {
        let line = line?;
        if let Some(json_str) = line.strip_prefix("TRAINAYA_EVENT:") {
            if let Ok(ev) = serde_json::from_str::<Value>(json_str) {
                let ev_type = ev.get("event").and_then(|v| v.as_str()).unwrap_or("");
                match ev_type {
                    "status" => {
                        if let Some(msg) = ev.get("message").and_then(|v| v.as_str()) {
                            if let Some(ref bar) = pb {
                                bar.println(format!("  {} {}", "•".cyan(), msg));
                            } else {
                                println!("  {} {}", "•".cyan(), msg);
                            }
                        }
                    }
                    "epoch" => {
                        let ep = ev.get("epoch").and_then(|v| v.as_u64()).unwrap_or(1);
                        let total = ev.get("total_epochs").and_then(|v| v.as_u64()).unwrap_or(1);
                        let tr_loss = ev.get("train_loss").and_then(|v| v.as_f64()).unwrap_or(0.0);
                        let tr_acc = ev.get("train_acc").and_then(|v| v.as_f64()).unwrap_or(0.0);
                        let ev_loss = ev.get("eval_loss").and_then(|v| v.as_f64()).unwrap_or(0.0);
                        let ev_acc = ev.get("eval_acc").and_then(|v| v.as_f64()).unwrap_or(0.0);

                        if pb.is_none() {
                            let bar = ProgressBar::new(total);
                            bar.set_style(
                                ProgressStyle::default_bar()
                                    .template("  {spinner:.green} [{elapsed_precise}] [{bar:36.cyan/blue}] {pos}/{len} epochs | {msg}")
                                    .unwrap()
                                    .progress_chars("━>-"),
                            );
                            pb = Some(bar);
                        }
                        if let Some(ref bar) = pb {
                            bar.set_position(ep);
                            bar.set_message(format!(
                                "loss: {:.4} | train_acc: {:.1}% | eval_acc: {:.1}%",
                                tr_loss,
                                tr_acc * 100.0,
                                ev_acc * 100.0
                            ));
                            bar.println(format!(
                                "    Epoch {:>2}/{:<2} — Train Loss: {:.4} ({:>5.1}%) | Eval Loss: {:.4} ({:>5.1}%)",
                                ep,
                                total,
                                tr_loss,
                                tr_acc * 100.0,
                                ev_loss,
                                ev_acc * 100.0
                            ));
                        }
                    }
                    "complete" => {
                        if let Some(bar) = pb.take() {
                            bar.finish_and_clear();
                        }
                        complete_payload = Some(ev);
                    }
                    "error" => {
                        if let Some(bar) = pb.take() {
                            bar.finish_and_clear();
                        }
                        let msg = ev
                            .get("message")
                            .and_then(|v| v.as_str())
                            .unwrap_or("Unknown Python engine error");
                        let tb = ev.get("traceback").and_then(|v| v.as_str()).unwrap_or("");
                        error_msg = Some(format!("{}\n{}", msg, tb));
                    }
                    _ => {}
                }
            }
        } else if !line.trim().is_empty() {
            println!("  {}", line.dimmed());
        }
    }

    let _ = fs::remove_file(&cfg_path);
    let status = child.wait()?;

    if let Some(err) = error_msg {
        bail!("PyTorch Engine Error: {}", err.trim());
    }
    if !status.success() {
        bail!("PyTorch Engine exited with non-zero status: {}", status);
    }

    Ok(complete_payload.unwrap_or_else(|| serde_json::json!({"event": "complete"})))
}

/// Query Python/PyTorch/ONNX environment status for `trainaya doctor`.
pub fn run_py_doctor() -> Result<Value> {
    let engine_dir = ensure_py_engine_dir()?;
    let mut cmd = build_python_command(&engine_dir, "doctor", None)?;
    let out = cmd.output().context("Failed to execute Python doctor check")?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    for line in stdout.lines() {
        if let Some(json_str) = line.strip_prefix("TRAINAYA_EVENT:") {
            if let Ok(val) = serde_json::from_str::<Value>(json_str) {
                return Ok(val);
            }
        }
    }
    bail!("Python doctor did not return a valid status event")
}
