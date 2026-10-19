"""PyTorch trainer, LoRA merger, and exporter for general HuggingFace & GGUF models."""
from __future__ import annotations

import json
import math
import os
import shutil
import struct
from typing import Any, Callable, Dict, List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore

try:
    from safetensors.torch import load_file, save_file
except Exception:
    load_file = None  # type: ignore
    save_file = None  # type: ignore

from .adapt_engine import adapt_model_to_ollaya
from .calibration import fit_per_qtype_temperatures
from .dataset_loader import load_decision_records


class LoRALinear(nn.Module if nn is not None else object):  # type: ignore
    """Lightweight PyTorch LoRA wrapper around nn.Linear."""

    def __init__(self, base_linear: Any, r: int = 16, alpha: float = 32.0, dropout: float = 0.05):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False
        in_f = base_linear.in_features
        out_f = base_linear.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / max(1, r)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        dev = base_linear.weight.device
        dt = torch.float32
        self.lora_a = nn.Parameter(torch.randn(r, in_f, device=dev, dtype=dt) * (1.0 / math.sqrt(in_f)))
        self.lora_b = nn.Parameter(torch.zeros(out_f, r, device=dev, dtype=dt))

    def forward(self, x: Any) -> Any:
        base_out = self.base(x)
        lora_out = (self.dropout(x.float()) @ self.lora_a.t()) @ self.lora_b.t() * self.scaling
        return base_out + lora_out.to(base_out.dtype)

    def merge_weight(self) -> Any:
        with torch.no_grad():
            delta = (self.lora_b @ self.lora_a) * self.scaling
            return (self.base.weight.float() + delta).to(self.base.weight.dtype)


def _is_gguf_target(base_model: str, models_dir: Optional[str] = None) -> bool:
    if base_model.endswith(".gguf"):
        return True
    if os.path.isdir(base_model) and os.path.exists(os.path.join(base_model, "model.gguf")):
        return True
    tag_safe = base_model.replace(":", "_")
    if os.path.isdir(os.path.join("output", tag_safe)) and os.path.exists(
        os.path.join("output", tag_safe, "model.gguf")
    ):
        return True
    root = _find_ollaya_models_dir(models_dir)
    fam, tag = base_model.split(":", 1) if ":" in base_model else (base_model, "latest")
    mpath = os.path.join(root, "manifests", "ollaya.dev", "library", fam, tag)
    if os.path.exists(mpath):
        try:
            manifest = json.load(open(mpath, "r", encoding="utf-8"))
            for layer in manifest.get("layers", []):
                if layer.get("mediaType") == "application/vnd.ollaya.weights.gguf":
                    return True
        except Exception:
            pass
    return base_model.startswith("winnow") or base_model.startswith("push_")


def run_generic_training_job(job: Dict[str, Any], emit: Callable[[Dict[str, Any]], None]) -> None:
    """Train a HuggingFace encoder/decoder or GGUF model via PyTorch / Multi-Stage GGUF Trainer."""
    base_model = job["base_model"]
    output_dir = job["output_dir"]
    data_path = job.get("data_path") or "auto"
    epochs = int(job.get("epochs", 5))
    batch_size = int(job.get("batch_size", 8))
    lr = float(job.get("lr", 2e-4))
    req_device = str(job.get("device", "auto")).lower()

    os.makedirs(output_dir, exist_ok=True)

    # Check if base_model is an Ollaya GGUF model, adapted GGUF bundle, or .gguf file
    if _is_gguf_target(base_model, job.get("models_dir")):
        _run_gguf_calibration_and_adapter_job(job, emit)
        return

    # Otherwise delegate to HuggingFace encoder/causal adaptation & training
    adapt_model_to_ollaya(
        source=base_model,
        output_dir=output_dir,
        target_tag=job.get("family") or "custom:v1",
        layout=job.get("family") or "auto",
        data_path=data_path,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        device=req_device,
        emit=emit,
    )


def _find_ollaya_models_dir(custom_dir: Optional[str] = None) -> str:
    if custom_dir and os.path.isdir(custom_dir):
        return custom_dir
    for candidate in (
        "/usr/share/ollaya/.ollaya/models",
        os.path.expanduser("~/.ollaya/models"),
    ):
        if os.path.isdir(candidate):
            return candidate
    return os.path.expanduser("~/.ollaya/models")


def _run_gguf_calibration_and_adapter_job(
    job: Dict[str, Any], emit: Callable[[Dict[str, Any]], None]
) -> None:
    """Train and calibrate a GGUF decision model using the full Multi-Stage GGUF Decision Trainer."""
    from pathlib import Path
    from .adapt_engine import SYSTEM_PROMPT, read_gguf_metadata, sha256_file
    from .dataset_loader import build_ollaya_preset_anchors, load_hf_jev_adaptation_corpus
    from .gguf_head_trainer import ensure_untied_output_weight, parse_gguf_layout, train_gguf_decision_head

    base_model = job["base_model"]
    output_dir = job["output_dir"]
    data_path = str(job.get("data_path") or "auto")
    epochs = int(job.get("epochs", 12))
    lr = float(job.get("lr", 4e-4))

    bundle_dir: Optional[str] = None
    if os.path.isdir(base_model) and os.path.exists(os.path.join(base_model, "model.gguf")):
        bundle_dir = base_model
    else:
        tag_safe = base_model.replace(":", "_")
        cand_dir = os.path.join("output", tag_safe)
        if os.path.isdir(cand_dir) and os.path.exists(os.path.join(cand_dir, "model.gguf")):
            bundle_dir = cand_dir

    # If the user did not specify a custom output_dir (it defaulted to ./output_adapter) and we found an existing bundle_dir, train in-place in bundle_dir
    if bundle_dir and os.path.normpath(output_dir) == os.path.normpath("./output_adapter"):
        output_dir = bundle_dir
    os.makedirs(output_dir, exist_ok=True)

    gguf_path = base_model
    decision_data = None
    ollaya_cfg_data = None

    if bundle_dir:
        gguf_path = os.path.join(bundle_dir, "model.gguf")
        dec_file = os.path.join(bundle_dir, "decision.json")
        if os.path.exists(dec_file):
            decision_data = json.load(open(dec_file, "r", encoding="utf-8"))
        cfg_file = os.path.join(bundle_dir, "ollaya_config.json")
        if os.path.exists(cfg_file):
            ollaya_cfg_data = json.load(open(cfg_file, "r", encoding="utf-8"))
        # Copy cache/backup files if training into a different output_dir
        if os.path.abspath(bundle_dir) != os.path.abspath(output_dir):
            for aux in ("hidden_states_cache.npz", "orig_head_backup.npz", "ollaya_config.json"):
                src_aux = os.path.join(bundle_dir, aux)
                dst_aux = os.path.join(output_dir, aux)
                if os.path.exists(src_aux) and not os.path.exists(dst_aux):
                    shutil.copy2(src_aux, dst_aux)

    if not os.path.exists(gguf_path):
        root = _find_ollaya_models_dir(job.get("models_dir"))
        fam, tag = base_model.split(":", 1) if ":" in base_model else (base_model, "latest")
        mpath = os.path.join(root, "manifests", "ollaya.dev", "library", fam, tag)
        if os.path.exists(mpath):
            manifest = json.load(open(mpath, "r", encoding="utf-8"))
            cfg_digest = manifest.get("config", {}).get("digest", "")
            if cfg_digest:
                cpath = os.path.join(root, "blobs", cfg_digest.replace(":", "-"))
                if os.path.exists(cpath):
                    ollaya_cfg_data = json.load(open(cpath, "r", encoding="utf-8"))
            for layer in manifest.get("layers", []):
                mt = layer.get("mediaType", "")
                bpath = os.path.join(root, "blobs", layer["digest"].replace(":", "-"))
                if mt == "application/vnd.ollaya.weights.gguf":
                    gguf_path = bpath
                elif mt == "application/vnd.ollaya.decision" and os.path.exists(bpath):
                    decision_data = json.load(open(bpath, "r", encoding="utf-8"))

    if not os.path.exists(gguf_path) or not decision_data:
        # Delegate full initial GGUF adaptation + training to adapt_model_to_ollaya
        adapt_model_to_ollaya(
            source=base_model,
            output_dir=output_dir,
            target_tag=job.get("family") or (base_model if ":" in base_model else "custom:v1"),
            layout="llm-logits-v1",
            data_path=data_path,
            epochs=epochs,
            lr=lr,
            batch_size=int(job.get("batch_size", 8)),
            device=str(job.get("device", "auto")),
            emit=emit,
        )
        return

    dst_gguf = os.path.join(output_dir, "model.gguf")
    def _emit_step(stage: str, msg: str) -> None:
        emit({"event": "status", "stage": stage, "message": msg})

    if os.path.abspath(gguf_path) != os.path.abspath(dst_gguf):
        _, tensors_layout = ensure_untied_output_weight(
            Path(os.path.realpath(gguf_path)),
            Path(dst_gguf),
            emit=_emit_step,
        )
    else:
        _, tensors_layout, _ = parse_gguf_layout(Path(dst_gguf))

    meta = read_gguf_metadata(dst_gguf)
    arch = str(meta.get("general.architecture", decision_data.get("gguf", {}).get("architecture", "llama")))
    softcap_val = float(meta.get(f"{arch}.final_logit_softcapping", 0.0) or 0.0)

    if data_path and data_path not in ("auto", "hf:jev", "hf:jev-v3", "hf:open-jev") and os.path.exists(data_path):
        records = build_ollaya_preset_anchors() + load_decision_records(data_path)
    else:
        records = load_hf_jev_adaptation_corpus(max_samples=4000, emit=emit)

    tpl = decision_data.get("template", {})
    chat_template_cfg = {
        "pre": tpl.get("pre", "<|im_start|>system\n"),
        "mid": tpl.get("mid", "<|im_end|>\n<|im_start|>user\n"),
        "post": tpl.get("post", "<|im_end|>\n<|im_start|>assistant\n"),
        "system": decision_data.get("system", SYSTEM_PROMPT),
        "assistant_prefix": decision_data.get("assistant_prefix", ""),
    }
    from .adapt_engine import _resolve_gguf_label_table
    tokens = meta.get("tokenizer.ggml.tokens", [])
    resolved_labels = _resolve_gguf_label_table(tokens) if tokens else {}
    yes_no_ids = resolved_labels.get("_yes_no_ids")

    labels = decision_data["labels"]
    choice_ids = labels["choice"]["ids"] if isinstance(labels.get("choice"), dict) else labels["ids"][:26]
    score_ids = labels["score"]["ids"] if isinstance(labels.get("score"), dict) else labels["ids"][:10]
    noul_ids = labels["noul"]["ids"] if isinstance(labels.get("noul"), dict) else choice_ids[:2]

    train_metrics = train_gguf_decision_head(
        gguf_path=Path(dst_gguf),
        tensor_info=tensors_layout["output.weight"],
        choice_token_ids=choice_ids,
        score_token_ids=score_ids,
        noul_token_ids=noul_ids,
        chat_template=chat_template_cfg,
        records=records,
        epochs=max(180, epochs * 30),
        lr=lr,
        l2_sp_weight=0.008,
        n_threads=12,
        add_bos=bool(decision_data.get("add_bos", True)),
        yes_no_token_ids=yes_no_ids,
        final_logit_softcap=softcap_val,
        emit=_emit_step,
    )

    oof_acc = train_metrics.get("oof_accuracy", train_metrics["post_quant_accuracy"])
    pq_acc = train_metrics["post_quant_accuracy"]
    emit(
        {
            "event": "epoch",
            "epoch": epochs,
            "total_epochs": epochs,
            "train_loss": round(1.0 - float(pq_acc["total"]), 4),
            "train_acc": float(pq_acc["total"]),
            "eval_loss": round(1.0 - float(oof_acc["total"]), 4),
            "eval_acc": float(oof_acc["total"]),
        }
    )

    gguf_sha = sha256_file(dst_gguf)
    if "gguf" in decision_data and isinstance(decision_data["gguf"], dict):
        decision_data["gguf"]["sha256"] = gguf_sha
        decision_data["gguf"]["size"] = os.path.getsize(dst_gguf)

    with open(os.path.join(output_dir, "decision.json"), "w", encoding="utf-8") as f:
        json.dump(decision_data, f, indent=2)

    if ollaya_cfg_data:
        with open(os.path.join(output_dir, "ollaya_config.json"), "w", encoding="utf-8") as f:
            json.dump(ollaya_cfg_data, f, indent=2)

    temps = train_metrics["temperatures"]
    temp_by_opts = train_metrics.get("temperature_by_options", {})
    calibration = {
        "temperature": temps,
        "temperature_by_options": temp_by_opts,
        "temperature_range": train_metrics.get("temperature_range", [0.25, 16.0]),
        "baseline_accuracy": train_metrics["baseline_accuracy"],
        "oof_accuracy": oof_acc,
        "post_quant_accuracy": pq_acc,
        "trained_samples": train_metrics["n_samples"],
        "source": f"Trainaya Multi-Stage GGUF Decision Trainer (55/45 Kernel Ridge + Masked Softmax KL+Brier+RPS AdamW, 5-Fold OOF Calibrated)",
    }
    with open(os.path.join(output_dir, "calibration.json"), "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)

    emit(
        {
            "event": "complete",
            "output_dir": output_dir,
            "temperature": temps,
            "temperature_by_options": temp_by_opts,
            "oof_accuracy": oof_acc,
            "post_quant_accuracy": pq_acc,
        }
    )



def run_merge_job(job: Dict[str, Any], emit: Callable[[Dict[str, Any]], None]) -> None:
    """Merge LoRA adapter safetensors into base model safetensors using PyTorch."""
    base_model = job["base_model"]
    lora_path = job["lora_path"]
    output_dir = job["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    adapter_file = (
        lora_path
        if lora_path.endswith(".safetensors")
        else os.path.join(lora_path, "adapter_model.safetensors")
    )
    cfg_file = os.path.join(os.path.dirname(adapter_file), "adapter_config.json")
    r, alpha = 16, 32.0
    if os.path.exists(cfg_file):
        cfg = json.load(open(cfg_file, "r", encoding="utf-8"))
        r = int(cfg.get("r", 16))
        alpha = float(cfg.get("lora_alpha", 32.0))
    scale = alpha / max(1, r)

    base_st_file = (
        base_model
        if base_model.endswith(".safetensors")
        else os.path.join(base_model, "model.safetensors")
    )
    if not os.path.exists(base_st_file):
        from huggingface_hub import hf_hub_download

        base_st_file = hf_hub_download(repo_id=base_model, filename="model.safetensors")

    base_tensors = load_file(base_st_file)
    adapter_tensors = load_file(adapter_file)

    merged = {}
    merged_count = 0
    for name, w in base_tensors.items():
        a_key = f"{name}.lora_a.weight"
        b_key = f"{name}.lora_b.weight"
        if a_key in adapter_tensors and b_key in adapter_tensors:
            a = adapter_tensors[a_key].float()
            b = adapter_tensors[b_key].float()
            w_new = (w.float() + (b @ a) * scale).to(w.dtype)
            merged[name] = w_new.contiguous()
            merged_count += 1
        else:
            merged[name] = w.contiguous()

    for name, t in adapter_tensors.items():
        if name.startswith("classifier.") or name.startswith("scorer.") or name.startswith("head."):
            if not (name.endswith(".lora_a.weight") or name.endswith(".lora_b.weight")):
                merged[name] = t.contiguous()

    out_file = os.path.join(output_dir, "model.safetensors")
    save_file(merged, out_file)

    if os.path.isdir(base_model):
        for fname in ("config.json", "tokenizer.json", "decision.json", "calibration.json"):
            src = os.path.join(base_model, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(output_dir, fname))

    emit(
        {
            "event": "complete",
            "total_base_tensors": len(merged),
            "merged_layers_count": merged_count,
            "output_file": out_file,
        }
    )
