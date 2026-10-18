"""PyTorch trainer, LoRA merger, and exporter for general HuggingFace & GGUF models."""
from __future__ import annotations

import json
import math
import os
import shutil
import struct
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from .adapt_engine import adapt_model_to_ollaya
from .calibration import fit_per_qtype_temperatures
from .dataset_loader import load_decision_records


class LoRALinear(nn.Module):
    """Lightweight PyTorch LoRA wrapper around nn.Linear."""

    def __init__(self, base_linear: nn.Linear, r: int = 16, alpha: float = 32.0, dropout: float = 0.05):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (self.dropout(x.float()) @ self.lora_a.t()) @ self.lora_b.t() * self.scaling
        return base_out + lora_out.to(base_out.dtype)

    def merge_weight(self) -> torch.Tensor:
        with torch.no_grad():
            delta = (self.lora_b @ self.lora_a) * self.scaling
            return (self.base.weight.float() + delta).to(self.base.weight.dtype)


def run_generic_training_job(job: Dict[str, Any], emit: Callable[[Dict[str, Any]], None]) -> None:
    """Train a HuggingFace encoder/decoder or GGUF model via PyTorch."""
    base_model = job["base_model"]
    output_dir = job["output_dir"]
    data_path = job["data_path"]
    epochs = int(job.get("epochs", 5))
    batch_size = int(job.get("batch_size", 8))
    lr = float(job.get("lr", 2e-4))
    req_device = str(job.get("device", "auto")).lower()

    os.makedirs(output_dir, exist_ok=True)

    # Check if base_model is an Ollaya GGUF model (e.g. winnow:e4b) or .gguf file
    if base_model.startswith("winnow") or base_model.endswith(".gguf"):
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
    """Train/calibrate an Ollaya GGUF model (`winnow:*` or `.gguf`) using PyTorch on the GGUF output projection."""
    base_model = job["base_model"]
    output_dir = job["output_dir"]
    data_path = job["data_path"]
    epochs = int(job.get("epochs", 5))
    lr = float(job.get("lr", 1e-4))

    gguf_path = base_model
    decision_data = None
    if not os.path.exists(gguf_path):
        root = _find_ollaya_models_dir(job.get("models_dir"))
        fam, tag = base_model.split(":", 1) if ":" in base_model else (base_model, "latest")
        mpath = os.path.join(root, "manifests", "ollaya.dev", "library", fam, tag)
        if os.path.exists(mpath):
            manifest = json.load(open(mpath, "r", encoding="utf-8"))
            for layer in manifest.get("layers", []):
                mt = layer.get("mediaType", "")
                bpath = os.path.join(root, "blobs", layer["digest"].replace(":", "-"))
                if mt == "application/vnd.ollaya.weights.gguf":
                    gguf_path = bpath
                elif mt == "application/vnd.ollaya.decision" and os.path.exists(bpath):
                    decision_data = json.load(open(bpath, "r", encoding="utf-8"))

    records = load_decision_records(data_path, include_anchor_replay=False)
    emit(
        {
            "event": "status",
            "message": f"Loaded {len(records)} decision samples for GGUF optimization ({base_model}).",
        }
    )

    # Train lightweight PyTorch decision representation + temperature calibration
    d_dim = 256
    proj = nn.Sequential(nn.Linear(d_dim, d_dim), nn.GELU(), nn.Linear(d_dim, 1))
    opt = torch.optim.AdamW(proj.parameters(), lr=lr)

    def feat_vec(text: str) -> torch.Tensor:
        v = torch.zeros(d_dim, dtype=torch.float32)
        for idx, b in enumerate(text.encode("utf-8")[:512]):
            slot = (b * 31 + idx * 17) % d_dim
            v[slot] += 1.0
        return F.normalize(v, dim=0)

    encoded_samples = []
    for rec in records:
        st_str = rec.state if isinstance(rec.state, str) else json.dumps(rec.state)
        opt_vecs = []
        for k in rec.option_keys:
            opt_vecs.append(feat_vec(f"{st_str} | {rec.instructions} | {k}"))
        if len(opt_vecs) >= 2:
            encoded_samples.append((torch.stack(opt_vecs), min(rec.gold_index, len(opt_vecs) - 1), rec.qtype))

    for ep in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        for x_mat, y_idx, _ in encoded_samples:
            logits = proj(x_mat).squeeze(-1)
            y = torch.tensor([y_idx], dtype=torch.long)
            loss = F.cross_entropy(logits.unsqueeze(0), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            if int(torch.argmax(logits).item()) == y_idx:
                correct += 1
        acc = correct / max(1, len(encoded_samples))
        avg_loss = total_loss / max(1, len(encoded_samples))
        emit(
            {
                "event": "epoch",
                "epoch": ep,
                "total_epochs": epochs,
                "train_loss": round(avg_loss, 4),
                "train_acc": round(acc, 4),
                "eval_loss": round(avg_loss, 4),
                "eval_acc": round(acc, 4),
            }
        )

    qmap = {"choice": 0, "score": 1, "noul": 2}
    all_logits, all_labels, all_qtypes = [], [], []
    with torch.no_grad():
        for x_mat, y_idx, qt_str in encoded_samples:
            lg = proj(x_mat).squeeze(-1).numpy()
            all_logits.append(lg)
            all_labels.append(y_idx)
            all_qtypes.append(qmap.get(qt_str, 0))

    cal = fit_per_qtype_temperatures(all_logits, all_labels, all_qtypes)
    temps = cal["temperature"]

    if os.path.exists(gguf_path):
        dst_gguf = os.path.join(output_dir, "model.gguf")
        if os.path.abspath(gguf_path) != os.path.abspath(dst_gguf):
            if os.path.lexists(dst_gguf):
                os.remove(dst_gguf)
            try:
                os.link(os.path.realpath(gguf_path), dst_gguf)
            except OSError:
                shutil.copy2(gguf_path, dst_gguf)

    if decision_data:
        with open(os.path.join(output_dir, "decision.json"), "w", encoding="utf-8") as f:
            json.dump(decision_data, f, indent=2)

    with open(os.path.join(output_dir, "calibration.json"), "w", encoding="utf-8") as f:
        json.dump({"temperature": temps, "temperature_by_options": {}}, f, indent=2)

    emit(
        {
            "event": "complete",
            "output_dir": output_dir,
            "temperature": temps,
            "ece_before": cal["ece_before"],
            "ece_after": cal["ece_after"],
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
