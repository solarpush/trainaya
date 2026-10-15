"""End-to-End PyTorch Trainer for Laya (`laya-markers-v1`) and Bidirectional Encoder Decision Models."""
from __future__ import annotations

import copy
import json
import math
import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from laya.common import QTYPES, DecisionModel, build_sequence, collate_items, proper_reward
from safetensors.torch import load_file, save_file
from transformers import ModernBertConfig, ModernBertModel, PreTrainedTokenizerFast

from .calibration import compute_ece, fit_per_qtype_temperatures
from .dataset_loader import NormalizedDecisionRecord, load_decision_records
from .onnx_patch import patch_laya_onnx_graph, sha256_file


def _find_ollaya_models_dir(custom_dir: Optional[str] = None) -> str:
    if custom_dir and os.path.isdir(custom_dir):
        return custom_dir
    env_dir = os.environ.get("OLLAYA_MODELS")
    if env_dir and os.path.isdir(env_dir):
        return env_dir
    sys_dir = "/usr/share/ollaya/.ollaya/models"
    if os.path.isdir(sys_dir):
        return sys_dir
    return os.path.expanduser("~/.ollaya/models")


def resolve_ollaya_laya_manifest(base_model: str, models_dir: Optional[str] = None) -> Dict[str, Any]:
    """Resolve blob paths from an Ollaya manifest (e.g. `laya:multilingual`, `laya:en`)."""
    root = _find_ollaya_models_dir(models_dir)
    if ":" in base_model:
        fam, tag = base_model.split(":", 1)
    else:
        fam, tag = base_model, "latest"

    manifest_path = os.path.join(root, "manifests", "ollaya.dev", "library", fam, tag)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Ollaya manifest not found at {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    def blob_path(digest: str) -> str:
        return os.path.join(root, "blobs", digest.replace(":", "-"))

    info: Dict[str, Any] = {
        "manifest_path": manifest_path,
        "manifest": manifest,
        "models_dir": root,
        "onnx_layers": [],
    }
    for layer in manifest.get("layers", []):
        mt = layer.get("mediaType", "")
        bpath = blob_path(layer["digest"])
        if mt == "application/vnd.ollaya.weights":
            info["weights_path"] = bpath
            info["weights_digest"] = layer["digest"]
        elif mt == "application/vnd.ollaya.tokenizer":
            info["tokenizer_path"] = bpath
        elif mt == "application/vnd.ollaya.decision":
            info["decision_path"] = bpath
        elif mt == "application/vnd.ollaya.calibration":
            info["calibration_path"] = bpath
        elif mt == "application/vnd.ollaya.arch":
            info["arch_path"] = bpath
        elif mt == "application/vnd.ollaya.graph.onnx":
            prec = (layer.get("annotations") or {}).get("org.ollaya.precision", "fp32")
            info["onnx_layers"].append({"precision": prec, "path": bpath, "digest": layer["digest"]})

    for req in ("weights_path", "tokenizer_path", "decision_path", "arch_path"):
        if req not in info or not os.path.exists(info[req]):
            raise FileNotFoundError(f"Missing required Laya blob '{req}' in {manifest_path}")

    return info


def load_laya_from_ollaya_blobs(
    manifest_info: Dict[str, Any],
    device: str = "cpu",
) -> Tuple[DecisionModel, PreTrainedTokenizerFast, Dict[str, Any], Dict[str, torch.dtype]]:
    """Construct DecisionModel and tokenizer offline from Ollaya blobs."""
    with open(manifest_info["arch_path"], "r", encoding="utf-8") as f:
        arch = json.load(f)
    with open(manifest_info["decision_path"], "r", encoding="utf-8") as f:
        decision = json.load(f)

    b = arch["backbone"]
    rope = b.get("rope_theta", {})
    cfg = ModernBertConfig(
        vocab_size=b["vocab_size"],
        hidden_size=b["hidden_size"],
        intermediate_size=b["intermediate_size"],
        num_hidden_layers=b["num_hidden_layers"],
        num_attention_heads=b["num_attention_heads"],
        norm_eps=b.get("norm_eps", 1e-5),
        norm_bias=b.get("norm_bias", False),
        attention_bias=b.get("attention_bias", False),
        mlp_bias=b.get("mlp_bias", False),
        hidden_activation=b.get("hidden_activation", "gelu"),
        local_attention=b.get("local_attention", 128),
        global_rope_theta=float(rope.get("full_attention", 160000.0)),
        local_rope_theta=float(rope.get("sliding_attention", 10000.0)),
        attn_implementation="sdpa",
    )
    enc = ModernBertModel(cfg)
    head_layers = arch.get("head", {}).get("layers", 2)
    model = DecisionModel(enc, head_layers=head_layers, n_act=2)

    raw_sd = load_file(manifest_info["weights_path"])
    orig_dtypes = {k: v.dtype for k, v in raw_sd.items()}
    fp32_sd = {k: v.float() if v.is_floating_point() else v for k, v in raw_sd.items()}
    model.load_state_dict(fp32_sd, strict=True)
    model.to(device)

    sp = decision["special_tokens"]
    tok = PreTrainedTokenizerFast(
        tokenizer_file=manifest_info["tokenizer_path"],
        mask_token=sp.get("mask_text", "<mask>"),
    )
    tok.cls_token_id = int(sp["cls"])
    tok.sep_token_id = int(sp["sep"])
    tok.mask_token_id = int(sp["mask"])
    tok.pad_token_id = int(sp["pad"])

    return model, tok, decision, orig_dtypes


def _encode_records_for_laya(
    tok: PreTrainedTokenizerFast,
    records: List[NormalizedDecisionRecord],
    max_len: int,
    head_max_len: int,
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for rec in records:
        q = rec.to_laya_q()
        seq, markers = build_sequence(tok, rec.state, q, max_len=max_len, head_max_len=head_max_len)
        k = len(markers)
        if k < 2 or rec.gold_index >= k:
            continue
        target = [0.0] * k
        target[rec.gold_index] = 1.0
        items.append(
            {
                "qid": rec.id,
                "ids": seq,
                "markers": markers,
                "qtype": QTYPES.get(rec.qtype, 0),
                "label": rec.gold_index,
                "target": target,
                "weight": rec.sample_weight,
            }
        )
    return items


def _forward_head_from_cached_encoder(
    model: DecisionModel,
    h_enc: torch.Tensor,
    attention_mask: torch.Tensor,
    marker_pos: torch.Tensor,
    marker_mask: torch.Tensor,
    qtype: torch.Tensor,
) -> torch.Tensor:
    """Run type_emb + 2-layer full-sequence Transformer head + scorer from cached encoder hidden states."""
    h = h_enc + model.type_emb(qtype)[:, None, :]
    if model.head is not None:
        pad = ~attention_mask.bool()
        for layer in model.head.layers:
            h = layer(h, src_key_padding_mask=pad)
    idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
    m = torch.gather(h, 1, idx)
    logits = model.scorer(m).squeeze(-1).float()
    logits = logits.masked_fill(~marker_mask, -1e4)
    return logits


def train_laya_module(
    model: DecisionModel,
    tok: PreTrainedTokenizerFast,
    data_path: str,
    epochs: int = 12,
    batch_size: int = 8,
    lr: float = 1.5e-4,
    unfreeze_layers: int = 0,
    max_len: int = 1024,
    head_max_len: int = 256,
    dev: str = "cpu",
    seed: int = 42,
    emit: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[List[float], float, float]:
    """Train Laya DecisionModel end-to-end in PyTorch with L2-SP anchoring and proper scoring loss."""
    if emit is None:
        emit = lambda d: print(json.dumps(d), flush=True)

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    records = load_decision_records(data_path, include_anchor_replay=True)
    items = _encode_records_for_laya(tok, records, max_len=max_len, head_max_len=head_max_len)
    if not items:
        raise ValueError(f"No valid decision items encoded from {data_path}")

    emit(
        {
            "event": "status",
            "message": f"Encoded {len(items)} decision sequences (including anchor replay) with max_len={max_len}.",
        }
    )

    # Configure trainable parameters and L2-SP reference snapshot
    for p in model.parameters():
        p.requires_grad = False

    head_params: List[nn.Parameter] = []
    for submod in (model.type_emb, model.head, model.scorer):
        if submod is not None:
            for p in submod.parameters():
                p.requires_grad = True
                head_params.append(p)

    backbone_params: List[nn.Parameter] = []
    enc_layers = getattr(model.encoder, "layers", None)
    if unfreeze_layers > 0 and enc_layers is not None:
        n_total = len(enc_layers)
        n_unfreeze = min(unfreeze_layers, n_total)
        for layer in enc_layers[n_total - n_unfreeze :]:
            for p in layer.parameters():
                p.requires_grad = True
                backbone_params.append(p)
        if hasattr(model.encoder, "final_norm"):
            for p in model.encoder.final_norm.parameters():
                p.requires_grad = True
                backbone_params.append(p)

    trainable_params = head_params + backbone_params
    # Save initial weights snapshot for L2-SP regularization (prevents catastrophic forgetting)
    init_snapshot = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    # Group items into length-sorted mini-batches for minimal padding
    sorted_items = sorted(items, key=lambda x: len(x["ids"]))
    batches: List[Dict[str, Any]] = []
    for i in range(0, len(sorted_items), batch_size):
        chunk = sorted_items[i : i + batch_size]
        col = collate_items([chunk], tok.pad_token_id)
        col["weights"] = torch.tensor([it["weight"] for it in chunk], dtype=torch.float32)
        batches.append(col)

    # If encoder is frozen (unfreeze_layers == 0), precompute encoder hidden states once!
    cached_h_enc: List[torch.Tensor] = []
    if unfreeze_layers == 0:
        cache_path = f"{data_path}.n{len(items)}.b{batch_size}.enc_cache.pt"
        loaded_from_cache = False
        if os.path.exists(cache_path) and os.path.getmtime(cache_path) >= os.path.getmtime(data_path):
            try:
                cached_data = torch.load(cache_path, map_location="cpu", weights_only=True)
                if isinstance(cached_data, list) and len(cached_data) == len(batches):
                    cached_h_enc = cached_data
                    loaded_from_cache = True
                    emit(
                        {
                            "event": "status",
                            "message": f"Loaded pre-extracted ModernBERT representations for {len(batches)} batches from {os.path.basename(cache_path)}.",
                        }
                    )
            except Exception:
                loaded_from_cache = False

        if not loaded_from_cache:
            emit(
                {
                    "event": "status",
                    "message": f"Pre-extracting ModernBERT contextual representations across {len(batches)} batches...",
                }
            )
            model.eval()
            with torch.no_grad():
                for b_idx, b in enumerate(batches):
                    ids = b["input_ids"].to(dev)
                    att = b["attention_mask"].to(dev)
                    h = model.encoder(input_ids=ids, attention_mask=att).last_hidden_state.cpu()
                    cached_h_enc.append(h)
                    if (b_idx + 1) % max(1, len(batches) // 5) == 0 or (b_idx + 1) == len(batches):
                        emit(
                            {
                                "event": "status",
                                "message": f"  Cached encoder batch {b_idx + 1}/{len(batches)}",
                            }
                        )
            try:
                torch.save(cached_h_enc, cache_path)
            except Exception:
                pass

    # Measure baseline accuracy & logits before training
    model.eval()
    base_correct = 0
    base_total = 0
    with torch.no_grad():
        for b_idx, b in enumerate(batches):
            att = b["attention_mask"].to(dev)
            mpos = b["marker_pos"].to(dev)
            mmask = b["marker_mask"].to(dev)
            qt = b["qtype"].to(dev)
            labels = b["label"].to(dev)
            if unfreeze_layers == 0:
                h_enc = cached_h_enc[b_idx].to(dev)
                logits = _forward_head_from_cached_encoder(model, h_enc, att, mpos, mmask, qt)
            else:
                ids = b["input_ids"].to(dev)
                logits, _ = model(ids, att, mpos, mmask, qt)
            preds = torch.argmax(logits, dim=-1)
            base_correct += int((preds == labels).sum().item())
            base_total += int(labels.numel())

    base_acc = base_correct / max(1, base_total)
    emit(
        {
            "event": "status",
            "message": f"Initial pre-training accuracy on combined failure+anchor set: {base_acc * 100:.2f}% ({base_correct}/{base_total})",
        }
    )

    param_groups = [{"params": head_params, "lr": lr, "weight_decay": 0.01}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr * 0.15, "weight_decay": 0.01})

    optimizer = torch.optim.AdamW(param_groups)
    total_steps = max(1, epochs * len(batches))
    warmup_steps = max(1, int(0.1 * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_acc = base_acc
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    l2_sp_coeff = 0.0005

    step_idx = 0
    for ep in range(1, epochs + 1):
        model.train()
        order = list(range(len(batches)))
        random.shuffle(order)

        ep_loss = 0.0
        ep_correct = 0
        ep_count = 0

        for b_idx in order:
            b = batches[b_idx]
            att = b["attention_mask"].to(dev)
            mpos = b["marker_pos"].to(dev)
            mmask = b["marker_mask"].to(dev)
            qt = b["qtype"].to(dev)
            labels = b["label"].to(dev)
            weights = b["weights"].to(dev)

            if unfreeze_layers == 0:
                h_enc = cached_h_enc[b_idx].to(dev)
                logits = _forward_head_from_cached_encoder(model, h_enc, att, mpos, mmask, qt)
            else:
                ids = b["input_ids"].to(dev)
                logits, _ = model(ids, att, mpos, mmask, qt)

            # 1. Weighted Cross-Entropy Loss (no label_smoothing on -1e4 masked padding columns!)
            ce_per_item = F.cross_entropy(logits, labels, reduction="none")
            ce_loss = (ce_per_item * weights).mean()

            # 2. Pairwise Margin Loss (pushes gold option >= hardest negative + 1.0)
            row_idx = torch.arange(logits.size(0), device=dev)
            gold_scores = logits[row_idx, labels]
            neg_logits = logits.clone()
            neg_logits[row_idx, labels] = -1e4
            hardest_neg = neg_logits.max(dim=-1).values
            margin_loss = (F.relu(1.0 - (gold_scores - hardest_neg)) * weights).mean()

            # 3. Gentle L2-SP Anchor Penalty towards pretrained weights
            l2_sp = torch.tensor(0.0, device=dev)
            for name, param in model.named_parameters():
                if param.requires_grad and name in init_snapshot:
                    l2_sp = l2_sp + torch.sum((param - init_snapshot[name].to(dev)) ** 2)

            loss = 0.85 * ce_loss + 0.15 * margin_loss + l2_sp_coeff * l2_sp

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()
            step_idx += 1

            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1)
                ep_correct += int((preds == labels).sum().item())
                ep_count += int(labels.numel())
                ep_loss += float(loss.item()) * int(labels.numel())

        train_acc = ep_correct / max(1, ep_count)
        train_loss = ep_loss / max(1, ep_count)

        # Evaluate in eval() mode (no dropout)
        model.eval()
        eval_correct = 0
        eval_loss_sum = 0.0
        with torch.no_grad():
            for b_idx, b in enumerate(batches):
                att = b["attention_mask"].to(dev)
                mpos = b["marker_pos"].to(dev)
                mmask = b["marker_mask"].to(dev)
                qt = b["qtype"].to(dev)
                labels = b["label"].to(dev)
                if unfreeze_layers == 0:
                    h_enc = cached_h_enc[b_idx].to(dev)
                    logits = _forward_head_from_cached_encoder(model, h_enc, att, mpos, mmask, qt)
                else:
                    ids = b["input_ids"].to(dev)
                    logits, _ = model(ids, att, mpos, mmask, qt)
                ce = F.cross_entropy(logits, labels, reduction="sum")
                eval_loss_sum += float(ce.item())
                preds = torch.argmax(logits, dim=-1)
                eval_correct += int((preds == labels).sum().item())

        eval_acc = eval_correct / max(1, ep_count)
        eval_loss = eval_loss_sum / max(1, ep_count)

        if eval_acc >= best_acc:
            best_acc = eval_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        emit(
            {
                "event": "epoch",
                "epoch": ep,
                "total_epochs": epochs,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 4),
                "eval_loss": round(eval_loss, 4),
                "eval_acc": round(eval_acc, 4),
            }
        )

    # Restore best checkpoint
    model.load_state_dict({k: v.to(dev) for k, v in best_state.items()}, strict=True)
    model.eval()

    # Collect eval logits per item for temperature calibration
    all_logits: List[np.ndarray] = []
    all_labels: List[int] = []
    all_qtypes: List[int] = []
    with torch.no_grad():
        for b_idx, b in enumerate(batches):
            att = b["attention_mask"].to(dev)
            mpos = b["marker_pos"].to(dev)
            mmask = b["marker_mask"].to(dev)
            qt = b["qtype"].to(dev)
            labels = b["label"].to(dev)
            if unfreeze_layers == 0:
                h_enc = cached_h_enc[b_idx].to(dev)
                logits = _forward_head_from_cached_encoder(model, h_enc, att, mpos, mmask, qt)
            else:
                ids = b["input_ids"].to(dev)
                logits, _ = model(ids, att, mpos, mmask, qt)
            logits_np = logits.cpu().numpy()
            mmask_np = mmask.cpu().numpy()
            labels_np = labels.cpu().numpy()
            qt_np = qt.cpu().numpy()
            for r_i in range(logits_np.shape[0]):
                k_valid = int(mmask_np[r_i].sum())
                all_logits.append(logits_np[r_i, :k_valid])
                all_labels.append(int(labels_np[r_i]))
                all_qtypes.append(int(qt_np[r_i]))

    cal = fit_per_qtype_temperatures(all_logits, all_labels, all_qtypes)
    return cal["temperature"], float(cal["ece_before"]), float(cal["ece_after"])


def run_laya_training_job(job: Dict[str, Any], emit: Callable[[Dict[str, Any]], None]) -> None:
    """Execute a complete Laya training job from Rust CLI config and export Ollaya artifacts."""
    base_model = job["base_model"]
    output_dir = job["output_dir"]
    data_path = job["data_path"]
    epochs = int(job.get("epochs", 12))
    batch_size = int(job.get("batch_size", 8))
    lr = float(job.get("lr", 1.5e-4))
    unfreeze_layers = int(job.get("unfreeze_layers", 0))
    req_device = str(job.get("device", "auto")).lower()
    models_dir = job.get("models_dir")

    dev = (
        "cuda"
        if (req_device in ("auto", "cuda", "rocm", "hip") and torch.cuda.is_available())
        else "cpu"
    )
    if dev == "cuda":
        gpu_label = torch.cuda.get_device_name(0)
        hip_v = getattr(torch.version, "hip", None)
        dev_desc = f"ROCm/HIP {hip_v} ({gpu_label})" if hip_v else f"CUDA ({gpu_label})"
    else:
        dev_desc = "CPU"
    os.makedirs(output_dir, exist_ok=True)

    manifest_info = resolve_ollaya_laya_manifest(base_model, models_dir=models_dir)
    emit(
        {
            "event": "status",
            "message": f"Loading '{base_model}' into PyTorch DecisionModel on {dev_desc}...",
        }
    )
    model, tok, decision, orig_dtypes = load_laya_from_ollaya_blobs(manifest_info, device=dev)

    old_scorer3_weight_t = (
        model.state_dict()["scorer.3.weight"].detach().cpu().float().numpy().T.copy()
    )

    max_len = int(decision.get("max_len", 1024))
    head_max_len = int(decision.get("head_max_len", 256))

    temps, ece_before, ece_after = train_laya_module(
        model=model,
        tok=tok,
        data_path=data_path,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        unfreeze_layers=unfreeze_layers,
        max_len=max_len,
        head_max_len=head_max_len,
        dev=dev,
        seed=int(job.get("seed", 42)),
        emit=emit,
    )

    new_scorer3_weight_t = (
        model.state_dict()["scorer.3.weight"].detach().cpu().float().numpy().T.copy()
    )

    # Update temperature buffer inside model state_dict as well
    with torch.no_grad():
        if hasattr(model, "temperature"):
            model.temperature.copy_(torch.tensor(temps, dtype=model.temperature.dtype, device=model.temperature.device))

    # Save model.safetensors preserving original tensor dtypes (FP16) and ensure 0644 permissions for ollaya service
    out_safetensors = os.path.join(output_dir, "model.safetensors")
    out_sd: Dict[str, torch.Tensor] = {}
    for k, v in model.state_dict().items():
        target_dt = orig_dtypes.get(k, torch.float16)
        out_sd[k] = v.detach().cpu().to(dtype=target_dt).contiguous()
    save_file(out_sd, out_safetensors)
    try:
        os.chmod(out_safetensors, 0o644)
    except OSError:
        pass

    new_weights_sha = sha256_file(out_safetensors)
    new_weights_blob_name = f"sha256-{new_weights_sha}"

    # Patch both fp32 and fp16 weightless ONNX graphs so ONNX Runtime reads the new weights & inline scorer.3.weight!
    patched_onnx: List[Dict[str, str]] = []
    for onnx_entry in manifest_info.get("onnx_layers", []):
        prec = onnx_entry["precision"]
        src_onnx = onnx_entry["path"]
        dst_onnx = os.path.join(output_dir, f"model_{prec}.onnx")
        stats = patch_laya_onnx_graph(
            src_onnx_path=src_onnx,
            dst_onnx_path=dst_onnx,
            new_safetensors_path=out_safetensors,
            new_weights_location=new_weights_blob_name,
            old_scorer3_weight_t=old_scorer3_weight_t,
            new_scorer3_weight_t=new_scorer3_weight_t,
        )
        try:
            os.chmod(dst_onnx, 0o644)
        except OSError:
            pass
        patched_onnx.append({"precision": prec, "path": dst_onnx})
        emit(
            {
                "event": "status",
                "message": f"Patched {prec} ONNX graph ({stats['external_updated']} external refs -> {new_weights_blob_name[:19]}..., {stats['inline_updated']} inline scorer tensors).",
            }
        )

    # Write calibration.json and decision.json
    calibration = {
        "temperature": temps,
        "temperature_by_options": {},
    }
    with open(os.path.join(output_dir, "calibration.json"), "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)
    with open(os.path.join(output_dir, "decision.json"), "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2)

    emit(
        {
            "event": "complete",
            "output_dir": output_dir,
            "weights_sha256": new_weights_sha,
            "patched_onnx": patched_onnx,
            "temperature": temps,
            "ece_before": ece_before,
            "ece_after": ece_after,
        }
    )
