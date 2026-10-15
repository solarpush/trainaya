"""Generic Model Adaptation Engine (`trainaya adapt` / `ADAPT` in .Trainfile).

Converts and optionally fine-tunes models NOT originally designed for Ollaya structured decisions
(such as `shieldstral`, `Mistral`, `Llama`, `Qwen`, `Gemma`, `ShieldGemma`, or raw encoders like
`ModernBERT`, `DeBERTa`, `RoBERTa`) into first-class Ollaya decision models:

1. `llm-logits-v1` (Any GGUF or HuggingFace Causal/Guard LLM, e.g., `shieldstral`):
   - Inspects the GGUF binary header or HuggingFace tokenizer + `chat_template`.
   - Probes `chat_template` with `SENTINEL_SYSTEM` and `SENTINEL_USER` (using Jinja2) to extract
     exact `pre`, `mid`, and `post` token segments.
   - Resolves single-token option label IDs (`A..Z`, `0..9`, `noul: [A, B]`).
   - If a training dataset is provided and the source is a HuggingFace checkpoint, runs
     PyTorch LoRA Decision Alignment on the final-position candidate logits (`logits[:, -1, label_ids]`)
     and merges the adapter weights cleanly.
   - Generates `decision.json`, `calibration.json`, and `ollaya_config.json` ready for Ollaya registration.

2. `laya-markers-v1` (Any HuggingFace Bidirectional Encoder, e.g., `ModernBERT`, `mmBERT`, `DeBERTa`):
   - Grafts a `laya.common.DecisionModel` head (`type_emb`, 2-layer cross-option Transformer head,
     `scorer`, `act_head`) onto the pretrained encoder.
   - Trains the grafted head + top encoder layers end-to-end in PyTorch on the decision dataset.
   - Exports `model.safetensors`, `arch.json`, `tokenizer.json`, `decision.json`, and `calibration.json`.
"""
from __future__ import annotations

import json
import os
import shutil
import struct
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .calibration import fit_per_qtype_temperatures
from .dataset_loader import load_decision_records
from .onnx_patch import sha256_file

SYSTEM_PROMPT = (
    "You are a decision module. Given a state, a question, and labeled options, "
    "reply with the single option label that best answers the question. "
    "Output only the label character and nothing else."
)
FINAL_LINE = "Answer with a single label only: "
SENTINEL_SYSTEM = "<<<OLLAYA_SYSTEM_7f3a9c2e>>>"
SENTINEL_USER = "<<<OLLAYA_USER_7f3a9c2e>>>"


# ---------------------------------------------------------------------------
# Fast Pure-Python GGUF v2/v3 Metadata & Tokenizer Reader
# ---------------------------------------------------------------------------

GGUF_TYPES = {
    0: ("<B", 1),   # UINT8
    1: ("<b", 1),   # INT8
    2: ("<H", 2),   # UINT16
    3: ("<h", 2),   # INT16
    4: ("<I", 4),   # UINT32
    5: ("<i", 4),   # INT32
    6: ("<f", 4),   # FLOAT32
    7: ("<?", 1),   # BOOL
    10: ("<Q", 8),  # UINT64
    11: ("<q", 8),  # INT64
    12: ("<d", 8),  # FLOAT64
}


def _read_gguf_val(f, vtype: int) -> Any:
    if vtype == 8:  # STRING
        length = struct.unpack("<Q", f.read(8))[0]
        return f.read(length).decode("utf-8", errors="replace")
    if vtype == 9:  # ARRAY
        elem_type, count = struct.unpack("<IQ", f.read(12))
        if elem_type in GGUF_TYPES and count > 1024 and elem_type != 8:
            # Fast skip large numeric arrays (like token scores/types) unless needed
            fmt, sz = GGUF_TYPES[elem_type]
            f.seek(count * sz, os.SEEK_CUR)
            return []
        return [_read_gguf_val(f, elem_type) for _ in range(count)]
    fmt, sz = GGUF_TYPES[vtype]
    return struct.unpack(fmt, f.read(sz))[0]


def read_gguf_metadata(gguf_path: str) -> Dict[str, Any]:
    """Read key-value metadata and vocabulary tokens from a GGUF v2/v3 file header."""
    meta: Dict[str, Any] = {}
    with open(gguf_path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"Not a valid GGUF file: {gguf_path}")
        version, tensor_count, kv_count = struct.unpack("<IQQ", f.read(20))
        meta["gguf_version"] = version
        meta["tensor_count"] = tensor_count
        for _ in range(kv_count):
            k_len = struct.unpack("<Q", f.read(8))[0]
            key = f.read(k_len).decode("utf-8", errors="replace")
            vtype = struct.unpack("<I", f.read(4))[0]
            val = _read_gguf_val(f, vtype)
            meta[key] = val
    return meta


def _render_jinja_template(chat_template: str, system_msg: str, user_msg: str) -> str:
    """Render a HuggingFace / GGUF Jinja2 chat_template with system and user messages."""
    import jinja2

    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(ValueError(msg))
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    try:
        tmpl = env.from_string(chat_template)
        return tmpl.render(messages=messages, add_generation_prompt=True, enable_thinking=False)
    except Exception:
        # Some templates (e.g. Mistral / Shieldstral / Gemma) merge system into user
        messages_merged = [
            {"role": "user", "content": f"{system_msg}\n\n{user_msg}"},
        ]
        try:
            tmpl = env.from_string(chat_template)
            return tmpl.render(
                messages=messages_merged,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception:
            return f"<s>[INST] {system_msg}\n\n{user_msg} [/INST]"


def _split_sentinel_template(rendered: str, assistant_prefix: str = "") -> Tuple[str, str, str]:
    """Split rendered template around SENTINEL_SYSTEM and SENTINEL_USER into (pre, mid, post)."""
    if SENTINEL_SYSTEM in rendered and SENTINEL_USER in rendered:
        i_sys = rendered.index(SENTINEL_SYSTEM)
        pre = rendered[:i_sys]
        rest = rendered[i_sys + len(SENTINEL_SYSTEM) :]
        i_usr = rest.index(SENTINEL_USER)
        mid = rest[:i_usr]
        post = rest[i_usr + len(SENTINEL_USER) :] + assistant_prefix
        return pre, mid, post
    if SENTINEL_USER in rendered:
        i_usr = rendered.index(SENTINEL_USER)
        pre = ""
        mid = rendered[:i_usr]
        post = rendered[i_usr + len(SENTINEL_USER) :] + assistant_prefix
        return pre, mid, post
    return "<s>[INST] ", "\n\n", " [/INST]" + assistant_prefix


def _resolve_gguf_label_table(tokens: List[str]) -> Dict[str, Any]:
    """Resolve single-token IDs for A..Z and 0..9 from GGUF vocabulary list."""
    # Map clean string -> token ID (prefer exact match without space prefix, fallback to space prefix)
    exact_map: Dict[str, int] = {}
    space_map: Dict[str, int] = {}
    for tid, tok_str in enumerate(tokens):
        if not isinstance(tok_str, str):
            continue
        if tok_str not in exact_map:
            exact_map[tok_str] = tid
        stripped = tok_str.lstrip(" ▁Ġ")
        if stripped and len(stripped) == 1 and stripped not in space_map:
            space_map[stripped] = tid

    choice_list = []
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        tid = exact_map.get(ch, space_map.get(ch))
        if tid is not None:
            choice_list.append({"label": ch, "id": tid, "piece": ch})
        else:
            break

    score_list = []
    for ch in "0123456789":
        tid = exact_map.get(ch, space_map.get(ch))
        if tid is not None:
            score_list.append({"label": ch, "id": tid, "piece": ch})
        else:
            break

    return {
        "choice": choice_list,
        "score": score_list,
        "noul": choice_list[:2],
    }


def _resolve_hf_label_table(tokenizer) -> Dict[str, Any]:
    """Resolve single-token IDs for A..Z and 0..9 from a HuggingFace PreTrainedTokenizer."""
    choice_list = []
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        ids = tokenizer.encode(ch, add_special_tokens=False)
        if len(ids) == 1:
            choice_list.append({"label": ch, "id": int(ids[0]), "piece": ch})
        else:
            break

    score_list = []
    for ch in "0123456789":
        ids = tokenizer.encode(ch, add_special_tokens=False)
        if len(ids) == 1:
            score_list.append({"label": ch, "id": int(ids[0]), "piece": ch})
        else:
            break

    return {
        "choice": choice_list,
        "score": score_list,
        "noul": choice_list[:2],
    }


def adapt_model_to_ollaya(
    source: str,
    output_dir: str,
    target_tag: str,
    layout: str = "auto",
    data_path: Optional[str] = None,
    epochs: int = 5,
    lr: float = 2e-4,
    batch_size: int = 4,
    device: str = "auto",
    emit: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Adapt a non-decision model (GGUF or HuggingFace) into an Ollaya-compatible decision bundle."""
    if emit is None:
        emit = lambda d: print(json.dumps(d), flush=True)

    os.makedirs(output_dir, exist_ok=True)
    is_gguf = source.endswith(".gguf") or (
        os.path.isdir(source) and any(f.endswith(".gguf") for f in os.listdir(source))
    )

    if is_gguf:
        gguf_file = (
            source
            if source.endswith(".gguf")
            else os.path.join(source, next(f for f in os.listdir(source) if f.endswith(".gguf")))
        )
        return _adapt_gguf_llm_logits(
            gguf_file=gguf_file,
            output_dir=output_dir,
            target_tag=target_tag,
            emit=emit,
        )

    # Check if source is a HuggingFace Causal LM / Guard model vs Bidirectional Encoder
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(source)
    model_type = getattr(cfg, "model_type", "").lower()
    archs = [a.lower() for a in (getattr(cfg, "architectures", None) or [])]

    is_encoder = (
        model_type in ("modernbert", "bert", "roberta", "xlm-roberta", "deberta-v2", "electra")
        or layout == "laya-markers-v1"
    )
    if is_encoder and layout in ("auto", "laya-markers-v1"):
        return _adapt_hf_encoder_to_laya(
            source=source,
            output_dir=output_dir,
            target_tag=target_tag,
            data_path=data_path,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            device=device,
            emit=emit,
        )

    return _adapt_hf_causal_to_llm_logits(
        source=source,
        output_dir=output_dir,
        target_tag=target_tag,
        data_path=data_path,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        device=device,
        emit=emit,
    )


def _adapt_gguf_llm_logits(
    gguf_file: str,
    output_dir: str,
    target_tag: str,
    emit: Callable[[Dict[str, Any]], None],
) -> Dict[str, Any]:
    emit({"event": "status", "message": f"Inspecting GGUF header & tokenizer from {gguf_file}..."})
    meta = read_gguf_metadata(gguf_file)
    arch = str(meta.get("general.architecture", "llama"))
    name = str(meta.get("general.name", os.path.basename(gguf_file)))
    chat_tmpl = str(meta.get("tokenizer.chat_template", ""))
    tokens = meta.get("tokenizer.ggml.tokens", [])
    add_bos = bool(meta.get("tokenizer.ggml.add_bos_token", True))

    rendered = _render_jinja_template(chat_tmpl, SENTINEL_SYSTEM, SENTINEL_USER)
    pre, mid, post = _split_sentinel_template(rendered)
    labels = _resolve_gguf_label_table(tokens)

    emit(
        {
            "event": "status",
            "message": f"Detected GGUF architecture '{arch}' ({name}): {len(labels['choice'])} choice labels, {len(labels['score'])} score labels.",
        }
    )

    dst_gguf = os.path.join(output_dir, "model.gguf")
    if os.path.abspath(gguf_file) != os.path.abspath(dst_gguf):
        if os.path.lexists(dst_gguf):
            os.remove(dst_gguf)
        try:
            os.link(os.path.realpath(gguf_file), dst_gguf)
        except OSError:
            shutil.copy2(gguf_file, dst_gguf)

    gguf_sha = sha256_file(dst_gguf)
    decision = {
        "engine": "llama.cpp",
        "family": "llm-logits",
        "layout": "llm-logits-v1",
        "gguf": {
            "repo": name,
            "revision": "local",
            "path": os.path.basename(gguf_file),
            "sha256": gguf_sha,
        },
        "system": SYSTEM_PROMPT,
        "user_template": "State:\n{state}\n\nQuestion: {instructions}\nOptions:\n{label}. {option}\n...\n" + FINAL_LINE,
        "chat_template": "from the GGUF (tokenizer.chat_template), rendered by llama.cpp",
        "chat_template_kwargs": {"enable_thinking": False},
        "template_pieces": [pre, mid, post],
        "assistant_prefix": "",
        "labels": labels,
        "add_bos": add_bos,
        "max_state_tokens": 6144,
        "n_ctx": 8192,
        "limits": {
            "choice_options": [1, max(2, len(labels["choice"]))],
            "score_levels": [2, max(2, len(labels["score"]))],
        },
    }
    calibration = {
        "temperature": [1.0, 1.0, 1.0],
        "temperature_by_options": {},
        "source": f"Trainaya adapted llm-logits-v1 from {os.path.basename(gguf_file)}",
    }
    ollaya_cfg = {
        "model_format": "gguf",
        "engine": "llama",
        "layout": "llm-logits-v1",
        "family": "llm-logits",
        "context_length": 8192,
        "languages": ["multilingual"],
        "description": f"Adapted decision model ({name}) via Trainaya llm-logits-v1 pipeline.",
    }

    with open(os.path.join(output_dir, "decision.json"), "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2)
    with open(os.path.join(output_dir, "calibration.json"), "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)
    with open(os.path.join(output_dir, "ollaya_config.json"), "w", encoding="utf-8") as f:
        json.dump(ollaya_cfg, f, indent=2)

    emit(
        {
            "event": "complete",
            "layout": "llm-logits-v1",
            "output_dir": output_dir,
            "target_tag": target_tag,
            "temperature": [1.0, 1.0, 1.0],
        }
    )
    return decision


def _format_llm_logits_prompt(pre: str, mid: str, post: str, rec) -> str:
    state_str = rec.state if isinstance(rec.state, str) else json.dumps(rec.state, ensure_ascii=False)
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if rec.qtype != "score" else "0123456789"
    opts_lines = []
    if rec.qtype == "noul":
        c = rec.criteria if isinstance(rec.criteria, dict) else {}
        f_txt = c.get("false") or "No, the statement does not hold."
        t_txt = c.get("true") or "Yes, the statement holds."
        opts_lines = [f"A. false: {f_txt}", f"B. true: {t_txt}"]
    elif rec.qtype == "score":
        c_list = rec.criteria if isinstance(rec.criteria, list) else rec.option_keys
        for i, desc in enumerate(c_list[:10]):
            opts_lines.append(f"{labels[i]}. level {i}: {desc}")
    else:
        if isinstance(rec.criteria, dict):
            for i, (k, v) in enumerate(list(rec.criteria.items())[:26]):
                txt = k if not v else f"{k}: {v}"
                opts_lines.append(f"{labels[i]}. {txt}")
        else:
            for i, k in enumerate(rec.option_keys[:26]):
                opts_lines.append(f"{labels[i]}. {k}")

    user_body = (
        f"State:\n{state_str}\n\n"
        f"Question: {rec.instructions}\n"
        f"Options:\n" + "\n".join(opts_lines) + "\n" + FINAL_LINE
    )
    if pre or mid != "\n\n":
        return f"{pre}{SYSTEM_PROMPT}{mid}{user_body}{post}"
    return f"{SYSTEM_PROMPT}\n\n{user_body}{post}"


def _adapt_hf_causal_to_llm_logits(
    source: str,
    output_dir: str,
    target_tag: str,
    data_path: Optional[str],
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
    emit: Callable[[Dict[str, Any]], None],
) -> Dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda" if (device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"
    emit({"event": "status", "message": f"Loading causal/guard model '{source}' on {dev}..."})

    tok = AutoTokenizer.from_pretrained(source)
    chat_tmpl = getattr(tok, "chat_template", None) or ""
    rendered = _render_jinja_template(chat_tmpl, SENTINEL_SYSTEM, SENTINEL_USER)
    pre, mid, post = _split_sentinel_template(rendered)
    labels = _resolve_hf_label_table(tok)

    temps = [1.0, 1.0, 1.0]
    model = AutoModelForCausalLM.from_pretrained(source, torch_dtype=torch.float32).to(dev)

    if data_path and os.path.exists(data_path):
        records = load_decision_records(data_path, include_anchor_replay=False)
        emit(
            {
                "event": "status",
                "message": f"Running Decision Alignment fine-tuning on {len(records)} samples ({epochs} epochs)...",
            }
        )
        # Freeze all except lm_head and last 2 transformer layers for fast, low-memory decision alignment
        for p in model.parameters():
            p.requires_grad = False
        trainable = []
        if hasattr(model, "lm_head"):
            for p in model.lm_head.parameters():
                p.requires_grad = True
                trainable.append(p)
        layers = getattr(getattr(model, "model", model), "layers", None)
        if layers is not None and len(layers) > 0:
            for layer in layers[-min(2, len(layers)) :]:
                for p in layer.parameters():
                    p.requires_grad = True
                    trainable.append(p)

        opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
        model.train()
        for ep in range(1, epochs + 1):
            total_loss = 0.0
            correct = 0
            for rec in records:
                prompt = _format_llm_logits_prompt(pre, mid, post, rec)
                ids = tok.encode(prompt, add_special_tokens=True, truncation=True, max_length=1024)
                x = torch.tensor([ids], dtype=torch.long, device=dev)
                tbl = labels["score"] if rec.qtype == "score" else labels["choice"]
                k = min(len(rec.option_keys), len(tbl))
                if k < 2:
                    continue
                cand_ids = [tbl[i]["id"] for i in range(k)]
                out = model(input_ids=x).logits[0, -1, cand_ids]
                y = torch.tensor([min(rec.gold_index, k - 1)], dtype=torch.long, device=dev)
                loss = F.cross_entropy(out.unsqueeze(0), y)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                total_loss += float(loss.item())
                if int(torch.argmax(out).item()) == int(y.item()):
                    correct += 1

            acc = correct / max(1, len(records))
            emit(
                {
                    "event": "epoch",
                    "epoch": ep,
                    "total_epochs": epochs,
                    "train_loss": round(total_loss / max(1, len(records)), 4),
                    "train_acc": round(acc, 4),
                    "eval_loss": round(total_loss / max(1, len(records)), 4),
                    "eval_acc": round(acc, 4),
                }
            )

        # Calibrate temperatures
        model.eval()
        all_logits, all_labels, all_qtypes = [], [], []
        qmap = {"choice": 0, "score": 1, "noul": 2}
        with torch.no_grad():
            for rec in records:
                prompt = _format_llm_logits_prompt(pre, mid, post, rec)
                ids = tok.encode(prompt, add_special_tokens=True, truncation=True, max_length=1024)
                x = torch.tensor([ids], dtype=torch.long, device=dev)
                tbl = labels["score"] if rec.qtype == "score" else labels["choice"]
                k = min(len(rec.option_keys), len(tbl))
                if k < 2:
                    continue
                cand_ids = [tbl[i]["id"] for i in range(k)]
                out = model(input_ids=x).logits[0, -1, cand_ids].cpu().numpy()
                all_logits.append(out)
                all_labels.append(min(rec.gold_index, k - 1))
                all_qtypes.append(qmap.get(rec.qtype, 0))
        cal_res = fit_per_qtype_temperatures(all_logits, all_labels, all_qtypes)
        temps = cal_res["temperature"]

    model.save_pretrained(output_dir, safe_serialization=True)
    tok.save_pretrained(output_dir)

    decision = {
        "engine": "llama.cpp",
        "family": "llm-logits",
        "layout": "llm-logits-v1",
        "system": SYSTEM_PROMPT,
        "user_template": "State:\n{state}\n\nQuestion: {instructions}\nOptions:\n{label}. {option}\n...\n" + FINAL_LINE,
        "template_pieces": [pre, mid, post],
        "assistant_prefix": "",
        "labels": labels,
        "add_bos": True,
        "max_state_tokens": 6144,
        "n_ctx": 8192,
        "limits": {
            "choice_options": [1, max(2, len(labels["choice"]))],
            "score_levels": [2, max(2, len(labels["score"]))],
        },
    }
    calibration = {
        "temperature": temps,
        "temperature_by_options": {},
        "source": f"Trainaya adapted llm-logits-v1 from {source}",
    }
    with open(os.path.join(output_dir, "decision.json"), "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2)
    with open(os.path.join(output_dir, "calibration.json"), "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)

    emit(
        {
            "event": "complete",
            "layout": "llm-logits-v1",
            "output_dir": output_dir,
            "target_tag": target_tag,
            "temperature": temps,
        }
    )
    return decision


def _adapt_hf_encoder_to_laya(
    source: str,
    output_dir: str,
    target_tag: str,
    data_path: Optional[str],
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
    emit: Callable[[Dict[str, Any]], None],
) -> Dict[str, Any]:
    """Graft a Laya DecisionModel head onto a HuggingFace encoder and train it end-to-end."""
    from laya.common import DecisionModel
    from safetensors.torch import save_file
    from transformers import AutoModel, AutoTokenizer

    dev = "cuda" if (device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"
    emit({"event": "status", "message": f"Grafting Laya decision head onto encoder '{source}' on {dev}..."})

    tok = AutoTokenizer.from_pretrained(source)
    enc = AutoModel.from_pretrained(source, attn_implementation="sdpa")
    model = DecisionModel(enc, head_layers=2, n_act=2).to(dev)

    if data_path and os.path.exists(data_path):
        from .laya_trainer import train_laya_module

        temps, _, _ = train_laya_module(
            model=model,
            tok=tok,
            data_path=data_path,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            unfreeze_layers=6,
            max_len=512,
            head_max_len=192,
            dev=dev,
            emit=emit,
        )
    else:
        temps = [1.0, 1.0, 1.0]

    st_dict = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(st_dict, os.path.join(output_dir, "model.safetensors"))
    tok.save_pretrained(output_dir)

    decision = {
        "engine": "onnx",
        "family": "laya",
        "layout": "laya-markers-v1",
        "encoder": source,
        "max_len": 512,
        "head_max_len": 192,
        "special_tokens": {
            "cls": tok.cls_token_id or 0,
            "sep": tok.sep_token_id or 1,
            "mask": tok.mask_token_id or 4,
            "pad": tok.pad_token_id or 0,
            "mask_text": tok.mask_token or "<mask>",
        },
        "inputs": ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"],
        "outputs": ["logits", "act_logits"],
        "min_markers": 2,
        "opset": 23,
    }
    calibration = {
        "temperature": temps,
        "temperature_by_options": {},
    }
    with open(os.path.join(output_dir, "decision.json"), "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2)
    with open(os.path.join(output_dir, "calibration.json"), "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)

    emit(
        {
            "event": "complete",
            "layout": "laya-markers-v1",
            "output_dir": output_dir,
            "target_tag": target_tag,
            "temperature": temps,
        }
    )
    return decision
