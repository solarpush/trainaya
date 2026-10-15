"""CLI entry point for Trainaya's embedded uv + PyTorch engine (`python -m trainaya_py`)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Any, Dict


def emit_event(payload: Dict[str, Any]) -> None:
    print("TRAINAYA_EVENT:" + json.dumps(payload, ensure_ascii=False), flush=True)


def cmd_doctor() -> int:
    import onnx
    import safetensors
    import torch
    import transformers

    try:
        import laya  # noqa: F401
        has_laya = True
    except Exception:
        has_laya = False

    cuda_avail = torch.cuda.is_available()
    cuda_dev = torch.cuda.get_device_name(0) if cuda_avail else None
    vram_mb = (
        int(torch.cuda.get_device_properties(0).total_memory // (1024 * 1024))
        if cuda_avail
        else 0
    )
    hip_ver = getattr(torch.version, "hip", None)
    mps_avail = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    emit_event(
        {
            "event": "doctor",
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "onnx_version": onnx.__version__,
            "safetensors_version": safetensors.__version__,
            "laya_available": has_laya,
            "cuda_available": cuda_avail,
            "cuda_device": cuda_dev,
            "vram_mb": vram_mb,
            "hip_version": hip_ver,
            "mps_available": mps_avail,
        }
    )
    return 0


def _load_job_config(cfg_arg: str) -> Dict[str, Any]:
    if os.path.exists(cfg_arg):
        with open(cfg_arg, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(cfg_arg)


def cmd_train(cfg_arg: str) -> int:
    job = _load_job_config(cfg_arg)
    base_model = str(job.get("base_model", ""))
    family = str(job.get("family") or "")

    from .laya_trainer import resolve_ollaya_laya_manifest, run_laya_training_job

    is_laya_manifest = False
    if base_model.startswith("laya") or family == "laya-markers-v1":
        try:
            resolve_ollaya_laya_manifest(base_model, models_dir=job.get("models_dir"))
            is_laya_manifest = True
        except Exception:
            is_laya_manifest = False

    if is_laya_manifest:
        run_laya_training_job(job, emit_event)
    else:
        from .hf_trainer import run_generic_training_job

        run_generic_training_job(job, emit_event)
    return 0


def cmd_adapt(cfg_arg: str) -> int:
    job = _load_job_config(cfg_arg)
    from .adapt_engine import adapt_model_to_ollaya

    adapt_model_to_ollaya(
        source=job["source"],
        output_dir=job["output_dir"],
        target_tag=job.get("target_tag", "adapted:v1"),
        layout=job.get("layout", "auto"),
        data_path=job.get("data_path"),
        epochs=int(job.get("epochs", 5)),
        lr=float(job.get("lr", 2e-4)),
        batch_size=int(job.get("batch_size", 4)),
        device=job.get("device", "auto"),
        emit=emit_event,
    )
    return 0


def cmd_merge(cfg_arg: str) -> int:
    job = _load_job_config(cfg_arg)
    from .hf_trainer import run_merge_job

    run_merge_job(job, emit_event)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="trainaya_py")
    sub = ap.add_subparsers(dest="subcmd", required=True)

    sub.add_parser("doctor")

    p_train = sub.add_parser("train")
    p_train.add_argument("--config", required=True)

    p_adapt = sub.add_parser("adapt")
    p_adapt.add_argument("--config", required=True)

    p_merge = sub.add_parser("merge")
    p_merge.add_argument("--config", required=True)

    args = ap.parse_args()
    try:
        if args.subcmd == "doctor":
            return cmd_doctor()
        if args.subcmd == "train":
            return cmd_train(args.config)
        if args.subcmd == "adapt":
            return cmd_adapt(args.config)
        if args.subcmd == "merge":
            return cmd_merge(args.config)
    except Exception as exc:
        emit_event(
            {
                "event": "error",
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
