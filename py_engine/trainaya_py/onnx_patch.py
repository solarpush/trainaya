"""ONNX weightless graph patching and export for Ollaya runtime compatibility.

In Ollaya's blob store (`/usr/share/ollaya/.ollaya/models/blobs/`), a weightless `model.onnx`
references tensors inside `model.safetensors` by exact filename (`sha256-<digest>`) and byte offset
(`offset`, `length`), while small transposed 2D tensors (`arr.size <= 4096`, specifically
`scorer.3.weight.T` of shape `[hidden_size, 1]`) are stored inline inside `model.onnx`.

Whenever `model.safetensors` is updated by PyTorch training, this module:
1. Rewrites all external initializer `location`, `offset`, and `length` attributes in both the
   `fp32` and `fp16` `model.onnx` graphs to point to the new `sha256-<new_weights_digest>` blob.
2. Updates the inline `scorer.3.weight.T` initializer (`val_1925` in fp32, `val_2460` in fp16)
   in-place with the newly trained `scorer.3.weight.T`.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
from typing import Dict, Tuple

import numpy as np


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def read_safetensors_index(path: str) -> Dict[str, Tuple[str, Tuple[int, ...], int, int]]:
    """Return {name: (dtype_str, shape_tuple, abs_byte_offset, byte_length)} from a safetensors header."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    base = 8 + n
    return {
        k: (
            v["dtype"],
            tuple(v["shape"]),
            base + v["data_offsets"][0],
            v["data_offsets"][1] - v["data_offsets"][0],
        )
        for k, v in header.items()
    }


def patch_laya_onnx_graph(
    src_onnx_path: str,
    dst_onnx_path: str,
    new_safetensors_path: str,
    new_weights_location: str,
    old_scorer3_weight_t: np.ndarray,
    new_scorer3_weight_t: np.ndarray,
) -> Dict[str, int]:
    """Patch an Ollaya weightless ONNX graph (fp32 or fp16) to reference updated safetensors and inline scorer.3.weight."""
    import onnx
    from onnx import TensorProto, numpy_helper

    index = read_safetensors_index(new_safetensors_path)
    model = onnx.load(src_onnx_path, load_external_data=False)

    updated_ext = 0
    updated_inline = 0
    d_dim = new_scorer3_weight_t.shape[0]

    for init in model.graph.initializer:
        if init.data_location == TensorProto.EXTERNAL:
            tname = init.name[2:] if init.name.startswith("w:") else init.name
            if tname in index:
                _, _, off, length = index[tname]
                del init.external_data[:]
                for k, v in (
                    ("location", new_weights_location),
                    ("offset", str(off)),
                    ("length", str(length)),
                ):
                    init.external_data.add(key=k, value=v)
                updated_ext += 1
        else:
            # Check for inline scorer.3.weight.T of shape [hidden_size, 1]
            if list(init.dims) == [d_dim, 1]:
                arr = numpy_helper.to_array(init)
                if np.allclose(arr.astype(np.float32), old_scorer3_weight_t.astype(np.float32), atol=1e-2):
                    replacement = new_scorer3_weight_t.astype(arr.dtype)
                    new_init = numpy_helper.from_array(replacement, name=init.name)
                    init.CopyFrom(new_init)
                    updated_inline += 1

    os.makedirs(os.path.dirname(os.path.abspath(dst_onnx_path)), exist_ok=True)
    onnx.save(model, dst_onnx_path)
    return {"external_updated": updated_ext, "inline_updated": updated_inline}
