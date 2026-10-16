"""Universal GGUF -> Ollaya Decision Head Trainer (`trainaya adapt`).

Implements:
1. GGUF v2/v3 tensor untying (`ensure_untied_output_weight`): if `output.weight` is tied to
   `token_embd.weight` (absent from GGUF tensor directory), clones `token_embd.weight` into a
   dedicated `output.weight` tensor at the end of the GGUF file so input embeddings are never
   mutated when training the decision head.
2. Native `libggml-base.so.0` row dequantizer/quantizer (`GgmlRowQuantizer`): dequantizes and
   re-quantizes individual rows of `output.weight` (`Q6_K`, `Q8_0`, `Q4_K`, `Q5_K`, `F16`, `F32`, etc.)
   with zero shape or block-layout mismatch.
3. Native `libllama.so.0` hidden-state extractor (`LlamaCppHiddenExtractor`): runs forward passes
   directly on the GGUF model with Longest-Common-Prefix (LCP) KV-cache reuse across prompts and
   extracts the normalized final hidden state `h_last` (`[d_model]`) at the decision position.
4. PyTorch Decision Head trainer (`train_gguf_decision_head`): trains the 26 choice rows (`A..Z`)
   and 10 score rows (`0..9`) of `output.weight` on ROCm / CUDA using soft probability targets
   (KL-divergence + Ranked Probability Score `RPS` for ordinal `score` + L2-SP anchor regularization
   + feature dropout), fits per-qtype temperatures `[T_choice, T_score, T_noul]`, and writes the
   quantized decision head back into `output.weight` of `model.gguf`.
"""

from __future__ import annotations

import ctypes
import json
import math
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset_loader import NormalizedDecisionRecord

CHOICE_LABELS: List[str] = [chr(ord("A") + i) for i in range(26)]
SCORE_LABELS: List[str] = [str(i) for i in range(10)]

SYSTEM_PROMPT: str = (
    "You are a calibrated decision engine. Read the state and the question, "
    "then select the single option label that best applies. Reply with the option label only."
)

# GGUF tensor type -> (block_elements, block_bytes, dequant_fn_name, quant_fn_name)
GGML_QUANT_SPECS: Dict[int, Tuple[int, int, Optional[str], Optional[str]]] = {
    0: (1, 4, None, None),  # F32
    1: (1, 2, None, None),  # F16
    2: (32, 18, "dequantize_row_q4_0", "quantize_row_q4_0_ref"),
    3: (32, 20, "dequantize_row_q4_1", "quantize_row_q4_1_ref"),
    6: (32, 22, "dequantize_row_q5_0", "quantize_row_q5_0_ref"),
    7: (32, 24, "dequantize_row_q5_1", "quantize_row_q5_1_ref"),
    8: (32, 34, "dequantize_row_q8_0", "quantize_row_q8_0_ref"),
    10: (256, 84, "dequantize_row_q2_K", "quantize_row_q2_K_ref"),
    11: (256, 110, "dequantize_row_q3_K", "quantize_row_q3_K_ref"),
    12: (256, 144, "dequantize_row_q4_K", "quantize_row_q4_K_ref"),
    13: (256, 176, "dequantize_row_q5_K", "quantize_row_q5_K_ref"),
    14: (256, 210, "dequantize_row_q6_K", "quantize_row_q6_K_ref"),
    30: (1, 2, "ggml_bf16_to_fp32_row", "ggml_fp32_to_bf16_row_ref"),
}


def _read_gguf_string(f) -> str:
    (length,) = struct.unpack("<Q", f.read(8))
    return f.read(length).decode("utf-8", errors="replace")


def _encode_gguf_string(s: str) -> bytes:
    raw = s.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


_GGUF_SCALAR_SIZES: Dict[int, int] = {
    0: 1,
    1: 1,
    2: 2,
    3: 2,
    4: 4,
    5: 4,
    6: 4,
    7: 1,
    10: 8,
    11: 8,
    12: 8,
}


def _read_gguf_value(f, vtype: int) -> Any:
    if vtype == 0:
        return struct.unpack("<B", f.read(1))[0]
    if vtype == 1:
        return struct.unpack("<b", f.read(1))[0]
    if vtype == 2:
        return struct.unpack("<H", f.read(2))[0]
    if vtype == 3:
        return struct.unpack("<h", f.read(2))[0]
    if vtype == 4:
        return struct.unpack("<I", f.read(4))[0]
    if vtype == 5:
        return struct.unpack("<i", f.read(4))[0]
    if vtype == 6:
        return struct.unpack("<f", f.read(4))[0]
    if vtype == 7:
        return struct.unpack("<?", f.read(1))[0]
    if vtype == 8:
        return _read_gguf_string(f)
    if vtype == 9:
        elem_type, count = struct.unpack("<IQ", f.read(12))
        if elem_type in _GGUF_SCALAR_SIZES and count > 64:
            f.seek(count * _GGUF_SCALAR_SIZES[elem_type], 1)
            return []
        return [_read_gguf_value(f, elem_type) for _ in range(count)]
    if vtype == 10:
        return struct.unpack("<Q", f.read(8))[0]
    if vtype == 11:
        return struct.unpack("<q", f.read(8))[0]
    if vtype == 12:
        return struct.unpack("<d", f.read(8))[0]
    raise ValueError(f"Unsupported GGUF metadata value type: {vtype}")


def ggml_row_bytes(ttype: int, n_cols: int) -> int:
    if ttype not in GGML_QUANT_SPECS:
        raise ValueError(f"Unsupported GGUF tensor quantization type: {ttype}")
    block_elems, block_bytes, _, _ = GGML_QUANT_SPECS[ttype]
    if n_cols % block_elems != 0:
        raise ValueError(
            f"Tensor row width {n_cols} is not divisible by block size {block_elems} for ttype={ttype}"
        )
    return (n_cols // block_elems) * block_bytes


def parse_gguf_layout(gguf_path: Path) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, int]]:
    """Parse GGUF v2/v3 metadata, tensor directory, and exact byte offsets."""
    meta: Dict[str, Any] = {}
    tensors: Dict[str, Dict[str, Any]] = {}
    with gguf_path.open("rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"{gguf_path} is not a valid GGUF file")
        version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
        kv_start = f.tell()
        for _ in range(n_kv):
            key = _read_gguf_string(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            val = _read_gguf_value(f, vtype)
            meta[key] = val
        kv_end = f.tell()

        for _ in range(n_tensors):
            tname = _read_gguf_string(f)
            (n_dims,) = struct.unpack("<I", f.read(4))
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            ttype, offset = struct.unpack("<IQ", f.read(12))
            tensors[tname] = {
                "dims": dims,
                "ttype": ttype,
                "offset": offset,
            }
        tensor_dir_end = f.tell()
        alignment = int(meta.get("general.alignment", 32))
        data_start = ((tensor_dir_end + alignment - 1) // alignment) * alignment

        for tname, tinfo in tensors.items():
            dims = tinfo["dims"]
            ttype = tinfo["ttype"]
            n_cols = int(dims[0])
            n_rows = 1
            for d in dims[1:]:
                n_rows *= int(d)
            try:
                rbytes = ggml_row_bytes(ttype, n_cols)
            except ValueError:
                rbytes = 0
            tinfo["n_cols"] = n_cols
            tinfo["n_rows"] = n_rows
            tinfo["row_bytes"] = rbytes
            tinfo["byte_len"] = rbytes * n_rows
            tinfo["abs_offset"] = data_start + int(tinfo["offset"])

    header_info = {
        "version": version,
        "n_tensors": n_tensors,
        "n_kv": n_kv,
        "kv_start": kv_start,
        "kv_end": kv_end,
        "tensor_dir_end": tensor_dir_end,
        "alignment": alignment,
        "data_start": data_start,
    }
    return meta, tensors, header_info


def _encode_tensor_info_entry(name: str, dims: List[int], ttype: int, offset: int) -> bytes:
    buf = bytearray(_encode_gguf_string(name))
    buf.extend(struct.pack("<I", len(dims)))
    for d in dims:
        buf.extend(struct.pack("<Q", int(d)))
    buf.extend(struct.pack("<IQ", int(ttype), int(offset)))
    return bytes(buf)


def ensure_untied_output_weight(
    src_gguf: Path,
    dst_gguf: Path,
    emit: Optional[Callable[[str, str], None]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Ensure `dst_gguf` has an explicit, untied `output.weight` tensor.

    If `src_gguf` omits `output.weight` (tying it to `token_embd.weight`), this function
    clones `token_embd.weight` as a new `output.weight` tensor at the aligned end of the
    GGUF data section so modifying `output.weight` never corrupts input token embeddings.
    """
    meta, tensors, hinfo = parse_gguf_layout(src_gguf)

    # Fast path: if dst_gguf already exists (not a symlink) and already has an untied output.weight of matching shape
    if dst_gguf.exists() and not dst_gguf.is_symlink() and src_gguf.resolve() != dst_gguf.resolve():
        try:
            meta2, tensors2, _ = parse_gguf_layout(dst_gguf)
            if "output.weight" in tensors2 and "token_embd.weight" in tensors:
                out_off = int(tensors2["output.weight"]["abs_offset"])
                blen = int(tensors2["output.weight"]["byte_len"])
                if "output.weight" in tensors and int(tensors["output.weight"]["byte_len"]) == blen:
                    if emit:
                        emit("reuse_untied_gguf", f"Reusing {dst_gguf.name} and resetting output.weight from source GGUF")
                    src_off = int(tensors["output.weight"]["abs_offset"])
                    with src_gguf.open("rb") as fsrc, dst_gguf.open("r+b") as fdst:
                        fsrc.seek(src_off)
                        fdst.seek(out_off)
                        chunk_sz = 64 * 1024 * 1024
                        rem = blen
                        while rem > 0:
                            buf = fsrc.read(min(chunk_sz, rem))
                            if not buf:
                                break
                            fdst.write(buf)
                            rem -= len(buf)
                    return meta2, tensors2
                elif (
                    "output.weight" not in tensors
                    and "token_embd.weight" in tensors2
                    and tensors2["output.weight"]["dims"] == tensors["token_embd.weight"]["dims"]
                ):
                    if emit:
                        emit("reuse_untied_gguf", f"Reusing untied {dst_gguf.name} and resetting output.weight from token_embd.weight")
                    embd_off = int(tensors2["token_embd.weight"]["abs_offset"])
                    with dst_gguf.open("r+b") as f:
                        chunk_sz = 64 * 1024 * 1024
                        copied = 0
                        while copied < blen:
                            to_read = min(chunk_sz, blen - copied)
                            f.seek(embd_off + copied)
                            buf = f.read(to_read)
                            f.seek(out_off + copied)
                            f.write(buf)
                            copied += len(buf)
                    return meta2, tensors2
        except Exception:
            pass

    if "output.weight" in tensors:
        if src_gguf.resolve() != dst_gguf.resolve():
            if emit:
                emit("gguf_copy", f"Copying GGUF with existing output.weight -> {dst_gguf.name}")
            shutil.copyfile(src_gguf, dst_gguf)
        meta2, tensors2, _ = parse_gguf_layout(dst_gguf)
        return meta2, tensors2

    if "token_embd.weight" not in tensors:
        raise RuntimeError(f"GGUF file {src_gguf} has neither output.weight nor token_embd.weight")

    embd_info = tensors["token_embd.weight"]
    embd_byte_len = int(embd_info["byte_len"])
    if embd_byte_len <= 0:
        raise RuntimeError(f"Invalid token_embd.weight byte_len={embd_byte_len}")

    alignment = hinfo["alignment"]
    src_size = src_gguf.stat().st_size
    orig_data_start = hinfo["data_start"]
    orig_data_len = src_size - orig_data_start
    aligned_orig_data_len = ((orig_data_len + alignment - 1) // alignment) * alignment
    pad_after_orig_data = aligned_orig_data_len - orig_data_len

    new_output_rel_offset = aligned_orig_data_len
    new_entry_bytes = _encode_tensor_info_entry(
        "output.weight",
        embd_info["dims"],
        embd_info["ttype"],
        new_output_rel_offset,
    )

    if emit:
        emit(
            "untie_output_weight",
            f"Untying output.weight from token_embd.weight ({embd_byte_len / (1024 * 1024):.1f} MB, ttype={embd_info['ttype']})",
        )

    tmp_dst = dst_gguf.with_suffix(".gguf.tmp")
    with src_gguf.open("rb") as fin, tmp_dst.open("wb") as fout:
        # 1. Write GGUF header with n_tensors + 1
        fout.write(b"GGUF")
        fout.write(struct.pack("<IQQ", hinfo["version"], hinfo["n_tensors"] + 1, hinfo["n_kv"]))

        # 2. Copy raw KV section + original tensor directory
        fin.seek(hinfo["kv_start"])
        header_and_dir_len = hinfo["tensor_dir_end"] - hinfo["kv_start"]
        fout.write(fin.read(header_and_dir_len))

        # 3. Append new tensor info entry for "output.weight"
        fout.write(new_entry_bytes)

        # 4. Pad to alignment for data_start
        cur_pos = fout.tell()
        new_data_start = ((cur_pos + alignment - 1) // alignment) * alignment
        if new_data_start > cur_pos:
            fout.write(b"\x00" * (new_data_start - cur_pos))

        # 5. Stream original tensor data section in 64 MB chunks
        fin.seek(orig_data_start)
        remaining = orig_data_len
        chunk_size = 64 * 1024 * 1024
        while remaining > 0:
            chunk = fin.read(min(chunk_size, remaining))
            if not chunk:
                break
            fout.write(chunk)
            remaining -= len(chunk)

        if pad_after_orig_data > 0:
            fout.write(b"\x00" * pad_after_orig_data)

        # 6. Copy token_embd.weight bytes to the new output.weight tensor at the end
        fin.seek(int(embd_info["abs_offset"]))
        remaining_embd = embd_byte_len
        while remaining_embd > 0:
            chunk = fin.read(min(chunk_size, remaining_embd))
            if not chunk:
                break
            fout.write(chunk)
            remaining_embd -= len(chunk)

    tmp_dst.replace(dst_gguf)
    meta2, tensors2, _ = parse_gguf_layout(dst_gguf)
    return meta2, tensors2


class GgmlRowQuantizer:
    """Wrapper around `/usr/local/lib/ollaya/llama/libggml-base.so.0` for row dequant/quant."""

    def __init__(self, lib_dir: Path = Path("/usr/local/lib/ollaya/llama")) -> None:
        self.lib_dir = lib_dir
        base_so = lib_dir / "libggml-base.so.0"
        if not base_so.exists():
            raise FileNotFoundError(f"libggml-base.so.0 not found at {base_so}")
        self.lib = ctypes.CDLL(str(base_so), mode=ctypes.RTLD_GLOBAL)
        self._dequant_fns: Dict[int, Any] = {}
        self._quant_fns: Dict[int, Any] = {}

        for ttype, (_, _, dq_name, q_name) in GGML_QUANT_SPECS.items():
            if dq_name and hasattr(self.lib, dq_name):
                fn_dq = getattr(self.lib, dq_name)
                fn_dq.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
                fn_dq.restype = None
                self._dequant_fns[ttype] = fn_dq
            if q_name and hasattr(self.lib, q_name):
                fn_q = getattr(self.lib, q_name)
                fn_q.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int64]
                fn_q.restype = None
                self._quant_fns[ttype] = fn_q

    def dequantize_row(self, raw_bytes: bytes, ttype: int, n_cols: int) -> np.ndarray:
        expected_len = ggml_row_bytes(ttype, n_cols)
        if len(raw_bytes) != expected_len:
            raise ValueError(f"Expected {expected_len} bytes for ttype={ttype}, got {len(raw_bytes)}")
        if ttype == 0:  # F32
            return np.frombuffer(raw_bytes, dtype=np.float32).copy()
        if ttype == 1:  # F16
            return np.frombuffer(raw_bytes, dtype=np.float16).astype(np.float32)

        fn_dq = self._dequant_fns.get(ttype)
        if fn_dq is None:
            raise ValueError(f"No dequantizer available in libggml-base.so.0 for ttype={ttype}")
        out = np.zeros(n_cols, dtype=np.float32)
        src_buf = ctypes.create_string_buffer(raw_bytes, len(raw_bytes))
        fn_dq(
            ctypes.cast(src_buf, ctypes.c_void_p),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int64(n_cols),
        )
        return out

    def quantize_row(self, vec_f32: np.ndarray, ttype: int) -> bytes:
        arr = np.ascontiguousarray(vec_f32, dtype=np.float32).reshape(-1)
        n_cols = int(arr.shape[0])
        out_len = ggml_row_bytes(ttype, n_cols)
        if ttype == 0:  # F32
            return arr.tobytes()
        if ttype == 1:  # F16
            return arr.astype(np.float16).tobytes()

        fn_q = self._quant_fns.get(ttype)
        if fn_q is None:
            raise ValueError(f"No quantizer available in libggml-base.so.0 for ttype={ttype}")
        dst_buf = ctypes.create_string_buffer(out_len)
        fn_q(
            arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(dst_buf, ctypes.c_void_p),
            ctypes.c_int64(n_cols),
        )
        return dst_buf.raw

    def read_tensor_rows(
        self, gguf_path: Path, tensor_info: Dict[str, Any], row_indices: List[int]
    ) -> np.ndarray:
        n_cols = int(tensor_info["n_cols"])
        n_rows = int(tensor_info["n_rows"])
        rbytes = int(tensor_info["row_bytes"])
        ttype = int(tensor_info["ttype"])
        abs_offset = int(tensor_info["abs_offset"])

        out = np.zeros((len(row_indices), n_cols), dtype=np.float32)
        with gguf_path.open("rb") as f:
            for i, ridx in enumerate(row_indices):
                if ridx < 0 or ridx >= n_rows:
                    raise IndexError(f"Row index {ridx} out of bounds [0, {n_rows})")
                f.seek(abs_offset + ridx * rbytes)
                raw = f.read(rbytes)
                out[i] = self.dequantize_row(raw, ttype, n_cols)
        return out

    def write_tensor_rows(
        self, gguf_path: Path, tensor_info: Dict[str, Any], row_map: Dict[int, np.ndarray]
    ) -> None:
        n_cols = int(tensor_info["n_cols"])
        n_rows = int(tensor_info["n_rows"])
        rbytes = int(tensor_info["row_bytes"])
        ttype = int(tensor_info["ttype"])
        abs_offset = int(tensor_info["abs_offset"])

        with gguf_path.open("r+b") as f:
            for ridx, vec in sorted(row_map.items()):
                if ridx < 0 or ridx >= n_rows:
                    raise IndexError(f"Row index {ridx} out of bounds [0, {n_rows})")
                if vec.shape[-1] != n_cols:
                    raise ValueError(f"Row vector length {vec.shape[-1]} != n_cols {n_cols}")
                qbytes = self.quantize_row(vec, ttype)
                f.seek(abs_offset + ridx * rbytes)
                f.write(qbytes)


class _LlamaModelParams(ctypes.Structure):
    _fields_ = [("data", ctypes.c_uint8 * 80)]


class _LlamaContextParams(ctypes.Structure):
    _fields_ = [("data", ctypes.c_uint8 * 160)]


class _LlamaBatch(ctypes.Structure):
    _fields_ = [
        ("n_tokens", ctypes.c_int32),
        ("_pad", ctypes.c_int32),
        ("token", ctypes.POINTER(ctypes.c_int32)),
        ("embd", ctypes.POINTER(ctypes.c_float)),
        ("pos", ctypes.POINTER(ctypes.c_int32)),
        ("n_seq_id", ctypes.POINTER(ctypes.c_int32)),
        ("seq_id", ctypes.POINTER(ctypes.POINTER(ctypes.c_int32))),
        ("logits", ctypes.POINTER(ctypes.c_int8)),
    ]


_LOG_CB_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)


class LlamaCppHiddenExtractor:
    """Direct FFI wrapper around `/usr/local/lib/ollaya/llama/libllama.so.0`.

    Extracts the normalized final hidden state `h_last` (`[d_model]`) for prompts formatted
    with Ollaya's `llm-logits-v1` template, using Longest-Common-Prefix (LCP) KV-cache reuse.
    """

    def __init__(
        self,
        gguf_path: Path,
        lib_dir: Path = Path("/usr/local/lib/ollaya/llama"),
        n_ctx: int = 4096,
        n_threads: int = 12,
        add_bos: bool = True,
    ) -> None:
        self.gguf_path = gguf_path
        self.lib_dir = lib_dir
        self.add_bos = add_bos
        base_path = lib_dir / "libggml-base.so.0"
        if base_path.exists():
            ctypes.CDLL(str(base_path), mode=ctypes.RTLD_GLOBAL)
        ggml_path = lib_dir / "libggml.so.0"
        if ggml_path.exists():
            self.ggml_lib = ctypes.CDLL(str(ggml_path), mode=ctypes.RTLD_GLOBAL)
            if hasattr(self.ggml_lib, "ggml_backend_load_all_from_path"):
                self.ggml_lib.ggml_backend_load_all_from_path.argtypes = [ctypes.c_char_p]
                self.ggml_lib.ggml_backend_load_all_from_path.restype = None
                rocm_dir = lib_dir.parent / "rocm"
                if rocm_dir.exists():
                    self.ggml_lib.ggml_backend_load_all_from_path(str(rocm_dir).encode("utf-8"))
                self.ggml_lib.ggml_backend_load_all_from_path(str(lib_dir).encode("utf-8"))
        llama_so = lib_dir / "libllama.so.0"
        if not llama_so.exists():
            raise FileNotFoundError(f"libllama.so.0 not found at {llama_so}")
        self.lib = ctypes.CDLL(str(llama_so), mode=ctypes.RTLD_GLOBAL)
        self._bind_functions()

        self._log_cb = _LOG_CB_TYPE(lambda lvl, txt, ud: None)
        if hasattr(self.lib, "llama_log_set"):
            self.lib.llama_log_set.argtypes = [_LOG_CB_TYPE, ctypes.c_void_p]
            self.lib.llama_log_set.restype = None
            self.lib.llama_log_set(self._log_cb, None)

        self.lib.llama_backend_init()
        mparams = self.lib.llama_model_default_params()
        # Offload all transformer layers to GPU when ROCm/Vulkan backend is present (falls back to CPU automatically)
        struct.pack_into("<i", mparams.data, 0x20, 999)

        self.model = self.lib.llama_model_load_from_file(
            str(gguf_path).encode("utf-8"), mparams
        )
        if not self.model:
            raise RuntimeError(f"llama_model_load_from_file failed for {gguf_path}")

        self.vocab = self.lib.llama_model_get_vocab(self.model)
        self.n_embd = int(self.lib.llama_model_n_embd(self.model))

        cparams = self.lib.llama_context_default_params()
        struct.pack_into("<I", cparams.data, 0x00, n_ctx)  # n_ctx
        struct.pack_into("<I", cparams.data, 0x04, 2048)   # n_batch
        struct.pack_into("<I", cparams.data, 0x08, 2048)   # n_ubatch

        self.ctx = self.lib.llama_init_from_model(self.model, cparams)
        if not self.ctx:
            self.lib.llama_model_free(self.model)
            raise RuntimeError(f"llama_init_from_model failed for {gguf_path}")

        self.lib.llama_set_embeddings(self.ctx, True)
        self.lib.llama_set_n_threads(self.ctx, n_threads, n_threads)
        self.mem = self.lib.llama_get_memory(self.ctx)
        self._cached_tokens: List[int] = []

    def _bind_functions(self) -> None:
        self.lib.llama_backend_init.argtypes = []
        self.lib.llama_backend_init.restype = None

        self.lib.llama_model_default_params.argtypes = []
        self.lib.llama_model_default_params.restype = _LlamaModelParams

        self.lib.llama_context_default_params.argtypes = []
        self.lib.llama_context_default_params.restype = _LlamaContextParams

        self.lib.llama_model_load_from_file.argtypes = [ctypes.c_char_p, _LlamaModelParams]
        self.lib.llama_model_load_from_file.restype = ctypes.c_void_p

        self.lib.llama_model_free.argtypes = [ctypes.c_void_p]
        self.lib.llama_model_free.restype = None

        self.lib.llama_model_get_vocab.argtypes = [ctypes.c_void_p]
        self.lib.llama_model_get_vocab.restype = ctypes.c_void_p

        self.lib.llama_model_n_embd.argtypes = [ctypes.c_void_p]
        self.lib.llama_model_n_embd.restype = ctypes.c_int32

        self.lib.llama_init_from_model.argtypes = [ctypes.c_void_p, _LlamaContextParams]
        self.lib.llama_init_from_model.restype = ctypes.c_void_p

        self.lib.llama_free.argtypes = [ctypes.c_void_p]
        self.lib.llama_free.restype = None

        self.lib.llama_set_embeddings.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        self.lib.llama_set_embeddings.restype = None

        self.lib.llama_set_n_threads.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32]
        self.lib.llama_set_n_threads.restype = None

        self.lib.llama_get_memory.argtypes = [ctypes.c_void_p]
        self.lib.llama_get_memory.restype = ctypes.c_void_p

        self.lib.llama_memory_clear.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        self.lib.llama_memory_clear.restype = None

        self.lib.llama_memory_seq_rm.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
        ]
        self.lib.llama_memory_seq_rm.restype = ctypes.c_bool

        self.lib.llama_tokenize.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
            ctypes.c_bool,
            ctypes.c_bool,
        ]
        self.lib.llama_tokenize.restype = ctypes.c_int32

        self.lib.llama_batch_get_one.argtypes = [
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
        ]
        self.lib.llama_batch_get_one.restype = _LlamaBatch

        self.lib.llama_decode.argtypes = [ctypes.c_void_p, _LlamaBatch]
        self.lib.llama_decode.restype = ctypes.c_int32

        self.lib.llama_get_embeddings_ith.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        self.lib.llama_get_embeddings_ith.restype = ctypes.POINTER(ctypes.c_float)

    def tokenize(self, text: str, add_special: bool = True, parse_special: bool = True) -> List[int]:
        raw = text.encode("utf-8")
        max_toks = len(raw) + 16
        buf = (ctypes.c_int32 * max_toks)()
        n = self.lib.llama_tokenize(
            self.vocab,
            raw,
            len(raw),
            buf,
            max_toks,
            add_special,
            parse_special,
        )
        if n < 0:
            max_toks = -n + 8
            buf = (ctypes.c_int32 * max_toks)()
            n = self.lib.llama_tokenize(
                self.vocab,
                raw,
                len(raw),
                buf,
                max_toks,
                add_special,
                parse_special,
            )
        return [int(buf[i]) for i in range(n)]

    def extract_hidden_state(self, prompt: str) -> np.ndarray:
        tokens = self.tokenize(prompt, add_special=self.add_bos, parse_special=True)
        if not tokens:
            raise ValueError("Prompt tokenized to 0 tokens")
        # Truncate middle if exceeding context window
        if len(tokens) > 1900:
            tokens = tokens[:900] + tokens[-1000:]

        # Find longest common prefix with currently cached KV sequence (leaving at least 1 token to decode)
        max_lcp = min(len(self._cached_tokens), len(tokens) - 1)
        lcp = 0
        while lcp < max_lcp and self._cached_tokens[lcp] == tokens[lcp]:
            lcp += 1

        if lcp == 0:
            self.lib.llama_memory_clear(self.mem, True)
        elif lcp < len(self._cached_tokens):
            ok = self.lib.llama_memory_seq_rm(self.mem, 0, lcp, -1)
            if not ok:
                self.lib.llama_memory_clear(self.mem, True)
                lcp = 0

        suffix = tokens[lcp:]
        n_suf = len(suffix)
        tok_arr = (ctypes.c_int32 * n_suf)(*suffix)
        batch = self.lib.llama_batch_get_one(tok_arr, n_suf)
        logits_flags = (ctypes.c_int8 * n_suf)()
        logits_flags[n_suf - 1] = 1
        batch.logits = logits_flags

        rc = self.lib.llama_decode(self.ctx, batch)
        if rc != 0:
            # Fallback: clear memory and decode full sequence
            self.lib.llama_memory_clear(self.mem, True)
            n_all = len(tokens)
            tok_arr_all = (ctypes.c_int32 * n_all)(*tokens)
            batch_all = self.lib.llama_batch_get_one(tok_arr_all, n_all)
            logits_flags_all = (ctypes.c_int8 * n_all)()
            logits_flags_all[n_all - 1] = 1
            batch_all.logits = logits_flags_all
            rc2 = self.lib.llama_decode(self.ctx, batch_all)
            if rc2 != 0:
                raise RuntimeError(f"llama_decode failed with code {rc2} (len={n_all})")

        self._cached_tokens = tokens
        emb_ptr = self.lib.llama_get_embeddings_ith(self.ctx, -1)
        if not emb_ptr:
            raise RuntimeError("llama_get_embeddings_ith(-1) returned NULL")

        vec = np.ctypeslib.as_array(emb_ptr, shape=(self.n_embd,)).copy()
        return vec

    def close(self) -> None:
        if getattr(self, "ctx", None):
            self.lib.llama_free(self.ctx)
            self.ctx = None
        if getattr(self, "model", None):
            self.lib.llama_model_free(self.model)
            self.model = None


@dataclass
class PreparedAdaptationSample:
    record_id: str
    state_key: str
    qtype: str  # "choice" | "score" | "noul"
    instructions: str
    prompt: str
    num_options: int
    target_probs: List[float]
    gold_index: int
    sample_weight: float
    subset: str


def build_llm_logits_prompt_and_targets(
    rec: NormalizedDecisionRecord,
    chat_template: Dict[str, str],
) -> Optional[PreparedAdaptationSample]:
    """Convert a NormalizedDecisionRecord into an exact `llm-logits-v1` prompt and target distribution.

    Crucially, in `ollaya_decision::llm_logits::user_message`:
    - `noul` orders options as `A. Yes` (`index 0 = true`) and `B. No` (`index 1 = false`),
      whereas `NormalizedDecisionRecord` stores `option_keys = ["false", "true"]` (`index 0 = false`,
      `index 1 = true`). We remap `noul` target probabilities to `[P(true), P(false)]` (`[A, B]`).
    """
    if isinstance(rec.state, str):
        state_str = rec.state
    else:
        state_str = json.dumps(rec.state, ensure_ascii=False)

    qtype = rec.qtype
    instructions = rec.instructions
    criteria = rec.criteria

    options_lines: List[str] = []
    if qtype == "noul":
        num_options = 2
        true_desc = criteria.get("true") if isinstance(criteria, dict) else None
        false_desc = criteria.get("false") if isinstance(criteria, dict) else None
        options_lines.append(f"A. Yes: {true_desc}" if true_desc else "A. Yes")
        options_lines.append(f"B. No: {false_desc}" if false_desc else "B. No")

        # Remap from NormalizedDecisionRecord [false, true] -> llm-logits-v1 [Yes (A), No (B)]
        if rec.target_probs and len(rec.target_probs) >= 2:
            # Check if rec.option_keys[0] is false/no
            first_key = str(rec.option_keys[0]).lower() if rec.option_keys else "false"
            if first_key in ("false", "no", "0"):
                p_true = float(rec.target_probs[1])
                p_false = float(rec.target_probs[0])
            else:
                p_true = float(rec.target_probs[0])
                p_false = float(rec.target_probs[1])
        else:
            first_key = str(rec.option_keys[0]).lower() if rec.option_keys else "false"
            is_true = (rec.gold_index == 1) if first_key in ("false", "no", "0") else (rec.gold_index == 0)
            p_true = 0.96 if is_true else 0.04
            p_false = 1.0 - p_true
        target_probs = [p_true, p_false]
        gold_idx = 0 if p_true >= p_false else 1

    elif qtype == "score":
        if isinstance(criteria, list) and len(criteria) >= 2:
            k = min(10, len(criteria))
            for i in range(k):
                options_lines.append(f"{SCORE_LABELS[i]}. {criteria[i]}")
        else:
            k = min(10, max(2, len(rec.option_keys)))
            for i in range(k):
                options_lines.append(f"{SCORE_LABELS[i]}. level {i}")
        num_options = k
        if rec.target_probs and len(rec.target_probs) >= k:
            target_probs = [float(p) for p in rec.target_probs[:k]]
        else:
            g = max(0, min(rec.gold_index, k - 1))
            # Smooth unimodal ordinal target around gold_index
            raw_p = [math.exp(-1.8 * abs(i - g)) for i in range(k)]
            s = sum(raw_p)
            target_probs = [p / s for p in raw_p]
        gold_idx = int(max(range(num_options), key=lambda i: target_probs[i]))

    else:  # "choice"
        if isinstance(criteria, dict) and len(criteria) >= 2:
            items = list(criteria.items())[:26]
            for i, (k_name, desc) in enumerate(items):
                if desc:
                    options_lines.append(f"{CHOICE_LABELS[i]}. {k_name}: {desc}")
                else:
                    options_lines.append(f"{CHOICE_LABELS[i]}. {k_name}")
            num_options = len(items)
        else:
            keys = rec.option_keys[:26] if len(rec.option_keys) >= 2 else ["false", "true"]
            for i, k_name in enumerate(keys):
                options_lines.append(f"{CHOICE_LABELS[i]}. {k_name}")
            num_options = len(keys)

        if rec.target_probs and len(rec.target_probs) >= num_options:
            target_probs = [float(p) for p in rec.target_probs[:num_options]]
        else:
            g = max(0, min(rec.gold_index, num_options - 1))
            smooth = 0.04 / max(1, num_options - 1)
            target_probs = [smooth] * num_options
            target_probs[g] = 0.96
        gold_idx = int(max(range(num_options), key=lambda i: target_probs[i]))

    # Normalize target_probs
    s = sum(target_probs)
    if s <= 0:
        return None
    target_probs = [p / s for p in target_probs]

    options_block = "\n".join(options_lines)
    user_body = (
        f"State:\n{state_str}\n\n"
        f"Question: {instructions}\n\n"
        f"Options:\n{options_block}\n"
        f"Answer with the label of the correct option only."
    )
    sys_prompt = chat_template.get("system", SYSTEM_PROMPT)
    prompt = (
        f"{chat_template['pre']}{sys_prompt}"
        f"{chat_template['mid']}{user_body}"
        f"{chat_template['post']}{chat_template.get('assistant_prefix', '')}"
    )

    return PreparedAdaptationSample(
        record_id=rec.record_id,
        state_key=state_str,
        qtype=qtype,
        instructions=instructions,
        prompt=prompt,
        num_options=num_options,
        target_probs=target_probs,
        gold_index=gold_idx,
        sample_weight=float(rec.sample_weight),
        subset=rec.subset,
    )


class GgufDecisionHeadModule(nn.Module):
    """Pretrained-Subspace-Preserving SVD-Whitened Decision Head.

    Decomposes each adapted output token row `w_k` (`[d_model]`) into three orthogonal terms:
    1. `w_k_base`: Pretrained token row with 1D constant frequency bias along `mu_hat` equalized
       (`cos(w_k_base, w_k_orig) > 0.99`).
    2. `b_k * (mu_hat / ||mu_h||_2)`: Trainable scalar logit threshold `b_k` along `mu_hat`
       (calibrates per-option prior inside bias-free GGUF `output.weight` with `< 0.1%` rotation).
    3. `TrustRegion_rho((z_k @ B_perp))`: Top-K centered SVD-whitened semantic subspace residual
       orthogonal to `mu_hat`, strictly bounded per row so `||Delta w_{k,perp}||_2 <= rho * ||w_k_base||_2`.
    """

    def __init__(
        self,
        w_choice_base: torch.Tensor,    # [26, d_model]
        w_score_base: torch.Tensor,     # [10, d_model]
        whitening_basis: torch.Tensor,  # [K, d_model] = b_scales.T * Vt_perp[:K]
        b_scales: torch.Tensor,         # [1, K]
        trust_rho: float = 0.32,
        dropout_p: float = 0.06,
    ) -> None:
        super().__init__()
        k_rank = whitening_basis.shape[0]
        dev = w_choice_base.device
        self.register_buffer("w_choice_base", w_choice_base.detach().clone())
        self.register_buffer("w_score_base", w_score_base.detach().clone())
        self.register_buffer("B", whitening_basis.detach().clone())
        self.register_buffer("b_scales", b_scales.detach().clone())

        c_norms = torch.norm(w_choice_base, dim=1, keepdim=True).clamp(min=1e-4)
        s_norms = torch.norm(w_score_base, dim=1, keepdim=True).clamp(min=1e-4)
        self.register_buffer("max_choice_delta", trust_rho * c_norms)
        self.register_buffer("max_score_delta", trust_rho * s_norms)

        self.z_choice = nn.Parameter(torch.zeros((w_choice_base.shape[0], k_rank), dtype=torch.float32, device=dev))
        self.z_score = nn.Parameter(torch.zeros((w_score_base.shape[0], k_rank), dtype=torch.float32, device=dev))
        self.dropout = nn.Dropout(p=dropout_p)

    def _bounded_z(self, z_mat: torch.Tensor, max_delta: torch.Tensor) -> torch.Tensor:
        # Since rows of Vt_perp are orthonormal, ||(z_mat @ B)_k||_2 == ||(z_mat * b_scales)_k||_2
        delta_norms = torch.norm(z_mat * self.b_scales, dim=1, keepdim=True)
        clip_coef = torch.clamp(max_delta / (delta_norms + 1e-7), max=1.0)
        return z_mat * clip_coef

    def forward_whitened(
        self,
        base_logits: torch.Tensor,
        z_feat: torch.Tensor,
        qtype: str,
        num_options: int,
        training: bool = False,
    ) -> torch.Tensor:
        if training:
            z_feat = self.dropout(z_feat)
        if qtype == "score":
            zs = self._bounded_z(self.z_score, self.max_score_delta)[:num_options]
            return base_logits + torch.mv(zs, z_feat)
        elif qtype == "noul":
            zc = self._bounded_z(self.z_choice, self.max_choice_delta)[:2]
            return base_logits + torch.mv(zc, z_feat)
        else:
            zc = self._bounded_z(self.z_choice, self.max_choice_delta)[:num_options]
            return base_logits + torch.mv(zc, z_feat)

    def materialize_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        zc = self._bounded_z(self.z_choice, self.max_choice_delta)
        zs = self._bounded_z(self.z_score, self.max_score_delta)
        wc = self.w_choice_base + torch.matmul(zc, self.B)
        ws = self.w_score_base + torch.matmul(zs, self.B)
        return wc, ws

    def l2_sp_loss(self) -> torch.Tensor:
        wc_delta = self.z_choice * self.b_scales
        ws_delta = self.z_score * self.b_scales
        return torch.mean(wc_delta * wc_delta) + torch.mean(ws_delta * ws_delta)


def _ranked_probability_score_loss(probs: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    cdf_pred = torch.cumsum(probs, dim=-1)
    cdf_targ = torch.cumsum(target_probs, dim=-1)
    return torch.mean((cdf_pred - cdf_targ) ** 2)


def train_gguf_decision_head(
    gguf_path: Path,
    tensor_info: Dict[str, Any],
    choice_token_ids: List[int],
    score_token_ids: List[int],
    noul_token_ids: List[int],
    chat_template: Dict[str, str],
    records: List[NormalizedDecisionRecord],
    epochs: int = 180,
    lr: float = 1.5e-2,
    l2_sp_weight: float = 0.005,
    n_threads: int = 12,
    add_bos: bool = True,
    yes_no_token_ids: Optional[List[int]] = None,
    emit: Optional[Callable[[str, str], None]] = None,
) -> Dict[str, Any]:
    """Extract hidden states from `gguf_path`, train the 36-row Decision Head in PyTorch
    (`26` choice/noul rows `A..Z` + `10` score rows `0..9`), quantize the updated rows back
    into `output.weight`, and return calibrated temperatures + metrics.
    """
    quantizer = GgmlRowQuantizer()
    w_choice_np = quantizer.read_tensor_rows(gguf_path, tensor_info, choice_token_ids)
    w_score_np = quantizer.read_tensor_rows(gguf_path, tensor_info, score_token_ids)

    # Prepare and sort samples by `state_key` to maximize LCP KV-cache hits
    prepared: List[PreparedAdaptationSample] = []
    for rec in records:
        s = build_llm_logits_prompt_and_targets(rec, chat_template)
        if s is not None:
            prepared.append(s)

    if not prepared:
        raise RuntimeError("No valid adaptation samples prepared for GGUF Decision Head training")

    prepared.sort(key=lambda x: x.state_key)

    # Incremental prompt-keyed cache in `hidden_states_cache.npz`
    cache_path = gguf_path.parent / "hidden_states_cache.npz"
    cache_map: Dict[str, np.ndarray] = {}
    if cache_path.exists():
        try:
            cached_data = np.load(cache_path, allow_pickle=True)
            c_prompts = cached_data["prompts"]
            c_hmat = cached_data["hidden_mat"].astype(np.float32)
            for p_str, h_row in zip(c_prompts, c_hmat):
                cache_map[str(p_str)] = h_row
        except Exception:
            cache_map = {}

    missing = [s for s in prepared if s.prompt not in cache_map]
    if not missing and cache_map:
        if emit:
            sample_dim = next(iter(cache_map.values())).shape[0]
            emit(
                "extract_hidden_cached",
                f"Loaded all {len(prepared)} cached h_last vectors (d_model={sample_dim}) from {cache_path.name}",
            )
    else:
        if emit:
            emit(
                "extract_hidden_start",
                f"Extracting h_last vectors for {len(missing)}/{len(prepared)} uncached samples via libllama.so.0 (LCP KV-cache enabled)...",
            )
        extractor = LlamaCppHiddenExtractor(gguf_path, n_threads=n_threads, add_bos=add_bos)
        try:
            for idx, sample in enumerate(missing):
                try:
                    h_vec = extractor.extract_hidden_state(sample.prompt)
                    if np.all(np.isfinite(h_vec)):
                        cache_map[sample.prompt] = h_vec.astype(np.float32)
                except Exception as e:
                    if emit:
                        emit("extract_warn", f"Skipped sample {sample.record_id}: {e}")
                if emit and ((idx + 1) % 100 == 0 or (idx + 1) == len(missing)):
                    emit(
                        "extract_hidden_progress",
                        f"Extracted {idx + 1}/{len(missing)} hidden states on GPU (d_model={extractor.n_embd})",
                    )
                if (idx + 1) % 500 == 0:
                    try:
                        ck_prompts = list(cache_map.keys())
                        ck_hmat = np.stack([cache_map[k] for k in ck_prompts], axis=0).astype(np.float32)
                        np.savez(cache_path, hidden_mat=ck_hmat, prompts=np.array(ck_prompts, dtype=object))
                    except Exception:
                        pass
        finally:
            extractor.close()

    valid_samples: List[PreparedAdaptationSample] = []
    hidden_list: List[np.ndarray] = []
    for s in prepared:
        h_vec = cache_map.get(s.prompt)
        if h_vec is not None:
            valid_samples.append(s)
            hidden_list.append(h_vec)

    hidden_mat = np.stack(hidden_list, axis=0).astype(np.float32)  # [N, d_model]
    if missing:
        try:
            ck_prompts = list(cache_map.keys())
            ck_hmat = np.stack([cache_map[k] for k in ck_prompts], axis=0).astype(np.float32)
            np.savez(
                cache_path,
                hidden_mat=ck_hmat,
                prompts=np.array(ck_prompts, dtype=object),
            )
        except Exception:
            pass

    n_samples, d_model = hidden_mat.shape

    # Balance sample weights across classes within repeated questions so template directions have zero class-prior bias
    from collections import defaultdict
    q_groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for i, s in enumerate(valid_samples):
        q_groups[(s.qtype, s.instructions)].append(i)
    for _, idxs in q_groups.items():
        if len(idxs) >= 3:
            cls_counts: Dict[int, int] = defaultdict(int)
            for i in idxs:
                cls_counts[valid_samples[i].gold_index] += 1
            if len(cls_counts) >= 2:
                avg_c = len(idxs) / float(len(cls_counts))
                for i in idxs:
                    c_g = cls_counts[valid_samples[i].gold_index]
                    factor = max(0.4, min(3.5, avg_c / float(c_g)))
                    valid_samples[i].sample_weight *= factor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if emit:
        dev_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
        emit(
            "pytorch_head_train_start",
            f"Training Pretrained-Subspace-Preserving GGUF Decision Head [36 x {d_model}] on {device.type.upper()} ({dev_name}) over {n_samples} samples...",
        )

    w_choice_orig_t = torch.from_numpy(w_choice_np).to(device=device, dtype=torch.float32)
    w_score_orig_t = torch.from_numpy(w_score_np).to(device=device, dtype=torch.float32)
    w_choice_init_t = w_choice_orig_t.clone()
    w_score_init_t = w_score_orig_t.clone()
    h_all_t = torch.from_numpy(hidden_mat).to(device=device, dtype=torch.float32)

    orig_choice_mean_norm = float(torch.norm(w_choice_orig_t, dim=1).mean().item())
    orig_score_mean_norm = float(torch.norm(w_score_orig_t, dim=1).mean().item())

    def _evaluate_accuracy(wc: torch.Tensor, ws: torch.Tensor) -> Dict[str, float]:
        correct_by_type = {"choice": 0, "score": 0, "noul": 0, "total": 0}
        count_by_type = {"choice": 0, "score": 0, "noul": 0, "total": 0}
        with torch.no_grad():
            for i, s in enumerate(valid_samples):
                h = h_all_t[i]
                if s.qtype == "score":
                    w = ws[: s.num_options]
                elif s.qtype == "noul":
                    w = wc[:2]
                else:
                    w = wc[: s.num_options]
                logits = torch.mv(w, h)
                pred = int(torch.argmax(logits).item())
                is_ok = int(pred == s.gold_index)
                correct_by_type[s.qtype] += is_ok
                count_by_type[s.qtype] += 1
                correct_by_type["total"] += is_ok
                count_by_type["total"] += 1
        return {
            k: round(correct_by_type[k] / max(1, count_by_type[k]), 4)
            for k in ("total", "choice", "score", "noul")
        }

    baseline_acc = _evaluate_accuracy(w_choice_init_t, w_score_init_t)

    # Step 1: Per-Qtype 1D Mean-Projection Equalization (`mu_cn_hat` for choice/noul, `mu_s_hat` for score)
    # Eliminates pretraining token-frequency bias (e.g. A/B > C..F on choice prompts and 1/2 > 4 on score prompts)
    # while keeping cosine similarity > 0.99 with the original pretrained token embeddings.
    with torch.no_grad():
        cn_indices = [i for i, s in enumerate(valid_samples) if s.qtype in ("choice", "noul")]
        c4_indices = [i for i, s in enumerate(valid_samples) if s.qtype == "choice" and s.num_options >= 4]
        s_indices = [i for i, s in enumerate(valid_samples) if s.qtype == "score"]
        s5_indices = [i for i, s in enumerate(valid_samples) if s.qtype == "score" and s.num_options >= 5]

        mu_cn = h_all_t[cn_indices].mean(dim=0) if cn_indices else h_all_t.mean(dim=0)
        mu_cn_hat = mu_cn / torch.clamp(torch.norm(mu_cn), min=1e-6)

        mu_c4 = h_all_t[c4_indices].mean(dim=0) if c4_indices else mu_cn
        mu_c4_hat = mu_c4 / torch.clamp(torch.norm(mu_c4), min=1e-6)

        mu_s = h_all_t[s_indices].mean(dim=0) if s_indices else h_all_t.mean(dim=0)
        mu_s_hat = mu_s / torch.clamp(torch.norm(mu_s), min=1e-6)

        mu_s5 = h_all_t[s5_indices].mean(dim=0) if s5_indices else mu_s
        mu_s5_hat = mu_s5 / torch.clamp(torch.norm(mu_s5), min=1e-6)

        w_choice_base = w_choice_init_t.clone()
        c4_proj = torch.mv(w_choice_init_t[:4], mu_c4_hat)
        c4_shift = torch.clamp(0.35 * (c4_proj - c4_proj.mean()), min=-1.0, max=1.0)
        w_choice_base[:4] = w_choice_init_t[:4] - torch.ger(c4_shift, mu_c4_hat)

        w_score_base = w_score_init_t.clone()
        s5_proj = torch.mv(w_score_init_t[:5], mu_s5_hat)
        s5_shift = torch.clamp(0.65 * (s5_proj - s5_proj.mean()), min=-2.2, max=2.2)
        w_score_base[:5] = w_score_init_t[:5] - torch.ger(s5_shift, mu_s5_hat)

        debiased_acc = _evaluate_accuracy(w_choice_base, w_score_base)

        # Only consider blending yes/no into A/B for unaligned models if it improves BOTH noul and choice
        if debiased_acc["total"] < 0.50 and yes_no_token_ids and len(yes_no_token_ids) == 2:
            w_yn_np = quantizer.read_tensor_rows(gguf_path, tensor_info, yes_no_token_ids)
            w_yn_t = torch.from_numpy(w_yn_np).to(device=device, dtype=torch.float32)
            yn_proj = torch.mv(w_yn_t, mu_cn_hat)
            c2_mean = torch.mv(w_choice_base[:2], mu_cn_hat).mean()
            w_yn_debiased = w_yn_t - torch.ger(yn_proj - c2_mean, mu_cn_hat)
            wc_cand = w_choice_base.clone()
            wc_cand[:2] = 0.65 * w_choice_base[:2] + 0.35 * w_yn_debiased
            cand_acc = _evaluate_accuracy(wc_cand, w_score_base)
            if cand_acc["noul"] > debiased_acc["noul"] and cand_acc["choice"] >= debiased_acc["choice"]:
                w_choice_base = wc_cand
                debiased_acc = cand_acc

    if emit:
        emit(
            "baseline_eval",
            f"Pre-adaptation accuracy: raw={baseline_acc['total']*100:.1f}% -> 1D-debiased={debiased_acc['total']*100:.1f}% "
            f"(noul={debiased_acc['noul']*100:.1f}%, choice={debiased_acc['choice']*100:.1f}%, score={debiased_acc['score']*100:.1f}%)",
        )

    # Step 2: Centered Top-K SVD Whitening in the subspace orthogonal to the per-qtype template means
    with torch.no_grad():
        h_perp = torch.zeros_like(h_all_t)
        for i, s in enumerate(valid_samples):
            m_hat = mu_s_hat if s.qtype == "score" else mu_cn_hat
            h_perp[i] = h_all_t[i] - torch.dot(h_all_t[i], m_hat) * m_hat
        _, s_vals, vt_perp = torch.linalg.svd(h_perp, full_matrices=False)
        s_max = float(s_vals[0].item()) if s_vals.numel() > 0 else 1.0
        # Keep principal semantic directions and discard the low-variance noise tail
        keep_mask = s_vals >= (0.045 * s_max)
        k_rank = int(torch.clamp(keep_mask.sum(), min=min(24, s_vals.numel()), max=min(96, s_vals.numel())).item())
        s_top = s_vals[:k_rank]
        vt_top = vt_perp[:k_rank]  # [K, d_model] with orthonormal rows

        tau = 0.25 * s_max
        # Scale basis proportionally to pretrained row norm so optimization steps are model-scale-invariant
        scale_vec = (1.5 * orig_choice_mean_norm * (s_max / (s_top + tau))).unsqueeze(1)  # [K, 1]
        b_scales = scale_vec.t().contiguous()  # [1, K]
        whitening_basis = scale_vec * vt_top   # [K, d_model]
        z_features_all = torch.matmul(h_perp, whitening_basis.t())  # [N, K]

        # Strong instruct models (debiased_acc >= 0.50) use a tight trust region (rho=0.35, cos >= 0.94)
        # to preserve zero-shot semantic alignment on unseen benchmarks; guard/base models use rho=0.85.
        trust_rho = 0.35 if debiased_acc["total"] >= 0.50 else 0.85

        base_logits_all: List[torch.Tensor] = []
        for i, s in enumerate(valid_samples):
            h = h_all_t[i]
            if s.qtype == "score":
                wb = w_score_base[: s.num_options]
            elif s.qtype == "noul":
                wb = w_choice_base[:2]
            else:
                wb = w_choice_base[: s.num_options]
            base_logits_all.append(torch.mv(wb, h))

    epochs = 90 if n_samples >= 1000 else epochs
    head = GgufDecisionHeadModule(
        w_choice_base,
        w_score_base,
        whitening_basis,
        b_scales,
        trust_rho=trust_rho,
        dropout_p=0.06,
    ).to(device)
    eff_lr = max(1.0e-2, lr)
    optimizer = torch.optim.AdamW(head.parameters(), lr=eff_lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=eff_lr * 0.05)

    target_tensors = [
        torch.tensor(s.target_probs, device=device, dtype=torch.float32) for s in valid_samples
    ]

    best_wc = w_choice_base.detach().clone()
    best_ws = w_score_base.detach().clone()
    best_cn_score = -1.0
    best_s_score = -1.0

    accum_steps = 64 if n_samples >= 1000 else 16
    rng = np.random.default_rng(42)
    for epoch in range(1, epochs + 1):
        head.train()
        perm = rng.permutation(n_samples)
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step_idx, sample_idx in enumerate(perm):
            idx_i = int(sample_idx)
            s = valid_samples[idx_i]
            z_feat = z_features_all[idx_i]
            b_log = base_logits_all[idx_i]
            targ = target_tensors[idx_i]

            logits = head.forward_whitened(b_log, z_feat, s.qtype, s.num_options, training=True)
            log_probs = F.log_softmax(logits, dim=-1)
            probs = torch.exp(log_probs)

            kl_loss = F.kl_div(log_probs, targ, reduction="sum")
            loss = kl_loss
            if s.qtype == "score":
                rps = _ranked_probability_score_loss(probs, targ)
                loss = loss + 1.5 * rps

            weighted_loss = loss * s.sample_weight
            (weighted_loss / float(accum_steps)).backward()
            epoch_loss += float(loss.item())

            if (step_idx + 1) % accum_steps == 0 or (step_idx + 1) == n_samples:
                reg_loss = l2_sp_weight * head.l2_sp_loss()
                reg_loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=2.0)
                optimizer.step()
                optimizer.zero_grad()

        scheduler.step()
        with torch.no_grad():
            wc_now, ws_now = head.materialize_weights()
            acc = _evaluate_accuracy(wc_now, ws_now)
            cn_metric = 0.5 * (acc["noul"] + acc["choice"]) - 1e-4 * (epoch_loss / n_samples)
            if cn_metric > best_cn_score:
                best_cn_score = cn_metric
                best_wc = wc_now.detach().clone()
            s_metric = acc["score"] - 1e-4 * (epoch_loss / n_samples)
            if s_metric > best_s_score:
                best_s_score = s_metric
                best_ws = ws_now.detach().clone()

        if emit and (epoch == 1 or epoch % 30 == 0 or epoch == epochs):
            best_acc_now = _evaluate_accuracy(best_wc, best_ws)
            emit(
                "pytorch_head_epoch",
                f"Epoch {epoch:03d}/{epochs:03d} — loss={epoch_loss/n_samples:.4f} — "
                f"acc={best_acc_now['total']*100:.1f}% (noul={best_acc_now['noul']*100:.1f}%, choice={best_acc_now['choice']*100:.1f}%, score={best_acc_now['score']*100:.1f}%)",
            )

    head.eval()
    # Globally rescale each head matrix by a single positive scalar so average row L2 norm matches
    # the pretrained `output.weight` rows (preserving 100% of argmax decision boundaries and Q6_K dynamic range)
    with torch.no_grad():
        curr_c_norm = float(torch.norm(best_wc, dim=1).mean().item())
        if curr_c_norm > 1e-6:
            best_wc = best_wc * (orig_choice_mean_norm / curr_c_norm)
        curr_s_norm = float(torch.norm(best_ws, dim=1).mean().item())
        if curr_s_norm > 1e-6:
            best_ws = best_ws * (orig_score_mean_norm / curr_s_norm)

        cos_c = float(F.cosine_similarity(best_wc[:6], w_choice_orig_t[:6], dim=1).mean().item())
        cos_s = float(F.cosine_similarity(best_ws[:5], w_score_orig_t[:5], dim=1).mean().item())

    w_choice_trained = best_wc.detach().cpu().numpy()
    w_score_trained = best_ws.detach().cpu().numpy()

    row_updates: Dict[int, np.ndarray] = {}
    ttype = int(tensor_info["ttype"])
    n_cols = int(tensor_info["n_cols"])

    w_choice_quant_dequant = np.zeros_like(w_choice_trained)
    for i, tid in enumerate(choice_token_ids):
        qbytes = quantizer.quantize_row(w_choice_trained[i], ttype)
        w_choice_quant_dequant[i] = quantizer.dequantize_row(qbytes, ttype, n_cols)
        row_updates[tid] = w_choice_trained[i]

    w_score_quant_dequant = np.zeros_like(w_score_trained)
    for i, tid in enumerate(score_token_ids):
        qbytes = quantizer.quantize_row(w_score_trained[i], ttype)
        w_score_quant_dequant[i] = quantizer.dequantize_row(qbytes, ttype, n_cols)
        row_updates[tid] = w_score_trained[i]

    quantizer.write_tensor_rows(gguf_path, tensor_info, row_updates)

    wc_q_t = torch.from_numpy(w_choice_quant_dequant).to(device=device, dtype=torch.float32)
    ws_q_t = torch.from_numpy(w_score_quant_dequant).to(device=device, dtype=torch.float32)
    post_quant_acc = _evaluate_accuracy(wc_q_t, ws_q_t)

    # Fit per-qtype temperatures [T_choice, T_score, T_noul] on quantized logits via KL minimization
    temps_by_type: Dict[str, float] = {}
    with torch.no_grad():
        for qt in ("choice", "score", "noul"):
            qt_indices = [i for i, s in enumerate(valid_samples) if s.qtype == qt]
            if not qt_indices:
                temps_by_type[qt] = 1.0
                continue
            best_t = 1.0
            best_kl = float("inf")
            for cand in np.linspace(0.25, 3.5, 66):
                t_val = float(cand)
                total_kl = 0.0
                for i in qt_indices:
                    s = valid_samples[i]
                    h = h_all_t[i]
                    if qt == "score":
                        w = ws_q_t[: s.num_options]
                    elif qt == "noul":
                        w = wc_q_t[:2]
                    else:
                        w = wc_q_t[: s.num_options]
                    logits = torch.mv(w, h) / t_val
                    log_p = F.log_softmax(logits, dim=-1)
                    total_kl += float(F.kl_div(log_p, target_tensors[i], reduction="sum").item())
                if total_kl < best_kl:
                    best_kl = total_kl
                    best_t = t_val
            temps_by_type[qt] = round(best_t, 4)

    temperatures = [
        temps_by_type["choice"],
        temps_by_type["score"],
        temps_by_type["noul"],
    ]

    if emit:
        emit(
            "gguf_head_write_complete",
            f"Wrote {len(row_updates)} quantized decision rows (ttype={ttype}, rank={k_rank}, rho={trust_rho:.2f}, "
            f"cos_choice={cos_c:.3f}, cos_score={cos_s:.3f}) to {gguf_path.name} — "
            f"post-quant accuracy={post_quant_acc['total']*100:.1f}% "
            f"(noul={post_quant_acc['noul']*100:.1f}%, choice={post_quant_acc['choice']*100:.1f}%, score={post_quant_acc['score']*100:.1f}%), "
            f"temperatures={temperatures}",
        )

    return {
        "n_samples": n_samples,
        "d_model": d_model,
        "ttype": ttype,
        "baseline_accuracy": baseline_acc,
        "debiased_accuracy": debiased_acc,
        "post_quant_accuracy": post_quant_acc,
        "cos_choice": round(cos_c, 4),
        "cos_score": round(cos_s, 4),
        "temperatures": temperatures,
    }
