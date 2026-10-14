# Trainaya

[![Rust](https://img.shields.io/badge/rust-2024%20edition-orange.svg?style=flat-square&logo=rust)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg?style=flat-square)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macos-lightgrey.svg?style=flat-square)](https://github.com/solarpush/trainaya)
[![CI](https://img.shields.io/github/actions/workflow/status/solarpush/trainaya/ci.yml?branch=main&style=flat-square&label=ci)](https://github.com/solarpush/trainaya/actions)

A high-performance, native Rust toolchain for curating, fine-tuning, merging, and benchmarking edge decision and classification models for the [Ollaya](https://github.com/solarpush/ollaya) runtime.

Zero Python runtime dependencies. Instant startup. Sub-millisecond SIMD kernels.

---

## Architecture Overview

```
                       ┌────────────────────────┐
                       │    Teacher Engines     │ (agy / claude / ollama)
                       └───────────┬────────────┘
                                   │ Multi-Teacher Consensus
                                   ▼
 ┌──────────────┐          trainaya curate           ┌──────────────────────┐
 │ Raw Inputs   │ ─────────────────────────────────► │ Contrastive Dataset  │
 └──────────────┘          (.Curatefile)             │ + Counterfactuals    │
        ▲                                            └──────────┬───────────┘
        │                                                       │
        │ EXPORT_FAILURES                                       ▼
        │ (Mined Overconfident                       trainaya train / build
        │  & Boundary Probes)                                (.Trainfile)
        │                                                       │
 ┌──────┴───────┐          trainaya bench            ┌──────────▼───────────┐
 │ 13 Public    │ ◄───────────────────────────────── │ Calibrated Weights   │
 │ Benchmarks   │          (.Benchfile)              │ (LoRA + T* Scaling)  │
 └──────────────┘                                    └──────────────────────┘
```

- **Pure Rust Autograd & Calibration**: Built on [Candle](https://github.com/huggingface/candle) with live differentiable LoRA graph injection, dynamic length-bucketed batching, cosine/linear LR schedules, global L2 gradient clipping, and post-training Temperature Scaling ($T^*$) baked directly into output weights.
- **Hardware-Vectorized Kernels (`src/quant/`)**:
  - Multithreaded LoRA merge with AVX2, FMA, and NEON vectorization ($W_{\text{merged}} = W_{\text{base}} + \frac{\alpha}{r} B \cdot A$).
  - 4-bit Normal Float 4 (`NF4`) quantization and dequantization.
  - Symmetric INT8 matrix quantization for fast CPU inference.
- **Multi-Family Support**:
  - **Bidirectional Encoders**: ModernBERT, DeBERTa, BERT, RoBERTa (`laya`, `nli`).
  - **Pointer / Causal Decoders**: Kev, Decider, Nimble (`kev-pointer-v1`).
  - **GGUF Vocab Logit & VLM Backbones**: Winnow, Gemma 3/4, Qwen (`winnow-v1`, `llm-logits-v1`).
- **Runtime Native Registration**: Direct registration into Ollaya's local manifest storage (`~/.ollaya/models/manifests/` or `/usr/share/ollaya/`).

---

## Installation

### One-Line Install Script

Installs the native prebuilt binary into `~/.local/bin` and runs self-diagnostics:

```bash
curl -fsSL https://raw.githubusercontent.com/solarpush/trainaya/main/install.sh | sh
```

### Build from Source

Requires Rust 1.85+ (2024 Edition):

```bash
git clone https://github.com/solarpush/trainaya.git
cd trainaya
cargo build --release
```

The optimized binary is located at `./target/release/trainaya`.

To build with GPU acceleration:
```bash
# NVIDIA CUDA
cargo build --release --features cuda

# Apple Silicon (Metal + Accelerate)
cargo build --release --features metal,accelerate
```

---

## The Three Declarative Files

Trainaya operates via three declarative configuration files designed for reproducible, closed-loop machine learning pipelines.

### 1. `.Curatefile` &mdash; Multi-Teacher Distillation & Counterfactual Twins

Distill raw data, re-annotate benchmark failures, or synthesize domain examples from one or more teacher engines (`agy`, `claude`, or local `ollama`). When multiple teachers are specified, Trainaya enforces strict consensus voting and automatically quarantines disagreements.

```dockerfile
# Single teacher or comma-separated multi-teacher consensus
TEACHER agy, claude

# Operational mode: label, synth, or augment
MODE augment

# Target output format: decision, classification, or nli
FORMAT decision

# Input and schema definitions
INPUT ./data/raw_inputs.jsonl
SCHEMA ./data/questions_schema.json
OUTPUT ./data/curated_contrastive.jsonl

# Execution parameters
PARAMETER batch_size 10
PARAMETER min_confidence 0.85
PARAMETER add_hard_negatives true
```

Run curation:
```bash
trainaya curate
```

When `add_hard_negatives true` or `MODE augment` is active, each validated sample is paired with a minimally perturbed counterfactual twin carrying the verified opposite label. Predictions below `min_confidence` or rejected by multi-teacher disagreement are isolated in a `.quarantine.jsonl` file.

---

### 2. `.Trainfile` &mdash; LoRA Fine-Tuning, Calibration & Packaging

Define the training recipe, optimizer schedule, LoRA target layers, and automatic Ollaya model tag:

```dockerfile
# Base foundation model (Hugging Face ID, Ollaya tag, or local directory)
FROM winnow:e4b
FAMILY winnow
TASK decision

# Training dataset
DATA ./data/curated_contrastive.jsonl

# Optimizer & schedule hyperparameters
PARAMETER epochs 3
PARAMETER batch_size 8
PARAMETER grad_accum 2
PARAMETER lr 0.0001
PARAMETER lr_scheduler cosine
PARAMETER warmup_ratio 0.1
PARAMETER max_grad_norm 1.0

# LoRA architecture parameters
PARAMETER lora_r 16
PARAMETER lora_alpha 32.0
PARAMETER lora_layers 8

# Ollaya tag versioning (auto-increments: v1 -> v2 -> v3)
TAG winnow:v{auto}

# Weight merging and quantization
MERGE true
QUANT Q8_0
```

Execute training:
```bash
trainaya train -f .Trainfile
```

At the end of training, Trainaya restores the best validation checkpoint and runs a two-stage grid search to find the optimal calibration temperature $T^*$ minimizing NLL and 10-bin Expected Calibration Error (ECE). The $1/T^*$ factor is folded directly into the output layer weights prior to saving or merging.

---

### 3. `.Benchfile` &mdash; Public Evaluation, Quality Gates & Failure Mining

Run standardized benchmark suites against the Ollaya runtime `/v1/systemone` endpoint and export failure probes back into the curation loop:

```dockerfile
# Model under test (resolves latest trained version automatically)
MODEL winnow:v{latest}

# Baseline reference for side-by-side comparative A/B evaluation
BASELINE winnow:e4b

# Standard benchmark subsets
SUBSETS multinli, paws, boolq, vitaminc-dev

# Ollaya daemon endpoint
URL http://127.0.0.1:11435

# Results report and active-learning failure export
OUTPUT ./results/bench_{model}_vs_{baseline}.json
EXPORT_FAILURES ./data/failures_{model}.jsonl

# Quality Gate assertion: fails build if overall accuracy drops below threshold
THRESHOLD min_accuracy 0.80
```

Run benchmark:
```bash
trainaya bench -f .Benchfile
```

---

## CLI Reference

### `trainaya doctor`
Audits the host environment: CPU SIMD feature sets (AVX2, AVX-512, NEON, FMA), GPU accelerators (CUDA, ROCm/HIP, Metal, Vulkan), release status, and local Ollaya daemon connectivity.

```bash
trainaya doctor
```

---

### `trainaya update`
Checks GitHub Releases for newer versions and atomically upgrades the local `trainaya` binary in place.

```bash
# Check and install the latest release
trainaya update

# Only check for updates without installing
trainaya update --check
```

---

### `trainaya curate`
Labels raw datasets, synthesizes domain data, or generates counterfactual twins using single or multi-teacher consensus.

```bash
# Using .Curatefile (default)
trainaya curate

# Augment exported benchmark failures with multi-teacher consensus
trainaya curate \
  --teacher agy,claude \
  --mode augment \
  --input ./data/failures_winnow_v1.jsonl \
  --schema ./data/schema.json \
  --output ./data/curated.jsonl
```

---

### `trainaya train`
Executes local LoRA fine-tuning and post-training temperature calibration on CPU or GPU.

```bash
trainaya train \
  --base "answerdotai/ModernBERT-base" \
  --data "./data/train.jsonl" \
  --task classification \
  --epochs 3 \
  --batch-size 8 \
  --grad-accum 2 \
  --lr 5e-5 \
  --lr-scheduler cosine \
  --warmup-ratio 0.1 \
  --max-grad-norm 1.0 \
  --lora-r 16 \
  --lora-alpha 32.0 \
  --output-dir "./output/adapter"
```

| Flag | Default | Description |
|---|---|---|
| `--base` | `answerdotai/ModernBERT-base` | HF repo identifier, Ollaya tag, or local model path |
| `--data` | *(required)* | Path to JSONL or CSV dataset |
| `--task` | `classification` | Task type (`classification`, `nli`, `decision`) |
| `--epochs` | `3` | Number of training epochs |
| `--batch-size` | `8` | Micro-batch size per forward pass |
| `--grad-accum` | `1` | Gradient accumulation steps |
| `--lr` | `5e-5` | Peak AdamW learning rate |
| `--lr-scheduler` | `cosine` | Schedule type (`cosine`, `linear`, `constant`) |
| `--warmup-ratio` | `0.1` | Fraction of total steps used for linear LR warmup |
| `--max-grad-norm` | `1.0` | Maximum global L2 gradient norm for clipping |
| `--lora-r` | `16` | LoRA rank dimension |
| `--lora-alpha` | `32.0` | LoRA scaling factor |
| `--lora-layers` | `4` | Number of top transformer blocks targeted in GGUF models |
| `--bf16` | `false` | Enable mixed-precision BF16 compute on supported GPUs |
| `--device` | `auto` | Execution device (`auto`, `cpu`, `cuda`, `metal`, `rocm`) |
| `--output-dir` | `./output_adapter` | Output directory for adapter weights |

---

### `trainaya merge`
Merges LoRA adapter weights directly into base model weights via SIMD vector kernels or `llama.cpp` GGUF fusion.

```bash
trainaya merge \
  --base "answerdotai/ModernBERT-base" \
  --lora "./output/adapter" \
  --output-dir "./output/merged"
```

---

### `trainaya export`
Packages merged models with optional INT8 / GGUF quantization and generates Ollaya runtime manifests.

```bash
# SafeTensors with INT8 quantization
trainaya export \
  --model "./output/merged" \
  --format safetensors \
  --quantize int8 \
  --output-dir "./output/dist"

# GGUF v3 format
trainaya export \
  --model "./output/merged" \
  --format gguf \
  --output-dir "./output/dist"
```

---

### `trainaya bench`
Evaluates models against standard public decision datasets, computing Accuracy, Latency percentiles (p50/p95), and 10-bin Expected Calibration Error (ECE).

```bash
# Comparative A/B run with active-learning failure mining
trainaya bench \
  --model winnow:v1 \
  --baseline winnow:e4b \
  --export-failures ./data/failures_winnow_v1.jsonl

# Specific subset run
trainaya bench --model winnow:v1 --subsets boolq,paws,multinli
```

#### Included Reference Subsets (3,880 Human Probes)

| Subset | Task | Target Type | Probe Fields |
|---|---|---|---|
| `vitaminc-dev` | Contrastive Claim Verification | Choice (3) | `claim`, `evidence` |
| `massive-en-US` | Intent Routing (English) | Choice (18) | `utterance`, `locale` |
| `massive-de-DE` | Intent Routing (German) | Choice (18) | `utterance`, `locale` |
| `boolq` | Boolean Question Answering | Noul (Bool) | `passage`, `question` |
| `squad2` | Unanswerability Detection | Noul (Bool) | `paragraph`, `question` |
| `paws` | Paraphrase Identification | Noul (Bool) | `sentence_1`, `sentence_2` |
| `multinli` | Natural Language Inference | Choice (3) | `premise`, `hypothesis` |
| `civil_comments`| Toxicity & Safety Moderation | Noul (Bool) | `comment` |
| `aegis2` | Guardrails & Prompt Injection | Noul (Bool) | `user_message` |
| `helpsteer2` | Response Quality Evaluation | Score (0-4) | `prompt`, `response` |
| `summeval-relevance` | Summary Relevance Assessment | Score (0-4) | `article`, `summary` |
| `summeval-consistency` | Summary Factuality Checking | Score (0-4) | `article`, `summary` |
| `pubmedqa` | Biomedical Question Answering | Choice (3) | `question`, `abstract` |

---

## Dataset Formats

### Single-Text Classification (`JSONL`)
```jsonl
{"text": "Production build succeeded without regressions.", "label": "positive"}
{"text": "Memory leak detected in connection pool worker.", "label": "negative"}
```

### Sentence-Pair / NLI (`JSONL`)
```jsonl
{"premise": "Rust enforces memory safety at compile time.", "hypothesis": "The language avoids dangling pointers.", "label": "entailment"}
{"premise": "The daemon process is running.", "hypothesis": "The port is closed and inactive.", "label": "contradiction"}
```

### Tabular Data (`CSV`)
Columns are detected automatically (`text`, `sentence`, `premise` for inputs; `label`, `target`, `category` for targets).

---

## External Tools & GGUF Support

Trainaya's core autograd, LoRA training, and SafeTensors pipeline are 100% native Rust without external dependencies.

For GGUF LoRA exporting and quantization on architectures requiring specialized GGUF manipulation (`llama-export-lora`, `llama-quantize`), Trainaya automatically discovers tools from:
1. Environment variables (`LLAMA_EXPORT_LORA`, `LLAMA_QUANTIZE`, or `LLAMA_CPP_DIR`)
2. Local build directory (`./vendor/llama.cpp/build/bin/`)
3. Standard system PATH (`/usr/local/bin`, `/usr/bin`)

To compile these optional GGUF binaries locally:
```bash
./scripts/setup_llama_cpp.sh
```

---

## Testing

Run the test suite:

```bash
cargo test
```

The test suite covers:
- `test_quant`: NF4 quantization/dequantization, INT8 quantization, and SIMD LoRA merge precision.
- `test_dataset`: JSONL, CSV, and Decision schema loading, dynamic batch padding, and stratified train/val splits.
- `test_lora`: Differentiable LoRA autograd graph injection, backward propagation, and weight updates.
- `test_families`: Architecture family dispatch (`winnow`, `kev`, `laya`, `nli`) and prompt formatting.
- `test_cli` & unit suites: `.Trainfile`, `.Curatefile`, `.Benchfile`, multi-teacher consensus, counterfactual twins, failure mining, and post-training Temperature Scaling ($T^*$).
- `test_e2e_pipeline`: End-to-end training, adapter fusion, and SafeTensors/GGUF packaging.

---

## License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
