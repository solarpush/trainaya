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
                      │    Teacher Engine      │ (agy / claude / ollama)
                      └───────────┬────────────┘
                                  │
                                  ▼
 ┌──────────────┐          trainaya curate           ┌──────────────────────┐
 │ Raw Inputs   │ ─────────────────────────────────► │ Curated Dataset      │
 └──────────────┘          (.Curatefile)             │ (JSONL / CSV)        │
                                                     └──────────┬───────────┘
                                                                │
                                                                ▼
 ┌──────────────┐          trainaya train / build    ┌──────────────────────┐
 │ Base Model   │ ─────────────────────────────────► │ Fine-Tuned Weights   │
 │ (HF / GGUF)  │          (.Trainfile)              │ (SafeTensors / GGUF) │
 └──────────────┘                                    └──────────┬───────────┘
                                                                │
                                                                ▼
 ┌──────────────┐          trainaya bench            ┌──────────────────────┐
 │ 13 Public    │ ─────────────────────────────────► │ Ollaya Registration  │
 │ Benchmarks   │          (.Benchfile)              │ & Quality Gate Check │
 └──────────────┘                                    └──────────────────────┘
```

- **Pure Rust Autograd & Layers**: Built on [Candle](https://github.com/huggingface/candle) with custom modules for classification heads, cross-entropy, and LoRA adaptors.
- **Hardware-Vectorized Kernels (`src/quant/`)**:
  - Multithreaded LoRA merge with AVX2, FMA, and NEON vectorization ($W_{merged} = W_{base} + \frac{\alpha}{r} B \cdot A$).
  - 4-bit Normal Float 4 (`NF4`) quantization and dequantization.
  - Symmetric INT8 matrix quantization for fast CPU inference.
- **Multi-Family Support**:
  - **Bidirectional Encoders**: ModernBERT, DeBERTa, BERT, GLiClass.
  - **Pointer / Causal Decoders**: Kev, Decider, Nimble, Qwen3Guard.
  - **GGUF Vocab Logit Decoders**: Winnow, Gemma 4, LLM Logits.
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

To build with CUDA acceleration:
```bash
cargo build --release --features cuda
```

---

## The Three Declarative Files

Trainaya operates via three declarative configuration files designed for reproducibility and version-controlled machine learning pipelines.

### 1. `.Curatefile` &mdash; Zero-Cost Dataset Distillation

Distill raw data or generate synthetic examples from a teacher CLI session (`agy`, `claude`, or local `ollama`) without separate API bills:

```dockerfile
# Teacher CLI engine (uses your active local CLI session)
TEACHER agy

# Operational mode: label, synth, or augment
MODE label

# Target output format: decision, classification, or nli
FORMAT decision

# Input and schema definitions
INPUT ./data/raw_inputs.jsonl
SCHEMA ./data/questions_schema.json
OUTPUT ./data/curated_from_teacher.jsonl

# Teacher execution parameters
PARAMETER batch_size 10
PARAMETER min_confidence 0.85
PARAMETER add_hard_negatives false
```

Run curation:
```bash
trainaya curate
```

Uncertain predictions below `min_confidence` are automatically isolated to a `.quarantine.jsonl` file for manual review.

---

### 2. `.Trainfile` &mdash; LoRA Fine-Tuning & Packaging

Define the training recipe, LoRA rank parameters, target format, and automatic Ollaya model tag:

```dockerfile
# Base foundation model (Hugging Face ID or local directory)
FROM winnow:e4b
FAMILY winnow
TASK decision

# Training dataset
DATA ./data/train_specific_decision.jsonl

# Hyperparameters
PARAMETER epochs 3
PARAMETER batch_size 4
PARAMETER lr 0.0001
PARAMETER lora_r 16
PARAMETER lora_alpha 32.0

# Ollaya tag versioning (auto-increments: v1 -> v2 -> v3)
TAG winnow:v{auto}

# Weight merging and export container
MERGE true
EXPORT gguf
```

Execute build:
```bash
trainaya build -f .Trainfile
```

---

### 3. `.Benchfile` &mdash; Public Evaluation & Quality Gates

Run standardized benchmark suites directly against the Ollaya runtime `/v1/systemone` endpoint:

```dockerfile
# Model under test (resolves latest trained version automatically)
MODEL winnow:v{latest}

# Baseline reference for side-by-side comparative A/B evaluation
BASELINE winnow:e4b

# Standard benchmark subsets
SUBSETS multinli, paws, boolq, vitaminc-dev

# Ollaya daemon endpoint
URL http://127.0.0.1:11435

# Results report output path
OUTPUT ./results/bench_{model}_vs_{baseline}.json

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
Audits the host environment: CPU SIMD feature sets (AVX2, AVX-512, NEON, FMA), GPU accelerators, release update status, and local Ollaya daemon connectivity.

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
Generates or labels datasets from teacher engines according to a schema.

```bash
# Using .Curatefile (default)
trainaya curate

# Ad-hoc command line usage
trainaya curate \
  --teacher agy \
  --input ./data/raw.jsonl \
  --schema ./data/schema.json \
  --output ./data/curated.jsonl
```

---

### `trainaya train`
Executes local LoRA fine-tuning on CPU or GPU.

```bash
trainaya train \
  --base "answerdotai/ModernBERT-base" \
  --data "./data/train.jsonl" \
  --task classification \
  --epochs 3 \
  --batch-size 8 \
  --lr 5e-5 \
  --lora-r 16 \
  --lora-alpha 32.0 \
  --output-dir "./output/adapter"
```

| Flag | Default | Description |
|---|---|---|
| `--base` | *(required)* | HF repo identifier or local model path |
| `--data` | *(required)* | Path to JSONL or CSV dataset |
| `--task` | `classification` | Task type (`classification`, `nli`, `decision`) |
| `--epochs` | `3` | Number of training epochs |
| `--batch-size` | `8` | Batch size per optimization step |
| `--lr` | `5e-5` | AdamW learning rate |
| `--lora-r` | `16` | LoRA rank dimension |
| `--lora-alpha` | `32.0` | LoRA scaling factor |
| `--device` | `auto` | Execution device (`cpu`, `cuda`, `rocm`) |
| `--output-dir` | `./output` | Output directory for adapter weights |

---

### `trainaya merge`
Merges LoRA adapter weights directly into base model weights via SIMD vector kernels.

```bash
trainaya merge \
  --base "answerdotai/ModernBERT-base" \
  --lora "./output/adapter" \
  --output-dir "./output/merged"
```

Outputs standard `model.safetensors`, `config.json`, and label metadata mappings.

---

### `trainaya export`
Packages merged models with optional INT8 quantization and generates Ollaya runtime manifests.

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
Evaluates models against standard public decision datasets, computing Accuracy, Macro F1, Latency percentiles, and Expected Calibration Error (ECE).

```bash
# Comparative A/B run
trainaya bench --model winnow:v1 --baseline winnow:e4b

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
- `test_quant`: NF4 quantization/dequantization and SIMD LoRA merge precision.
- `test_dataset`: JSONL and CSV parsing, class mapping, and stratified train/val splits.
- `test_lora`: LoRA forward propagation, weight initialization, and zero-residual validation.
- `test_families`: Architecture family dispatch (ModernBERT, Winnow, Decider).
- `test_e2e_pipeline`: End-to-end synthetic training, adapter fusion, and SafeTensors/GGUF packaging.

---

## License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
