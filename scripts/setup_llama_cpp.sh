#!/usr/bin/env bash
# ==============================================================================
# Setup helper: Build optional llama.cpp native tools for GGUF LoRA merging
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENDOR_DIR="$PROJECT_ROOT/vendor/llama.cpp"

echo "==> Setting up llama.cpp tools for GGUF merging in $VENDOR_DIR..."

if [ ! -d "$VENDOR_DIR" ]; then
    echo "==> Cloning llama.cpp repository..."
    mkdir -p "$PROJECT_ROOT/vendor"
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "$VENDOR_DIR"
fi

echo "==> Building llama-export-lora and llama-quantize..."
cmake -B "$VENDOR_DIR/build" -S "$VENDOR_DIR" -DLLAMA_BUILD_EXAMPLES=ON
cmake --build "$VENDOR_DIR/build" --target llama-export-lora llama-quantize -j"$(nproc)"

echo "==> Done! Binaries available at $VENDOR_DIR/build/bin/"
