#!/usr/bin/env bash
# ==============================================================================
# Setup helper: Build optional llama.cpp native tools for GGUF LoRA merging
# Automatically enables GPU backends (ROCm/HIP, Vulkan, CUDA, or Metal)
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

CMAKE_GPU_FLAGS=""
OS="$(uname -s)"

if [ "$OS" = "Darwin" ]; then
    echo "==> Detected macOS: enabling Apple Silicon Metal acceleration (-DGGML_METAL=ON)"
    CMAKE_GPU_FLAGS="-DGGML_METAL=ON"
elif command -v nvcc >/dev/null 2>&1; then
    echo "==> Detected NVIDIA CUDA toolkit: enabling CUDA acceleration (-DGGML_CUDA=ON)"
    CMAKE_GPU_FLAGS="-DGGML_CUDA=ON"
elif command -v hipcc >/dev/null 2>&1; then
    echo "==> Detected AMD ROCm / HIP toolkit: enabling HIP acceleration (-DGGML_HIP=ON)"
    CMAKE_GPU_FLAGS="-DGGML_HIP=ON"
elif command -v vulkaninfo >/dev/null 2>&1 || [ -f "/usr/include/vulkan/vulkan.h" ]; then
    echo "==> Detected Vulkan SDK: enabling Vulkan GPU acceleration (-DGGML_VULKAN=ON)"
    CMAKE_GPU_FLAGS="-DGGML_VULKAN=ON"
elif [ -e "/dev/kfd" ]; then
    echo "==> Detected AMD GPU (/dev/kfd): checking for Vulkan headers..."
    if [ -f "/usr/include/vulkan/vulkan.h" ]; then
        CMAKE_GPU_FLAGS="-DGGML_VULKAN=ON"
    fi
fi

echo "==> Building llama-export-lora and llama-quantize ($CMAKE_GPU_FLAGS)..."
cmake -B "$VENDOR_DIR/build" -S "$VENDOR_DIR" -DLLAMA_BUILD_EXAMPLES=ON $CMAKE_GPU_FLAGS
cmake --build "$VENDOR_DIR/build" --target llama-export-lora llama-quantize -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"

echo "==> Done! Binaries available at $VENDOR_DIR/build/bin/"
