#!/usr/bin/env bash
# ==============================================================================
# Trainaya One-Line Installer
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/solarpush/trainaya/main/install.sh | sh
# ==============================================================================
set -e

REPO="solarpush/trainaya"
BINARY_NAME="trainaya"

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${CYAN}${BOLD}"
echo "============================================================"
echo "  ⚡ Trainaya: Decision Model Adaptation, Curation & Training"
echo "============================================================"
echo -e "${NC}"

# 1. Detect Operating System
OS="$(uname -s)"
case "$OS" in
    Linux*)     PLATFORM="linux" ;;
    Darwin*)    PLATFORM="darwin" ;;
    *)          
        echo -e "${RED}❌ Unsupported operating system: $OS${NC}"
        echo "Trainaya currently supports Linux and macOS (Darwin)."
        exit 1
        ;;
esac

# 2. Detect Architecture
ARCH="$(uname -m)"
case "$ARCH" in
    x86_64|amd64)   TARGET_ARCH="x86_64" ;;
    arm64|aarch64)  TARGET_ARCH="aarch64" ;;
    *)
        echo -e "${RED}❌ Unsupported architecture: $ARCH${NC}"
        echo "Trainaya supports x86_64 and aarch64 (ARM64)."
        exit 1
        ;;
esac

echo -e "  • Detected System : ${BOLD}${PLATFORM} (${TARGET_ARCH})${NC}"

# 3. Determine Installation Directory
if [ -n "${TRAINAYA_INSTALL_DIR:-}" ]; then
    INSTALL_DIR="$TRAINAYA_INSTALL_DIR"
elif [ "$(id -u)" -eq 0 ]; then
    INSTALL_DIR="/usr/local/bin"
else
    INSTALL_DIR="$HOME/.local/bin"
fi

mkdir -p "$INSTALL_DIR"
echo -e "  • Target Directory: ${BOLD}${INSTALL_DIR}${NC}"

# 4. Ensure 'uv' is installed for the embedded PyTorch engine
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ] && [ ! -x "$HOME/.cargo/bin/uv" ]; then
    echo -e "\n  ${BLUE}==>${NC} Installing 'uv' (Python & PyTorch runtime manager)..."
    if curl -LsSf https://astral.sh/uv/install.sh 2>/dev/null | sh; then
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    else
        echo -e "  • Note: Could not reach astral.sh (offline/sandboxed); relying on local Python environment."
    fi
else
    echo -e "  • Python Manager  : ${BOLD}uv detected${NC}"
fi

# 5. Installation Method: Check if local built binary exists or fetch from GitHub
INSTALLED=false

# Case A: Local repository checkout with pre-existing release binary
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd || echo "")"
if [ -n "$SCRIPT_DIR" ] && [ -f "$SCRIPT_DIR/target/release/$BINARY_NAME" ]; then
    echo -e "\n  ${BLUE}==>${NC} Installing from local target/release build..."
    cp "$SCRIPT_DIR/target/release/$BINARY_NAME" "$INSTALL_DIR/$BINARY_NAME"
    chmod +x "$INSTALL_DIR/$BINARY_NAME"
    if [ -d "$SCRIPT_DIR/.venv" ]; then
        mkdir -p "$HOME/.cache/trainaya/py_engine"
        if [ ! -e "$HOME/.cache/trainaya/py_engine/.venv" ]; then
            ln -s "$SCRIPT_DIR/.venv" "$HOME/.cache/trainaya/py_engine/.venv" 2>/dev/null || true
        fi
    fi
    INSTALLED=true
fi

# Case B: Download prebuilt release asset from GitHub
if [ "$INSTALLED" = false ]; then
    RELEASE_TAG="latest"
    ASSET_NAME="${BINARY_NAME}-${PLATFORM}-${TARGET_ARCH}.tar.gz"
    DOWNLOAD_URL="https://github.com/${REPO}/releases/${RELEASE_TAG}/download/${ASSET_NAME}"

    echo -e "\n  ${BLUE}==>${NC} Fetching prebuilt release from GitHub (${DOWNLOAD_URL})..."
    
    TMP_DIR="$(mktemp -d)"
    trap 'rm -rf "$TMP_DIR"' EXIT

    if curl -fsSL "$DOWNLOAD_URL" -o "$TMP_DIR/$ASSET_NAME" 2>/dev/null; then
        echo -e "  • Unpacking release archive..."
        tar -xzf "$TMP_DIR/$ASSET_NAME" -C "$TMP_DIR"
        cp -f "$TMP_DIR/$BINARY_NAME" "$INSTALL_DIR/$BINARY_NAME"
        chmod +x "$INSTALL_DIR/$BINARY_NAME"
        INSTALLED=true
    else
        echo -e "${RED}❌ Failed to download prebuilt binary from GitHub Releases.${NC}"
        echo "Release asset '${ASSET_NAME}' was not found at:"
        echo "  ${DOWNLOAD_URL}"
        echo ""
        echo "Please check available releases at:"
        echo "  https://github.com/${REPO}/releases"
        exit 1
    fi
fi

# 6. Check PATH environment
IN_PATH=false
case ":$PATH:" in
    *":$INSTALL_DIR:"*) IN_PATH=true ;;
esac

echo -e "\n${GREEN}${BOLD}✓ Trainaya successfully installed in $INSTALL_DIR/$BINARY_NAME!${NC}\n"

if [ "$IN_PATH" = false ]; then
    echo -e "${CYAN}⚠️  Note: $INSTALL_DIR is not in your current PATH.${NC}"
    echo "Add the following line to your ~/.bashrc or ~/.zshrc:"
    echo -e "  ${BOLD}export PATH=\"$INSTALL_DIR:\$PATH\"${NC}\n"
    export PATH="$INSTALL_DIR:$PATH"
fi

# 7. Verify Installation with doctor
if command -v "$BINARY_NAME" >/dev/null 2>&1; then
    "$BINARY_NAME" doctor
else
    "$INSTALL_DIR/$BINARY_NAME" doctor
fi

echo -e "${GREEN}${BOLD}🚀 All set! Run 'trainaya curate' or 'trainaya --help' to get started.${NC}\n"
