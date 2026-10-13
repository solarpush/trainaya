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
echo "  🚀 Trainaya: High-Speed Model Fine-Tuning & Distillation"
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
if [ "$(id -u)" -eq 0 ]; then
    INSTALL_DIR="/usr/local/bin"
else
    INSTALL_DIR="$HOME/.local/bin"
fi

mkdir -p "$INSTALL_DIR"
echo -e "  • Target Directory: ${BOLD}${INSTALL_DIR}${NC}"

# 4. Installation Method: Check if local built binary exists or fetch from GitHub
INSTALLED=false

# Case A: Local repository checkout with pre-existing release binary
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd || echo "")"
if [ -n "$SCRIPT_DIR" ] && [ -f "$SCRIPT_DIR/target/release/$BINARY_NAME" ]; then
    echo -e "\n  ${BLUE}==>${NC} Installing from local target/release build..."
    cp "$SCRIPT_DIR/target/release/$BINARY_NAME" "$INSTALL_DIR/$BINARY_NAME"
    chmod +x "$INSTALL_DIR/$BINARY_NAME"
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
        cp "$TMP_DIR/$BINARY_NAME" "$INSTALL_DIR/$BINARY_NAME"
        chmod +x "$INSTALL_DIR/$BINARY_NAME"
        INSTALLED=true
    else
        echo -e "  ${CYAN}Notice:${NC} Prebuilt asset not found on GitHub releases."
        
        # Case C: Fallback to local Cargo compilation if Rust is installed
        if command -v cargo >/dev/null 2>&1; then
            echo -e "  ${BLUE}==>${NC} Rust & Cargo detected. Compiling native release binary..."
            if [ -n "$SCRIPT_DIR" ] && [ -f "$SCRIPT_DIR/Cargo.toml" ]; then
                cargo build --release --manifest-path "$SCRIPT_DIR/Cargo.toml"
                cp "$SCRIPT_DIR/target/release/$BINARY_NAME" "$INSTALL_DIR/$BINARY_NAME"
                chmod +x "$INSTALL_DIR/$BINARY_NAME"
                INSTALLED=true
            else
                cargo install --git "https://github.com/${REPO}.git" --root "$HOME/.local"
                INSTALLED=true
            fi
        else
            echo -e "${RED}❌ Unable to install prebuilt release and Cargo is not installed.${NC}"
            echo "Please install Rust/Cargo (https://rustup.rs) or download a release from:"
            echo "https://github.com/${REPO}/releases"
            exit 1
        fi
    fi
fi

# 5. Check PATH environment
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

# 6. Verify Installation with doctor
if command -v "$BINARY_NAME" >/dev/null 2>&1; then
    "$BINARY_NAME" doctor
else
    "$INSTALL_DIR/$BINARY_NAME" doctor
fi

echo -e "${GREEN}${BOLD}🚀 All set! You can now run 'trainaya --help'${NC}\n"
