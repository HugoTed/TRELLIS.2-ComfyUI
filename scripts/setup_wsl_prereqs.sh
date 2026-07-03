#!/usr/bin/env bash
# One-time system prerequisites for TRELLIS.2-ComfyUI on WSL2 Ubuntu.
# Everything the Setup node CANNOT do itself (requires sudo).
# Idempotent — safe to re-run.
#
# Usage:
#   sudo bash scripts/setup_wsl_prereqs.sh
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "Please run with sudo: sudo bash scripts/setup_wsl_prereqs.sh" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_VER="${TRELLIS2_CUDA_VER:-12-8}"          # apt package suffix, e.g. 12-8
CUDA_HOME_GUESS="/usr/local/cuda-${CUDA_VER//-/.}"

echo "=== [1/5] Base build tools + Python venv support ==="
apt-get update
apt-get install -y git build-essential python3-venv python3-pip

echo "=== [2/5] GCC 13 (CUDA 12.x rejects newer default GCC on Ubuntu 25.04+) ==="
if ! command -v gcc-13 >/dev/null 2>&1; then
    apt-get install -y gcc-13 g++-13
else
    echo "gcc-13 already installed"
fi

echo "=== [3/5] CUDA toolkit (nvcc + dev headers) ==="
if ! command -v nvcc >/dev/null 2>&1 && [[ ! -x "${CUDA_HOME_GUESS}/bin/nvcc" ]]; then
    if ! apt-get install -y "cuda-nvcc-${CUDA_VER}" "cuda-cudart-dev-${CUDA_VER}" "cuda-libraries-dev-${CUDA_VER}"; then
        echo "CUDA packages not found in current apt sources; adding NVIDIA CUDA WSL repo..."
        KEYRING=/tmp/cuda-keyring.deb
        wget -qO "$KEYRING" https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
        dpkg -i "$KEYRING"
        apt-get update
        apt-get install -y "cuda-nvcc-${CUDA_VER}" "cuda-cudart-dev-${CUDA_VER}" "cuda-libraries-dev-${CUDA_VER}"
    fi
else
    echo "nvcc already available"
fi

echo "=== [4/5] Patch CUDA headers for glibc >= 2.38 (no-op if not needed) ==="
python3 "${SCRIPT_DIR}/patch_cuda_math_functions.py" --ensure || {
    echo "Header patch failed — check output above" >&2
    exit 1
}

echo "=== [5/5] Sanity check ==="
NVCC_BIN="$(command -v nvcc || echo "${CUDA_HOME_GUESS}/bin/nvcc")"
"$NVCC_BIN" --version | tail -1
gcc-13 --version | head -1
echo
echo "All system prerequisites ready."
echo
echo "Next steps:"
echo "  1. Restart ComfyUI (recommended flags on WSL: --disable-pinned-memory)"
echo "  2. Run the 'TRELLIS.2 Setup (Install Worker)' node once (or just run Image to 3D)"
echo
echo "WSL/Windows memory tips (important for large generations):"
echo "  - Keep .wslconfig memory at ~50% of physical RAM (e.g. memory=32GB on a 64GB box)."
echo "    GPU allocations from WSL require Windows-side commit backing; starving Windows"
echo "    causes spurious CUDA OOM with plenty of free VRAM."
echo "  - Set a generous Windows pagefile (e.g. initial 32GB / max 64GB)."
