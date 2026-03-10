#!/usr/bin/env bash
set -euo pipefail

BACKEND="${1:-cuda}"
BUILD_ROOT="${2:-./env}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_ROOT="$(cd "$SCRIPT_DIR" && mkdir -p "$BUILD_ROOT" && cd "$BUILD_ROOT" && pwd)"

if [[ "$BACKEND" != "cuda" && "$BACKEND" != "rocm" ]]; then
    echo "Usage: $0 [cuda|rocm] [build_root]"
    exit 1
fi

echo "[INFO] Installing for backend: $BACKEND"
echo "[INFO] Build root: $BUILD_ROOT"

sudo apt-get update
sudo apt-get install -y \
    cmake \
    ffmpeg \
    git \
    libsm6 \
    libxext6

python -m pip install --upgrade pip
python -m pip install uv

uv_pip() {
    python -m uv pip install --system "$@"
}

if [[ "$BACKEND" == "rocm" ]]; then
    uv_pip torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
else
    uv_pip torch torchvision --index-url https://download.pytorch.org/whl/cu126
    uv_pip cupy-cuda12x==13.5.1 --no-cache-dir
fi

uv_pip -r "$SCRIPT_DIR/requirements.txt" --no-cache-dir

if [[ "$BACKEND" == "rocm" ]]; then
    FLASH_ATTN_DIR="$BUILD_ROOT/flash-attention"
    if [[ ! -d "$FLASH_ATTN_DIR/.git" ]]; then
        git clone https://github.com/Dao-AILab/flash-attention.git "$FLASH_ATTN_DIR"
    fi
    git -C "$FLASH_ATTN_DIR" fetch --tags
    git -C "$FLASH_ATTN_DIR" checkout v2.7.4
    (cd "$FLASH_ATTN_DIR" && python setup.py install)
else
    uv_pip flash-attn --no-build-isolation --no-cache-dir
fi

uv_pip "git+https://github.com/modelscope/DiffSynth-Studio.git@v1.1.9" --no-build-isolation
uv_pip "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation
uv_pip "git+https://github.com/facebookresearch/map-anything.git"

python -c "from mapanything.models import MapAnything; MapAnything.from_pretrained('facebook/map-anything')"
