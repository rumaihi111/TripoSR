#!/usr/bin/env bash
set -euo pipefail

echo "[postCreate] Installing system build tools…"
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  git ffmpeg libgl1 libglib2.0-0 libx11-dev \
  build-essential cmake ninja-build pkg-config \
  python3-dev clang

echo "[postCreate] Upgrade pip…"
python -m pip install --upgrade pip

echo "[postCreate] Install CPU PyTorch…"
python -m pip install --no-cache-dir \
  torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "[postCreate] Base Python deps…"
python -m pip install --no-cache-dir \
  omegaconf==2.3.0 Pillow==10.1.0 einops==0.7.0 \
  transformers==4.35.0 trimesh==4.0.5 rembg imageio \
  huggingface-hub onnxruntime

echo "[postCreate] Build helpers for native wheels…"
python -m pip install --no-cache-dir -U scikit-build-core pybind11

echo "[postCreate] Install torchmcubes from GitHub (no PyPI wheel)…"
python -m pip install --no-cache-dir git+https://github.com/tatsy/torchmcubes.git

echo "[postCreate] Install xatlas (try wheel, else source)…"
if ! python -m pip install --no-cache-dir xatlas==0.0.9 --only-binary=:all: ; then
  export CC=gcc CXX=g++
  python -m pip install --no-cache-dir --no-binary xatlas xatlas==0.0.9 -v
fi

echo "[postCreate] Done."
