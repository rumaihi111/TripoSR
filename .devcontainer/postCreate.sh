#!/usr/bin/env bash
set -e

# System packages needed to compile TripoSR deps
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  git ffmpeg libgl1 libglib2.0-0 libx11-dev \
  build-essential cmake ninja-build

# Python deps
pip install --upgrade pip

# CPU PyTorch wheels
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# TripoSR deps (minimal CLI set)
pip install --no-cache-dir \
  omegaconf==2.3.0 Pillow==10.1.0 einops==0.7.0 \
  transformers==4.35.0 trimesh==4.0.5 rembg huggingface-hub imageio \
  torchmcubes==0.1.0 xatlas==0.0.9 onnxruntime
