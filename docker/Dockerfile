# RunPod Serverless Dockerfile for PBR Inference
# Optimized for CUDA GPU processing

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cu118

# Install Python dependencies for PBR inference
RUN pip3 install --no-cache-dir \
    runpod==1.5.0 \
    opencv-python==4.8.1.78 \
    numpy==1.24.3 \
    requests==2.31.0 \
    pillow==10.1.0

# Create app directory
WORKDIR /app

# Copy handler script
COPY runpod_handler.py /app/handler.py

# Test imports
RUN python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
RUN python3 -c "import cv2; import numpy; import runpod; print('All dependencies OK')"

# Set the entrypoint
CMD ["python3", "-u", "handler.py"]
