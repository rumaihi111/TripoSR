#!/bin/bash
# RunPod Startup Script
# This downloads and runs the PBR inference handler

echo "Installing dependencies..."
pip3 install opencv-python-headless numpy runpod requests --quiet

echo "Downloading handler code..."
cd /workspace
cat > handler.py << 'EOF'
