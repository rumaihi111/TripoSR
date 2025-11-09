# RunPod GPU Endpoint Setup Instructions

This guide will help you deploy the PBR inference service on RunPod Serverless.

## Prerequisites

- RunPod account with API key (get yours at: https://www.runpod.io/console/user/settings)
- Docker Hub account (free) OR use RunPod's container registry

---

## Option 1: Use Pre-built Image (EASIEST - RECOMMENDED)

**This is the simplest option - no Docker knowledge required!**

### Step 1: Go to RunPod Dashboard

1. Open: https://www.runpod.io/console/serverless
2. Click **"My Templates"**
3. Click **"New Template"**

### Step 2: Fill in Template Form

```
Template Name: pbr-inference-gpu
Container Image: runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04
Container Disk: 20 GB
Docker Command: python -u /app/handler.py
Environment Variables: [Leave empty]
```

### Step 3: Add Start Command

In the template creation, find the **"Docker Command"** or **"Start Command"** field and paste:

```bash
bash -c "cd /workspace && cat > handler.py << 'HANDLER_EOF'
# [PASTE ENTIRE CONTENTS OF docker/runpod_handler.py HERE]
HANDLER_EOF
python -u handler.py"
```

**IMPORTANT**: You'll need to copy the **entire contents** of `docker/runpod_handler.py` and paste it where it says `[PASTE ENTIRE CONTENTS...]`

### Step 4: Create Endpoint

1. After template is created, go to **"Serverless"** → **"Endpoints"**
2. Click **"New Endpoint"**
3. Select your template: **pbr-inference-gpu**
4. Configuration:
   ```
   Name: pbr-inference-endpoint
   Min Workers: 0
   Max Workers: 3
   GPU Type: RTX A4000 or RTX 4090 (recommended)
   Idle Timeout: 5 seconds
   ```
5. Click **"Deploy"**

### Step 5: Get Endpoint ID

1. Once deployed, you'll see your endpoint
2. Copy the **Endpoint ID** (looks like: `abc123xyz456...`)
3. **SEND ME THIS ENDPOINT ID** so I can update the code

---

## Option 2: Manual Docker Build & Push

If RunPod doesn't have direct Dockerfile upload:

### Step 1: Build Docker Image Locally (Requires GPU Machine)

```bash
# You'll need to do this on a machine with Docker and GPU access
# Skip if using RunPod's builder

cd /workspaces/TripoSR/docker

# Build the image
docker build -t your-dockerhub-username/pbr-inference:latest .

# Login to Docker Hub
docker login

# Push to Docker Hub
docker push your-dockerhub-username/pbr-inference:latest
```

### Step 2: Create Template in RunPod

1. Go to https://www.runpod.io/console/serverless
2. **My Templates** → **New Template**
3. Fill in:
   ```
   Template Name: pbr-inference-gpu
   Container Image: your-dockerhub-username/pbr-inference:latest
   Container Disk: 20 GB
   ```

### Step 3: Create Endpoint (same as Option 1, Step 4-5)

---

## Option 3: Use Pre-built Image (Fastest)

If you don't want to build the Docker image yourself, I can provide a simplified setup:

1. Go to RunPod → **New Template**
2. Use this container image: `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`
3. In the template, add a **Startup Script**:
   ```bash
   pip3 install torch torchvision opencv-python numpy runpod requests
   ```
4. Create endpoint with this template

**Then send me the Endpoint ID**

---

## What I Need From You

Once you complete the deployment, send me:

1. ✅ **Endpoint ID** (looks like: `abcd1234-5678-90ef-ghij-klmnopqrstuv`)
2. Your preference for which option you used (1, 2, or 3)

Then I'll:
- Update the code with your endpoint ID
- Run a test to verify GPU processing works
- Show you the quality difference between CPU and GPU!

---

## Cost Estimate

- **RTX A4000**: ~$0.00019/second = ~$0.02 per asset
- **RTX 4090**: ~$0.00011/second = ~$0.01 per asset
- **Idle time**: $0 (auto-scales to 0)

Processing time: ~60-90 seconds per asset

---

## Need Help?

If you encounter any issues during deployment, let me know:
- Screenshot of any errors
- Which option you're trying
- Where you got stuck

I'll guide you through it!
