# Simple RunPod Setup (5 Minutes)

**Easiest way to deploy - no Docker building required!**

## Step 1: Login to RunPod

Go to: https://www.runpod.io/console/serverless

## Step 2: Create New Template

1. Click **"My Templates"** → **"New Template"**
2. Fill in these fields:

| Field | Value |
|-------|-------|
| Template Name | `pbr-inference-gpu` |
| Container Image | `runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04` |
| Container Disk | `20 GB` |
| Docker Command | Leave default or use: `python -u handler.py` |

3. Click **"Save Template"**

## Step 3: Create Endpoint from Template

1. Go to **"Serverless"** → **"Endpoints"**
2. Click **"New Endpoint"**
3. Select your template: **pbr-inference-gpu**
4. Configure:
   - **Name**: `pbr-inference-endpoint`
   - **Min Workers**: `0` (auto-scale to zero when idle = $0 cost)
   - **Max Workers**: `3`
   - **GPU Type**: Select **RTX 4090** (cheapest) or **RTX A4000**
   - **Idle Timeout**: `5 seconds`
5. Click **"Deploy"**

## Step 4: Get Your Endpoint ID

After deployment (takes 1-2 minutes):

1. You'll see your endpoint listed
2. Click on the endpoint name
3. Copy the **Endpoint ID** (looks like: `abc123xyz456...`)

**SEND ME THIS ENDPOINT ID** - I'll update the code!

---

## Wait, What About the Handler Code?

**Don't worry!** I'll provide you with an alternative approach:

Since RunPod templates are complex, I'll create a **GitHub Gist** or **Pastebin** link where you can:
1. Upload the handler code once
2. Have RunPod download it automatically on startup

This is **much simpler** than pasting code into their UI.

---

## Alternative: Let Me Build and Host the Image

If you want the **full Docker approach**, I can:
1. Build the Docker image
2. Push it to a public Docker Hub repository
3. Give you the image URL to paste into RunPod

**Which approach do you prefer?**

A) Simple template with startup script (I'll create the script for you)
B) I build the Docker image and host it for you
C) You want to build the Docker image yourself (I'll give detailed instructions)

Just let me know!

---

## Cost Reminder

- **RTX 4090**: ~$0.00011/second = **~$0.01 per asset**
- **Auto-scales to 0** when not in use = **$0 idle cost**
- Processing time: ~60-90 seconds per asset
