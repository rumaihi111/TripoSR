"""
RunPod Serverless Handler for PBR Material Inference
Base64 encoding - no external file hosting needed
"""

import runpod
import os
import base64
import time
from pathlib import Path
import cv2
import numpy as np


def decode_base64_to_file(base64_string, output_path):
    """Decode base64 and save"""
    file_bytes = base64.b64decode(base64_string)
    with open(output_path, 'wb') as f:
        f.write(file_bytes)
    return len(file_bytes) / 1024


def encode_file_to_base64(file_path):
    """Encode file as base64"""
    with open(file_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def generate_pbr_textures(image_path, output_dir, texture_size=2048):
    """Generate PBR textures using GPU-accelerated OpenCV"""
    
    print(f"  🎨 Generating PBR textures ({texture_size}x{texture_size})")
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    img = cv2.resize(img, (texture_size, texture_size))
    
    os.makedirs(output_dir, exist_ok=True)
    textures = {}
    
    # Albedo
    print("     → Albedo...")
    img_float = img.astype(np.float32) / 255.0
    scales = [15, 80, 250]
    msr = np.zeros_like(img_float)
    for scale in scales:
        blurred = cv2.GaussianBlur(img_float, (0, 0), scale)
        msr += np.log10(img_float + 1e-4) - np.log10(blurred + 1e-4)
    albedo = np.exp(msr / len(scales))
    albedo = np.clip(albedo * 255, 0, 255).astype(np.uint8)
    albedo_path = f"{output_dir}/albedo.png"
    cv2.imwrite(albedo_path, albedo)
    textures['albedo'] = albedo_path
    
    # Roughness
    print("     → Roughness...")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    gradient = np.sqrt(gx**2 + gy**2)
    roughness = 1.0 - (gradient / (gradient.max() + 1e-6))
    roughness = (roughness * 0.65 + 0.3) * 255
    roughness_path = f"{output_dir}/roughness.png"
    cv2.imwrite(roughness_path, roughness.astype(np.uint8))
    textures['roughness'] = roughness_path
    
    # Metallic
    print("     → Metallic...")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    metallic = np.zeros_like(hsv[:,:,2], dtype=np.float32)
    bright = hsv[:,:,2] > 180
    metallic[bright] = (255 - hsv[:,:,1][bright]) / 255.0
    metallic_path = f"{output_dir}/metallic.png"
    cv2.imwrite(metallic_path, (metallic * 255).astype(np.uint8))
    textures['metallic'] = metallic_path
    
    # Normal
    print("     → Normal map...")
    dx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=5)
    dy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=5)
    normal = np.zeros((*gray.shape, 3), dtype=np.float32)
    normal[:,:,0] = -dx / 255.0
    normal[:,:,1] = -dy / 255.0
    normal[:,:,2] = 1.0
    length = np.sqrt(np.sum(normal**2, axis=2, keepdims=True))
    normal = normal / (length + 1e-6)
    normal = ((normal + 1.0) * 0.5 * 255).astype(np.uint8)
    normal_path = f"{output_dir}/normal.png"
    cv2.imwrite(normal_path, normal)
    textures['normal'] = normal_path
    
    # AO
    print("     → Ambient occlusion...")
    ao = 255 - gray
    ao = cv2.GaussianBlur(ao, (21, 21), 0)
    ao = cv2.normalize(ao, None, 50, 255, cv2.NORM_MINMAX)
    ao_path = f"{output_dir}/ao.png"
    cv2.imwrite(ao_path, ao)
    textures['ao'] = ao_path
    
    return textures


def handler(job):
    """Main RunPod handler - receives base64 encoded files, returns base64 results"""
    try:
        print("\n" + "="*60)
        print("🚀 RunPod PBR Job Started")
        print("="*60)
        start = time.time()
        
        inp = job['input']
        image_b64 = inp.get('image_b64')
        mesh_b64 = inp.get('mesh_b64')
        tex_size = inp.get('texture_size', 2048)
        
        if not image_b64 or not mesh_b64:
            raise ValueError("Missing image_b64 or mesh_b64 in input")
        
        print(f"  📦 Decoding inputs...")
        print(f"     Image: {len(image_b64):,} chars")
        print(f"     Mesh: {len(mesh_b64):,} chars")
        print(f"     Texture size: {tex_size}x{tex_size}")
        
        # Decode inputs
        temp = "/tmp/pbr_job"
        os.makedirs(temp, exist_ok=True)
        
        img_path = f"{temp}/input.jpg"
        mesh_path = f"{temp}/input.obj"
        
        img_kb = decode_base64_to_file(image_b64, img_path)
        mesh_kb = decode_base64_to_file(mesh_b64, mesh_path)
        
        print(f"  ✓ Decoded: {img_kb:.1f} KB image, {mesh_kb:.1f} KB mesh")
        
        # Generate textures
        out_dir = f"{temp}/output"
        textures = generate_pbr_textures(img_path, out_dir, tex_size)
        
        # Encode results
        print(f"  📤 Encoding results...")
        encoded = {}
        for name, path in textures.items():
            encoded[name] = encode_file_to_base64(path)
            size = os.path.getsize(path) / 1024
            print(f"     ✓ {name}.png ({size:.1f} KB)")
        
        elapsed = time.time() - start
        print(f"\n✅ Job Complete!")
        print(f"   GPU time: {elapsed:.1f}s")
        print(f"   Cost: ${elapsed * 0.00039:.4f}")
        print("="*60 + "\n")
        
        return {
            "success": True,
            "textures": encoded,
            "execution_time": elapsed
        }
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"\n❌ Error: {e}")
        print(error_trace)
        print("="*60 + "\n")
        return {
            "success": False,
            "error": str(e),
            "traceback": error_trace
        }


# Start the RunPod serverless handler
if __name__ == "__main__":
    print("🔧 RunPod PBR Handler Starting...")
    runpod.serverless.start({"handler": handler})