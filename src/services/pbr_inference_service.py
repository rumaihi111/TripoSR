"""
Stage 3: PBR Material Inference Service

This service generates realistic PBR (Physically Based Rendering) materials from:
- Original input photo (ground truth for material properties)
- Cleaned mesh from Stage 2

Supports:
- RunPod GPU backend (nvdiffrec-quality, ~$0.01/asset)
- CPU-only fallback (OpenCV, free but lower quality)

PBR Maps Generated:
- Albedo (base color without lighting)
- Roughness (glossy vs matte)
- Metallic (metal vs dielectric)
- Normal (surface detail)
- Ambient Occlusion (shadows in crevices)
"""

import os
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, Optional

import numpy as np


class PBRInferenceService:
    """
    Stage 3: PBR Material Inference
    Uses RunPod GPU or falls back to CPU-only processing
    """
    
    def __init__(self, backend: str = "auto"):
        """
        Initialize PBR inference service
        
        Args:
            backend: "auto" (try GPU, fallback to CPU), "gpu", or "cpu"
        """
        self.backend = backend
        self.api_key = os.getenv("RUNPOD_API_KEY")
        self.endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID")
        
        if backend == "auto":
            if self.api_key:
                print("✓ RunPod API key found - GPU backend available")
                self.use_gpu = True
            else:
                print("⚠️  No RunPod API key - using CPU fallback")
                self.use_gpu = False
        elif backend == "gpu":
            if not self.api_key:
                raise ValueError("GPU backend requested but RUNPOD_API_KEY not set")
            self.use_gpu = True
        else:
            self.use_gpu = False
    
    def infer_pbr_materials(
        self,
        input_image_path: str,
        mesh_path: str,
        output_dir: str,
        config: Optional[Dict] = None
    ) -> Dict:
        """
        Generate PBR materials from original photo and mesh
        
        Args:
            input_image_path: Original photograph (NOT mesh texture!)
            mesh_path: Cleaned mesh from Stage 2
            output_dir: Where to save PBR textures
            config: Optional configuration dict
            
        Returns:
            Dict with:
                - success: bool
                - backend: "runpod_gpu" or "cpu_opencv"
                - textures: dict of texture paths
                - cost_usd: processing cost
                - gpu_time_seconds: processing time
        """
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        config = config or self._get_default_config()
        
        print("\n" + "="*70)
        print("🎨 STAGE 3: PBR MATERIAL INFERENCE")
        print("="*70)
        print(f"Input Photo: {input_image_path}")
        print(f"Mesh: {mesh_path}")
        print(f"Output: {output_dir}")
        print(f"Backend: {'GPU (RunPod)' if self.use_gpu else 'CPU (OpenCV)'}")
        print("="*70 + "\n")
        
        # Validate inputs
        if not Path(input_image_path).exists():
            return {
                "success": False,
                "error": f"Input image not found: {input_image_path}"
            }
        
        if not Path(mesh_path).exists():
            return {
                "success": False,
                "error": f"Mesh not found: {mesh_path}"
            }
        
        # Choose backend
        if self.use_gpu:
            try:
                return self._runpod_processing(
                    input_image_path, 
                    mesh_path, 
                    output_path, 
                    config
                )
            except Exception as e:
                print(f"❌ GPU processing failed: {e}")
                print("💻 Falling back to CPU processing...\n")
                return self._cpu_fallback(input_image_path, mesh_path, output_path, config)
        else:
            return self._cpu_fallback(input_image_path, mesh_path, output_path, config)
    
    def _get_default_config(self) -> Dict:
        """Default configuration"""
        return {
            "texture_size": 2048,
            "ao_samples": 128,
            "normal_strength": 1.0,
            "base_roughness": 0.5,
            "base_metallic": 0.0
        }
    
    def _runpod_processing(
        self,
        image_path: str,
        mesh_path: str,
        output_path: Path,
        config: Dict
    ) -> Dict:
        """
        Process using RunPod GPU (requires runpod package)
        """
        
        try:
            import runpod
            runpod.api_key = self.api_key
        except ImportError:
            raise ImportError(
                "runpod package not installed. "
                "Install with: pip install runpod"
            )
        
        print("☁️  RunPod GPU Backend")
        print("-" * 70)
        
        # Upload files to temporary storage
        print("📤 Uploading files to cloud storage...")
        image_url = self._upload_file(image_path)
        mesh_url = self._upload_file(mesh_path)
        
        print("\n🚀 Starting GPU processing...")
        print(f"   Texture resolution: {config['texture_size']}x{config['texture_size']}")
        
        if not self.endpoint_id:
            print("\n⚠️  RUNPOD_ENDPOINT_ID not set")
            print("   For now, using CPU fallback")
            print("   To use GPU: Create RunPod endpoint and set RUNPOD_ENDPOINT_ID")
            raise Exception("No RunPod endpoint configured")
        
        endpoint = runpod.Endpoint(self.endpoint_id)
        
        job_input = {
            "image_url": image_url,
            "mesh_url": mesh_url,
            "texture_size": config["texture_size"],
            "method": "nvdiffrec"
        }
        
        run_request = endpoint.run(job_input)
        
        # Poll for completion
        print("⏳ Processing on GPU...")
        start_time = time.time()
        
        while True:
            status = run_request.status()
            elapsed = int(time.time() - start_time)
            
            if status == "COMPLETED":
                result = run_request.output()
                break
            elif status == "FAILED":
                error = run_request.output()
                raise Exception(f"GPU job failed: {error}")
            
            print(f"   Status: {status} ({elapsed}s elapsed)")
            time.sleep(5)
            
            if elapsed > 600:
                raise TimeoutError("GPU processing timed out after 10 minutes")
        
        # Download results
        print("\n📥 Downloading PBR textures...")
        textures = {}
        
        for texture_name, texture_url in result.get("textures", {}).items():
            texture_path = output_path / f"{texture_name}.png"
            self._download_file(texture_url, texture_path)
            textures[texture_name] = str(texture_path)
            print(f"   ✓ {texture_name}.png")
        
        gpu_time = time.time() - start_time
        cost = gpu_time * 0.00011  # RTX 4090 pricing
        
        print(f"\n✅ GPU processing complete!")
        print(f"   Processing time: {gpu_time:.1f}s")
        print(f"   Cost: ${cost:.4f}")
        
        return {
            "success": True,
            "backend": "runpod_gpu",
            "textures": textures,
            "gpu_time_seconds": gpu_time,
            "cost_usd": cost
        }
    
    def _cpu_fallback(
        self,
        image_path: str,
        mesh_path: str,
        output_path: Path,
        config: Dict
    ) -> Dict:
        """
        CPU-only fallback using OpenCV
        Lower quality but requires no external services
        """
        
        print("💻 CPU-Only Backend (OpenCV)")
        print("-" * 70)
        print("Using computer vision for PBR inference")
        print("Note: GPU backend provides higher quality results\n")
        
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "opencv-python package not installed. "
                "Install with: pip install opencv-python"
            )
        
        img = cv2.imread(image_path)
        if img is None:
            return {
                "success": False,
                "error": f"Failed to load image: {image_path}"
            }
        
        textures = {}
        
        # 1. Albedo (lighting separation)
        print("1/5 Generating albedo (base color)...")
        albedo = self._retinex_decomposition(img)
        albedo_path = output_path / "albedo.png"
        cv2.imwrite(str(albedo_path), albedo)
        textures["albedo"] = str(albedo_path)
        
        # 2. Roughness
        print("2/5 Generating roughness map...")
        roughness = self._generate_roughness_cpu(img, config)
        roughness_path = output_path / "roughness.png"
        cv2.imwrite(str(roughness_path), roughness)
        textures["roughness"] = str(roughness_path)
        
        # 3. Metallic
        print("3/5 Generating metallic map...")
        metallic = self._generate_metallic_cpu(img, config)
        metallic_path = output_path / "metallic.png"
        cv2.imwrite(str(metallic_path), metallic)
        textures["metallic"] = str(metallic_path)
        
        # 4. Normal map
        print("4/5 Generating normal map...")
        normal = self._generate_normal_cpu(img, config)
        normal_path = output_path / "normal.png"
        cv2.imwrite(str(normal_path), normal)
        textures["normal"] = str(normal_path)
        
        # 5. Ambient Occlusion
        print("5/5 Generating ambient occlusion...")
        if mesh_path and Path(mesh_path).exists():
            try:
                ao_path = self._bake_ao_blender(mesh_path, output_path / "ao.png", config)
                textures["ao"] = str(ao_path)
            except Exception as e:
                print(f"   ⚠️  Blender AO baking failed: {e}")
                print("   Using image-based AO approximation...")
                ao = self._generate_fake_ao(img)
                ao_path = output_path / "ao.png"
                cv2.imwrite(str(ao_path), ao)
                textures["ao"] = str(ao_path)
        else:
            ao = self._generate_fake_ao(img)
            ao_path = output_path / "ao.png"
            cv2.imwrite(str(ao_path), ao)
            textures["ao"] = str(ao_path)
        
        print("\n✅ CPU inference complete!")
        print("   Note: Consider using RunPod GPU for higher quality")
        
        return {
            "success": True,
            "backend": "cpu_opencv",
            "textures": textures,
            "cost_usd": 0.0,
            "gpu_time_seconds": 0.0
        }
    
    # ==== CPU Processing Methods ====
    
    def _retinex_decomposition(self, img: np.ndarray) -> np.ndarray:
        """
        Multi-scale Retinex algorithm for lighting separation
        Separates albedo (material color) from shading (lighting)
        """
        import cv2
        
        img_float = img.astype(np.float32) / 255.0
        scales = [15, 80, 250]
        msr = np.zeros_like(img_float)
        
        for scale in scales:
            blurred = cv2.GaussianBlur(img_float, (0, 0), scale)
            msr += np.log10(img_float + 1e-4) - np.log10(blurred + 1e-4)
        
        msr = msr / len(scales)
        albedo = np.exp(msr)
        albedo = np.clip(albedo * 255, 0, 255).astype(np.uint8)
        
        return albedo
    
    def _generate_roughness_cpu(self, img: np.ndarray, config: Dict) -> np.ndarray:
        """
        Infer roughness from specular highlight sharpness
        High gradient = sharp reflection = low roughness (glossy)
        Low gradient = diffuse = high roughness (matte)
        """
        import cv2
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize
        gradient_norm = gradient / (gradient.max() + 1e-6)
        
        # Invert: high gradient = low roughness
        roughness = 1.0 - gradient_norm
        
        # Blend with base roughness
        base_roughness = config.get("base_roughness", 0.5)
        roughness = roughness * 0.4 + base_roughness * 0.6
        
        # Convert to 8-bit and smooth
        roughness_map = (roughness * 255).astype(np.uint8)
        roughness_map = cv2.GaussianBlur(roughness_map, (9, 9), 0)
        
        return roughness_map
    
    def _generate_metallic_cpu(self, img: np.ndarray, config: Dict) -> np.ndarray:
        """
        Infer metallic from reflection properties
        Metals have colored reflections, dielectrics have white specular
        """
        import cv2
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        value = hsv[:, :, 2]
        saturation = hsv[:, :, 1]
        
        # Bright areas with low saturation = potential metallic
        bright_mask = value > 180
        metallic = np.zeros_like(value, dtype=np.float32)
        metallic[bright_mask] = (255 - saturation[bright_mask]) / 255.0
        
        # Apply base metallic
        base_metallic = config.get("base_metallic", 0.0)
        if base_metallic > 0.1:
            metallic = metallic * 0.5 + base_metallic * 0.5
        
        # Convert and smooth
        metallic_map = (metallic * 255).astype(np.uint8)
        metallic_map = cv2.GaussianBlur(metallic_map, (11, 11), 0)
        
        return metallic_map
    
    def _generate_normal_cpu(self, img: np.ndarray, config: Dict) -> np.ndarray:
        """
        Generate normal map from image gradients
        Captures surface detail from photo
        """
        import cv2
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # Sobel derivatives
        dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        # Create normal map (tangent space)
        normal = np.zeros((*gray.shape, 3), dtype=np.float32)
        normal_strength = config.get("normal_strength", 1.0)
        normal[:, :, 0] = -dx / 255.0 * normal_strength  # Red (X)
        normal[:, :, 1] = -dy / 255.0 * normal_strength  # Green (Y)
        normal[:, :, 2] = 1.0                             # Blue (Z)
        
        # Normalize vectors
        length = np.sqrt(np.sum(normal**2, axis=2, keepdims=True))
        normal = normal / (length + 1e-6)
        
        # Convert to 0-255 range (128 = neutral)
        normal = ((normal + 1.0) * 0.5 * 255).astype(np.uint8)
        
        return normal
    
    def _generate_fake_ao(self, img: np.ndarray) -> np.ndarray:
        """
        Generate approximate AO from image darkness
        Not as accurate as geometry-based AO but better than nothing
        """
        import cv2
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Dark areas approximate occlusion
        ao = 255 - gray
        ao = cv2.GaussianBlur(ao, (15, 15), 0)
        ao = cv2.normalize(ao, None, 50, 255, cv2.NORM_MINMAX)
        
        return ao
    
    def _bake_ao_blender(self, mesh_path: str, output_path: Path, config: Dict) -> str:
        """
        Use Blender to bake proper AO from mesh geometry
        """
        script_path = Path(__file__).parent.parent.parent / "blender_scripts" / "bake_ao.py"
        
        if not script_path.exists():
            raise FileNotFoundError(f"Blender AO script not found: {script_path}")
        
        # Create config for Blender
        ao_samples = config.get("ao_samples", 128)
        config_path = output_path.parent / "ao_config_temp.json"
        with open(config_path, 'w') as f:
            json.dump({"samples": ao_samples}, f)
        
        cmd = [
            "blender",
            "--background",
            "--python", str(script_path),
            "--",
            mesh_path,
            str(output_path),
            str(config_path)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=120, text=True)
            
            # Clean up temp config
            if config_path.exists():
                config_path.unlink()
            
            return str(output_path)
        except subprocess.TimeoutExpired:
            if config_path.exists():
                config_path.unlink()
            raise TimeoutError("Blender AO baking timed out")
        except subprocess.CalledProcessError as e:
            if config_path.exists():
                config_path.unlink()
            raise Exception(f"Blender AO baking failed: {e.stderr}")
    
    # ==== File Transfer Methods ====
    
    def _upload_file(self, file_path: str) -> str:
        """
        Upload file to temporary storage (transfer.sh)
        Free, no account needed, 1-day retention
        """
        import requests
        
        file_name = Path(file_path).name
        
        try:
            with open(file_path, 'rb') as f:
                response = requests.post(
                    'https://transfer.sh/',
                    files={'file': (file_name, f)},
                    headers={'Max-Days': '1'}
                )
                response.raise_for_status()
                url = response.text.strip()
                print(f"   → {file_name}: {url}")
                return url
        except Exception as e:
            raise Exception(f"File upload failed: {e}")
    
    def _download_file(self, url: str, output_path: Path):
        """Download file from URL"""
        import requests
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)


if __name__ == "__main__":
    # Test the service
    service = PBRInferenceService(backend="auto")
    
    result = service.infer_pbr_materials(
        input_image_path="examples/captured.jpeg",
        mesh_path="out/0/mesh_cleaned.obj",
        output_dir="out/0/pbr_materials"
    )
    
    print("\n" + "="*70)
    print("RESULT:")
    print("="*70)
    print(json.dumps({k: v for k, v in result.items() if k not in ['textures']}, indent=2))
    
    if result.get("success"):
        print("\nGenerated textures:")
        for name, path in result.get("textures", {}).items():
            print(f"  - {name}: {path}")
