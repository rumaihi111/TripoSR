"""
Test Script for Stage 3: PBR Material Inference

Tests the PBR inference service with different backends and configurations.
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.pbr_inference_service import PBRInferenceService


def test_cpu_inference():
    """Test CPU-only PBR inference (OpenCV)"""
    print("\n" + "="*70)
    print("TEST 1: CPU-Only Inference (OpenCV)")
    print("="*70)
    
    service = PBRInferenceService(backend="cpu")
    
    result = service.infer_pbr_materials(
        input_image_path="examples/captured.jpeg",
        mesh_path="out/0/mesh_cleaned.obj",
        output_dir="out/0/pbr_cpu"
    )
    
    print_result(result)
    return result['success']


def test_gpu_inference():
    """Test GPU inference via RunPod (if API key is set)"""
    print("\n" + "="*70)
    print("TEST 2: GPU Inference (RunPod)")
    print("="*70)
    
    import os
    if not os.getenv("RUNPOD_API_KEY"):
        print("⚠️  RUNPOD_API_KEY not set - skipping GPU test")
        print("   Set RUNPOD_API_KEY to test GPU backend")
        return True  # Don't fail if GPU not configured
    
    if not os.getenv("RUNPOD_ENDPOINT_ID"):
        print("⚠️  RUNPOD_ENDPOINT_ID not set")
        print("   GPU backend requires a RunPod endpoint")
        print("   For now, this will fall back to CPU")
        print("   Skipping GPU-specific test")
        return True
    
    service = PBRInferenceService(backend="gpu")
    
    result = service.infer_pbr_materials(
        input_image_path="examples/captured.jpeg",
        mesh_path="out/0/mesh_cleaned.obj",
        output_dir="out/0/pbr_gpu"
    )
    
    print_result(result)
    return result['success']


def test_auto_backend():
    """Test auto backend selection (GPU if available, otherwise CPU)"""
    print("\n" + "="*70)
    print("TEST 3: Auto Backend Selection")
    print("="*70)
    
    service = PBRInferenceService(backend="auto")
    
    result = service.infer_pbr_materials(
        input_image_path="examples/captured.jpeg",
        mesh_path="out/0/mesh_cleaned.obj",
        output_dir="out/0/pbr_auto",
        config={
            "texture_size": 1024,  # Lower res for faster testing
            "ao_samples": 64,      # Fewer samples for speed
            "base_roughness": 0.6,
            "base_metallic": 0.0
        }
    )
    
    print_result(result)
    return result['success']


def test_with_decimated_mesh():
    """Test with decimated (optimized) mesh"""
    print("\n" + "="*70)
    print("TEST 4: PBR Inference with Decimated Mesh")
    print("="*70)
    
    decimated_mesh = "out/0/mesh_cleaned_decimated.obj"
    
    if not Path(decimated_mesh).exists():
        print(f"⚠️  Decimated mesh not found: {decimated_mesh}")
        print("   Run Stage 2 mesh cleanup first")
        return True  # Don't fail if mesh not available
    
    service = PBRInferenceService(backend="auto")
    
    result = service.infer_pbr_materials(
        input_image_path="examples/captured.jpeg",
        mesh_path=decimated_mesh,
        output_dir="out/0/pbr_decimated"
    )
    
    print_result(result)
    return result['success']


def print_result(result: dict):
    """Pretty print test result"""
    print("\n" + "-"*70)
    print("RESULT:")
    print("-"*70)
    
    if result['success']:
        print("✅ SUCCESS")
        print(f"\nBackend: {result.get('backend', 'unknown')}")
        print(f"Cost: ${result.get('cost_usd', 0):.4f}")
        
        if result.get('gpu_time_seconds', 0) > 0:
            print(f"GPU time: {result['gpu_time_seconds']:.1f}s")
        
        print(f"\nGenerated textures:")
        for name, path in result.get('textures', {}).items():
            size_kb = Path(path).stat().st_size / 1024 if Path(path).exists() else 0
            print(f"  ✓ {name:12s} → {path} ({size_kb:.1f} KB)")
        
        if 'note' in result:
            print(f"\nNote: {result['note']}")
    else:
        print("❌ FAILED")
        print(f"\nError: {result.get('error', 'Unknown error')}")
    
    print("-"*70)


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("STAGE 3: PBR MATERIAL INFERENCE - TEST SUITE")
    print("="*70)
    
    # Check prerequisites
    input_image = Path("examples/captured.jpeg")
    input_mesh = Path("out/0/mesh_cleaned.obj")
    
    print("\nChecking prerequisites...")
    
    if not input_image.exists():
        print(f"❌ Error: Input image not found: {input_image}")
        print("   Please ensure the example image exists")
        sys.exit(1)
    print(f"✓ Input image: {input_image}")
    
    if not input_mesh.exists():
        print(f"❌ Error: Cleaned mesh not found: {input_mesh}")
        print("   Please run Stage 2 (mesh cleanup) first")
        sys.exit(1)
    print(f"✓ Cleaned mesh: {input_mesh}")
    
    import os
    api_key_set = bool(os.getenv("RUNPOD_API_KEY"))
    print(f"{'✓' if api_key_set else '⚠️ '} RunPod API key: {'Set' if api_key_set else 'Not set (will use CPU)'}")
    
    print("\n" + "="*70)
    
    # Run tests
    tests = [
        ("CPU Inference", test_cpu_inference),
        ("GPU Inference", test_gpu_inference),
        ("Auto Backend", test_auto_backend),
        ("Decimated Mesh", test_with_decimated_mesh)
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*70 + "\n")
    
    # Show output locations
    if passed > 0:
        print("Generated PBR materials saved to:")
        for output_dir in ["out/0/pbr_cpu", "out/0/pbr_gpu", "out/0/pbr_auto", "out/0/pbr_decimated"]:
            if Path(output_dir).exists():
                print(f"  - {output_dir}/")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
