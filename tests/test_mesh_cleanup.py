"""
Test script for Stage 2: Mesh Cleanup

This tests the mesh cleanup service on the output from Stage 1 (TripoSR).
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.mesh_cleanup_service import MeshCleanupService


def test_basic_cleanup():
    """Test basic cleanup with default settings"""
    print("\n" + "="*70)
    print("TEST 1: Basic Cleanup (Default Settings)")
    print("="*70)
    
    service = MeshCleanupService()
    
    result = service.cleanup_mesh(
        input_mesh_path="out/0/mesh.obj",
        output_mesh_path="out/0/mesh_cleaned.obj"
    )
    
    print_result(result)
    return result['success']


def test_aggressive_cleanup():
    """Test more aggressive cleanup with smoothing"""
    print("\n" + "="*70)
    print("TEST 2: Aggressive Cleanup (More Smoothing)")
    print("="*70)
    
    service = MeshCleanupService()
    
    result = service.cleanup_mesh(
        input_mesh_path="out/0/mesh.obj",
        output_mesh_path="out/0/mesh_cleaned_smooth.obj",
        cleanup_config={
            "smooth_iterations": 5,
            "smooth_factor": 0.7,
            "auto_uv_unwrap": True,
            "uv_method": "smart"
        }
    )
    
    print_result(result)
    return result['success']


def test_with_decimation():
    """Test cleanup with mesh simplification"""
    print("\n" + "="*70)
    print("TEST 3: Cleanup with Decimation (50% faces)")
    print("="*70)
    
    service = MeshCleanupService()
    
    result = service.cleanup_mesh(
        input_mesh_path="out/0/mesh.obj",
        output_mesh_path="out/0/mesh_cleaned_decimated.obj",
        cleanup_config={
            "smooth_iterations": 2,
            "decimate_ratio": 0.5,  # Reduce to 50% of original faces
            "auto_uv_unwrap": True
        }
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
        print(f"\nInput:  {result.get('input_path', 'N/A')}")
        print(f"Output: {result.get('output_path', 'N/A')}")
        print(f"\nConfiguration used:")
        if 'config' in result:
            for key, value in result['config'].items():
                print(f"  - {key}: {value}")
    else:
        print("❌ FAILED")
        print(f"\nError: {result.get('error', 'Unknown error')}")
        if 'stderr' in result:
            print(f"\nStderr:\n{result['stderr']}")
    
    print("-"*70)


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("STAGE 2: MESH CLEANUP - TEST SUITE")
    print("="*70)
    
    # Check if input exists
    input_path = Path("out/0/mesh.obj")
    if not input_path.exists():
        print(f"\n❌ Error: Input mesh not found at {input_path}")
        print("Please run Stage 1 (TripoSR) first to generate a mesh.")
        sys.exit(1)
    
    print(f"✓ Found input mesh: {input_path}")
    print(f"  Size: {input_path.stat().st_size / 1024:.2f} KB")
    
    # Run tests
    tests = [
        ("Basic Cleanup", test_basic_cleanup),
        ("Aggressive Cleanup", test_aggressive_cleanup),
        ("With Decimation", test_with_decimation)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed with exception: {e}")
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
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
