"""
Stage 2: Mesh Cleanup Service

This service takes raw meshes from Stage 1 (TripoSR reconstruction) and:
- Removes noise and artifacts
- Smooths geometry
- Fixes normals
- Auto-UV unwraps for proper texturing
- Optionally decimates mesh for performance

Uses Blender in headless mode for robust mesh operations.
"""
import os
import subprocess
import json
from pathlib import Path
from typing import Dict, Optional, Any


class MeshCleanupService:
    """
    Stage 2: Mesh Cleanup
    Converts raw AI-generated mesh into clean, usable geometry
    """
    
    def __init__(self, blender_executable: str = "blender"):
        """
        Initialize the mesh cleanup service
        
        Args:
            blender_executable: Path to Blender executable (default: "blender" from PATH)
        """
        self.blender_executable = blender_executable
        self.script_dir = Path(__file__).parent.parent.parent / "blender_scripts"
        
        # Verify Blender is available
        self._verify_blender()
    
    def _verify_blender(self):
        """Verify Blender is installed and accessible"""
        try:
            result = subprocess.run(
                [self.blender_executable, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                print(f"✓ Blender found: {result.stdout.split()[0:2]}")
            else:
                raise RuntimeError("Blender not responding correctly")
        except FileNotFoundError:
            raise RuntimeError(
                f"Blender not found at '{self.blender_executable}'. "
                "Please install Blender or provide correct path."
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Blender verification timed out")
    
    def cleanup_mesh(
        self,
        input_mesh_path: str,
        output_mesh_path: str,
        cleanup_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main cleanup pipeline
        
        Args:
            input_mesh_path: Path to raw mesh from Stage 1
            output_mesh_path: Path to save cleaned mesh
            cleanup_config: Optional config overrides (dict)
            
        Returns:
            Dict with cleanup results:
                - success: bool
                - input_path: str
                - output_path: str
                - config: dict (config used)
                - stats: dict (vertex/face counts, etc.)
                - stdout/stderr: str (if available)
        """
        # Merge default config with user overrides
        config = self._get_default_config()
        if cleanup_config:
            config.update(cleanup_config)
        
        # Validate paths
        input_path = Path(input_mesh_path)
        if not input_path.exists():
            return {
                "success": False,
                "error": f"Input mesh not found: {input_mesh_path}"
            }
        
        output_path = Path(output_mesh_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"🧹 STAGE 2: MESH CLEANUP")
        print(f"{'='*60}")
        print(f"Input:  {input_mesh_path}")
        print(f"Output: {output_mesh_path}")
        print(f"Config: {json.dumps(config, indent=2)}")
        print(f"{'='*60}\n")
        
        # Run Blender cleanup script
        result = self._run_blender_cleanup(
            str(input_path),
            str(output_path),
            config
        )
        
        if result['success']:
            print(f"\n✅ Mesh cleanup complete!")
            print(f"   Cleaned mesh saved to: {output_path}")
        else:
            print(f"\n❌ Mesh cleanup failed!")
            print(f"   Error: {result.get('error', 'Unknown error')}")
        
        return result
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default cleanup configuration"""
        return {
            # Noise removal
            "remove_noise": True,
            "noise_threshold": 0.0001,  # Remove duplicate vertices within this distance
            
            # Smoothing
            "smooth_iterations": 2,
            "smooth_factor": 0.5,  # 0.0-1.0, higher = more smoothing
            
            # Normals
            "fix_normals": True,
            
            # Decimation (mesh simplification)
            "decimate_ratio": None,  # e.g., 0.5 = reduce to 50% faces, None = no decimation
            
            # UV unwrapping
            "auto_uv_unwrap": True,
            "uv_method": "smart",  # 'smart', 'cube', 'sphere'
            "uv_margin": 0.01,  # Space between UV islands
            
            # Additional options
            "recalculate_normals": True,
            "smooth_shading": True
        }
    
    def _run_blender_cleanup(
        self,
        input_path: str,
        output_path: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute Blender in headless mode with cleanup script
        
        Args:
            input_path: Input mesh file path
            output_path: Output mesh file path
            config: Cleanup configuration
            
        Returns:
            Dict with execution results
        """
        script_path = self.script_dir / "mesh_cleanup.py"
        
        if not script_path.exists():
            return {
                "success": False,
                "error": f"Blender script not found: {script_path}"
            }
        
        # Create temp config file for Blender script
        config_path = Path(output_path).parent / "cleanup_config_temp.json"
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to write config file: {e}"
            }
        
        # Blender command: run in background with Python script
        cmd = [
            self.blender_executable,
            "--background",  # Headless mode (no GUI)
            "--python", str(script_path),
            "--",  # Everything after this goes to Python script
            input_path,
            output_path,
            str(config_path)
        ]
        
        print("Running Blender cleanup...")
        print(f"Command: {' '.join(cmd[:4])} ...")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=300  # 5 minute timeout
            )
            
            # Clean up temp config
            if config_path.exists():
                config_path.unlink()
            
            # Check if output was created
            if not Path(output_path).exists():
                return {
                    "success": False,
                    "error": "Blender completed but output file not found",
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            
            return {
                "success": True,
                "input_path": input_path,
                "output_path": output_path,
                "config": config,
                "stdout": result.stdout,
            }
            
        except subprocess.CalledProcessError as e:
            # Clean up temp config
            if config_path.exists():
                config_path.unlink()
            
            return {
                "success": False,
                "error": f"Blender process failed with exit code {e.returncode}",
                "stdout": e.stdout,
                "stderr": e.stderr
            }
        except subprocess.TimeoutExpired:
            # Clean up temp config
            if config_path.exists():
                config_path.unlink()
            
            return {
                "success": False,
                "error": "Blender cleanup timed out after 5 minutes"
            }
        except Exception as e:
            # Clean up temp config
            if config_path.exists():
                config_path.unlink()
            
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }


if __name__ == "__main__":
    # Quick test when run directly
    service = MeshCleanupService()
    
    result = service.cleanup_mesh(
        input_mesh_path="out/0/mesh.obj",
        output_mesh_path="out/0/mesh_cleaned.obj"
    )
    
    print("\n" + "="*60)
    print("RESULT:")
    print("="*60)
    print(json.dumps({k: v for k, v in result.items() if k not in ['stdout', 'stderr']}, indent=2))
