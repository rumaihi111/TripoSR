"""
Blender Script for Ambient Occlusion Baking

Bakes AO (Ambient Occlusion) from mesh geometry.
AO represents how much ambient light reaches each point on the surface.

Usage:
    blender --background --python bake_ao.py -- <mesh_path> <output_path> <config_path>
"""

import bpy
import sys
import json
from pathlib import Path


def bake_ambient_occlusion(mesh_path: str, output_path: str, config_path: str):
    """
    Bake AO from mesh geometry
    
    Args:
        mesh_path: Path to input mesh (.obj, .glb, etc.)
        output_path: Path to save AO map (.png)
        config_path: Path to JSON config file
    """
    
    print("\n" + "="*60)
    print("🌑 BLENDER AO BAKING")
    print("="*60)
    print(f"Mesh: {mesh_path}")
    print(f"Output: {output_path}")
    print("="*60 + "\n")
    
    # Load configuration
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        samples = config.get("samples", 128)
        print(f"AO samples: {samples}\n")
    except Exception as e:
        print(f"⚠️  Config load failed: {e}, using defaults")
        samples = 128
    
    # Clear scene
    print("🗑️  Clearing scene...")
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Import mesh
    print(f"📥 Importing mesh...")
    try:
        if mesh_path.endswith('.obj'):
            bpy.ops.wm.obj_import(filepath=mesh_path)
        elif mesh_path.endswith('.glb') or mesh_path.endswith('.gltf'):
            bpy.ops.import_scene.gltf(filepath=mesh_path)
        elif mesh_path.endswith('.fbx'):
            bpy.ops.import_scene.fbx(filepath=mesh_path)
        else:
            raise ValueError(f"Unsupported mesh format: {mesh_path}")
        print("✓ Mesh imported\n")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        sys.exit(1)
    
    # Get object
    if not bpy.context.selected_objects:
        print("❌ No objects imported!")
        sys.exit(1)
    
    obj = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = obj
    
    mesh = obj.data
    print(f"📊 Mesh stats:")
    print(f"   Vertices: {len(mesh.vertices)}")
    print(f"   Faces: {len(mesh.polygons)}\n")
    
    # Setup for baking
    print("🔧 Setting up bake configuration...")
    
    # Switch to Cycles render engine (required for AO baking)
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = samples
    bpy.context.scene.cycles.device = 'CPU'
    
    # Create UV map if needed
    if not mesh.uv_layers:
        print("   Creating UV map...")
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project()
        bpy.ops.object.mode_set(mode='OBJECT')
    else:
        print("   ✓ UV map exists")
    
    # Create new image for baking
    resolution = 2048
    print(f"   Creating {resolution}x{resolution} bake target...")
    
    image = bpy.data.images.new(
        name="AO_Bake",
        width=resolution,
        height=resolution,
        alpha=False
    )
    
    # Setup material with image texture node
    if not obj.data.materials:
        mat = bpy.data.materials.new(name="BakeMaterial")
        obj.data.materials.append(mat)
    else:
        mat = obj.data.materials[0]
    
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    # Add image texture node
    tex_node = nodes.new(type='ShaderNodeTexImage')
    tex_node.image = image
    tex_node.select = True
    nodes.active = tex_node
    
    print("✓ Bake configuration complete\n")
    
    # Bake AO
    print(f"🚀 Baking ambient occlusion ({samples} samples)...")
    print("   This may take 30-60 seconds...\n")
    
    try:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        
        bpy.ops.object.bake(
            type='AO',
            pass_filter={'AO'},
            width=resolution,
            height=resolution,
            margin=16,
            use_clear=True
        )
        
        print("✓ Baking complete!\n")
    except Exception as e:
        print(f"❌ Baking failed: {e}")
        sys.exit(1)
    
    # Save image
    print(f"💾 Saving AO map to {output_path}...")
    
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image.filepath_raw = output_path
    image.file_format = 'PNG'
    image.save()
    
    print("✓ AO map saved\n")
    
    print("="*60)
    print("✅ AO BAKING COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Parse command line arguments
    argv = sys.argv
    
    if "--" not in argv:
        print("❌ Error: No arguments provided")
        print("Usage: blender --background --python bake_ao.py -- <mesh> <output> <config>")
        sys.exit(1)
    
    argv = argv[argv.index("--") + 1:]
    
    if len(argv) < 3:
        print("❌ Error: Missing arguments")
        print("Usage: blender --background --python bake_ao.py -- <mesh> <output> <config>")
        sys.exit(1)
    
    mesh_path = argv[0]
    output_path = argv[1]
    config_path = argv[2]
    
    bake_ambient_occlusion(mesh_path, output_path, config_path)
