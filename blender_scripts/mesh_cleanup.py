"""
Blender Script for Mesh Cleanup (Stage 2)

This script runs inside Blender (headless mode) and performs:
- Noise removal (duplicate vertices, loose geometry)
- Geometry smoothing
- Normal fixing
- UV unwrapping
- Optional decimation

Usage:
    blender --background --python mesh_cleanup.py -- <input> <output> <config.json>
"""
import bpy
import sys
import json
from pathlib import Path


def cleanup_mesh(input_path: str, output_path: str, config_path: str):
    """
    Main cleanup function executed in Blender
    
    Args:
        input_path: Path to input mesh file
        output_path: Path to save cleaned mesh
        config_path: Path to JSON config file
    """
    print("\n" + "="*60)
    print("🔧 BLENDER MESH CLEANUP")
    print("="*60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Config: {config_path}")
    print("="*60 + "\n")
    
    # Load configuration
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✓ Config loaded: {json.dumps(config, indent=2)}\n")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        sys.exit(1)
    
    # Clear default scene
    print("🗑️  Clearing default scene...")
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Import mesh
    print(f"📥 Importing mesh from {input_path}...")
    try:
        if input_path.endswith('.obj'):
            bpy.ops.wm.obj_import(filepath=input_path)
        elif input_path.endswith('.glb') or input_path.endswith('.gltf'):
            bpy.ops.import_scene.gltf(filepath=input_path)
        elif input_path.endswith('.fbx'):
            bpy.ops.import_scene.fbx(filepath=input_path)
        else:
            raise ValueError(f"Unsupported file format: {input_path}")
        print("✓ Mesh imported successfully\n")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        sys.exit(1)
    
    # Get imported object
    if not bpy.context.selected_objects:
        print("❌ No objects were imported!")
        sys.exit(1)
    
    obj = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = obj
    
    # Print initial stats
    mesh = obj.data
    initial_verts = len(mesh.vertices)
    initial_faces = len(mesh.polygons)
    print(f"📊 Initial mesh stats:")
    print(f"   Vertices: {initial_verts}")
    print(f"   Faces: {initial_faces}\n")
    
    # === CLEANUP OPERATIONS ===
    
    # Enter edit mode for mesh operations
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    
    # 1. Remove noise and duplicates
    if config.get('remove_noise', True):
        print("🧹 Step 1: Removing noise...")
        
        # Remove duplicate vertices
        threshold = config.get('noise_threshold', 0.0001)
        print(f"   - Merging vertices by distance (threshold: {threshold})...")
        bpy.ops.mesh.remove_doubles(threshold=threshold)
        
        # Delete loose geometry (vertices/edges not part of faces)
        print("   - Removing loose geometry...")
        bpy.ops.mesh.delete_loose()
        
        # Remove degenerate faces
        print("   - Removing degenerate faces...")
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.mesh.select_face_by_sides(number=3, type='LESS')
        bpy.ops.mesh.delete(type='FACE')
        bpy.ops.mesh.select_all(action='SELECT')
        
        print("   ✓ Noise removal complete\n")
    
    # 2. Fix normals
    if config.get('fix_normals', True):
        print("🔧 Step 2: Fixing normals...")
        bpy.ops.mesh.select_all(action='SELECT')
        
        if config.get('recalculate_normals', True):
            print("   - Recalculating normals (outside)...")
            bpy.ops.mesh.normals_make_consistent(inside=False)
        
        print("   ✓ Normals fixed\n")
    
    # 3. Smooth geometry
    smooth_iterations = config.get('smooth_iterations', 0)
    if smooth_iterations > 0:
        print(f"✨ Step 3: Smoothing mesh ({smooth_iterations} iterations)...")
        smooth_factor = config.get('smooth_factor', 0.5)
        
        for i in range(smooth_iterations):
            bpy.ops.mesh.vertices_smooth(factor=smooth_factor)
            if i % 5 == 4:  # Progress update every 5 iterations
                print(f"   - Iteration {i+1}/{smooth_iterations}")
        
        print("   ✓ Smoothing complete\n")
    
    # Back to object mode for modifiers
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # 4. Optional decimation (mesh simplification)
    decimate_ratio = config.get('decimate_ratio')
    if decimate_ratio is not None and 0 < decimate_ratio < 1:
        print(f"🔻 Step 4: Decimating mesh (target ratio: {decimate_ratio})...")
        modifier = obj.modifiers.new(name="Decimate", type='DECIMATE')
        modifier.ratio = decimate_ratio
        modifier.use_collapse_triangulate = True
        bpy.ops.object.modifier_apply(modifier="Decimate")
        print("   ✓ Decimation complete\n")
    
    # 5. UV unwrapping
    if config.get('auto_uv_unwrap', True):
        print("📐 Step 5: UV unwrapping...")
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        
        uv_method = config.get('uv_method', 'smart')
        uv_margin = config.get('uv_margin', 0.01)
        
        if uv_method == 'smart':
            print(f"   - Using Smart UV Project (margin: {uv_margin})...")
            bpy.ops.uv.smart_project(
                angle_limit=66.0,
                island_margin=uv_margin,
                area_weight=0.0
            )
        elif uv_method == 'cube':
            print("   - Using Cube Projection...")
            bpy.ops.uv.cube_project()
        elif uv_method == 'sphere':
            print("   - Using Sphere Projection...")
            bpy.ops.uv.sphere_project()
        else:
            print(f"   ⚠️  Unknown UV method '{uv_method}', using smart...")
            bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=uv_margin)
        
        bpy.ops.object.mode_set(mode='OBJECT')
        print("   ✓ UV unwrapping complete\n")
    
    # 6. Apply smooth shading
    if config.get('smooth_shading', True):
        print("🎨 Step 6: Applying smooth shading...")
        bpy.ops.object.shade_smooth()
        print("   ✓ Smooth shading applied\n")
    
    # Print final stats
    mesh = obj.data
    final_verts = len(mesh.vertices)
    final_faces = len(mesh.polygons)
    print(f"📊 Final mesh stats:")
    print(f"   Vertices: {final_verts} (change: {final_verts - initial_verts:+d})")
    print(f"   Faces: {final_faces} (change: {final_faces - initial_faces:+d})")
    print(f"   UV Maps: {len(mesh.uv_layers)}\n")
    
    # Export cleaned mesh
    print(f"💾 Exporting cleaned mesh to {output_path}...")
    
    try:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if output_path.endswith('.obj'):
            bpy.ops.wm.obj_export(
                filepath=output_path,
                export_selected_objects=True,
                export_uv=True,
                export_normals=True,
                export_materials=True,
                path_mode='COPY'
            )
        elif output_path.endswith('.glb'):
            bpy.ops.export_scene.gltf(
                filepath=output_path,
                use_selection=True,
                export_format='GLB',
                export_uv=True,
                export_normals=True
            )
        elif output_path.endswith('.fbx'):
            bpy.ops.export_scene.fbx(
                filepath=output_path,
                use_selection=True,
                mesh_smooth_type='FACE'
            )
        else:
            print(f"⚠️  Unsupported output format, exporting as OBJ...")
            output_path = output_path.rsplit('.', 1)[0] + '.obj'
            bpy.ops.wm.obj_export(
                filepath=output_path,
                export_selected_objects=True,
                export_uv=True,
                export_normals=True,
                export_materials=True
            )
        
        print("✓ Export successful\n")
    except Exception as e:
        print(f"❌ Export failed: {e}")
        sys.exit(1)
    
    print("="*60)
    print("✅ MESH CLEANUP COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Parse command line arguments
    # Blender passes args after '--'
    argv = sys.argv
    
    if "--" not in argv:
        print("❌ Error: No arguments provided")
        print("Usage: blender --background --python mesh_cleanup.py -- <input> <output> <config>")
        sys.exit(1)
    
    argv = argv[argv.index("--") + 1:]  # Get all args after --
    
    if len(argv) < 3:
        print("❌ Error: Missing required arguments")
        print("Usage: blender --background --python mesh_cleanup.py -- <input> <output> <config>")
        sys.exit(1)
    
    input_path = argv[0]
    output_path = argv[1]
    config_path = argv[2]
    
    # Run cleanup
    cleanup_mesh(input_path, output_path, config_path)
