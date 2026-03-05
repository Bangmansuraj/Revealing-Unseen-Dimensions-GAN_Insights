# File: render_full_raw.py (Version 3, with robust camera)
import os
import pyvista as pv
import trimesh
import random

# --- 1. CONFIGURATION (Should be correct from before) ---
SHAPENET_PATH = "C:/Users/suraj ns/Downloads/major_project/datasets/shapenet_raw"
RAW_OUTPUT_PATH = "C:/Users/suraj ns/Downloads/major_project/datasets/rendered_images_raw_multicolor"

# --- 2. SCRIPT (DO NOT EDIT) ---
front_dir = os.path.join(RAW_OUTPUT_PATH, 'full_raw_front')
side_dir = os.path.join(RAW_OUTPUT_PATH, 'full_raw_side')
os.makedirs(front_dir, exist_ok=True)
os.makedirs(side_dir, exist_ok=True)

# Set up the plotter. It will be reused for each model.
plotter = pv.Plotter(off_screen=True, window_size=[256, 256])

def render_model(model_path, model_id):
    """ Loads a model, renders front and side views with a smart camera, and saves them. """
    try:
        # Use trimesh to load and normalize the object
        mesh = trimesh.load(model_path, force='mesh')
        mesh.apply_translation(-mesh.centroid)
        mesh.apply_scale(1 / mesh.scale)

        # Convert to a PyVista mesh for rendering
        pv_mesh = pv.wrap(mesh)

        # Clear any previous models from the plotter
        plotter.clear()

        # Assign a random color
        random_color = random.choice(['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'white', 'brown'])
        plotter.add_mesh(pv_mesh, color=random_color, smooth_shading=True)

        # --- CAMERA FIX: Use auto-fitting views ---

        # Render Front View (view from the front, looking along the Y-axis)
        plotter.view_xz() # This automatically sets the camera for a front view
        plotter.camera.zoom(1.2) # Zoom in slightly to fill the frame
        plotter.screenshot(os.path.join(front_dir, f"{model_id}.png"), transparent_background=True)

        # Render Side View (view from the side, looking along the X-axis)
        plotter.view_yz() # This automatically sets the camera for a side view
        plotter.camera.zoom(1.2) # Zoom in slightly
        plotter.screenshot(os.path.join(side_dir, f"{model_id}.png"), transparent_background=True)

        # -----------------------------------------

        print(f"Rendered: {model_id} in {random_color}")

    except Exception as e:
        print(f"FAILED to render {model_id}: {e}")

# --- 3. MAIN LOOP (Updated with the nested path) ---
chair_models_dir = os.path.join(SHAPENET_PATH, '03001627', '03001627')
model_ids = os.listdir(chair_models_dir)

print(f"Found {len(model_ids)} models. Starting rendering...")
for model_id in model_ids:
    model_file = os.path.join(chair_models_dir, model_id, 'models', 'model_normalized.obj')
    if os.path.exists(model_file):
        render_model(model_file, model_id)

plotter.close()
print("--- Raw multi-color rendering complete! ---")