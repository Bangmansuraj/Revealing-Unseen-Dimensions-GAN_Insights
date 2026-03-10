# This is the FINAL CORRECTED version of the script

import pyvista as pv
import os
import time

print("--- Starting FINAL CORRECTED Paired Image Rendering Script ---")

# --- Define File Paths ---
base_path = "C:/Users/suraj ns/Downloads/major_project/datasets/shapenet_raw"
chair_folder_path = os.path.join(base_path, "03001627", "03001627")
output_dir_front = "C:/Users/suraj ns/Downloads/major_project/view_front"
output_dir_side = "C:/Users/suraj ns/Downloads/major_project/view_side"

# --- Create the output folders if they don't exist ---
if not os.path.exists(output_dir_front): os.makedirs(output_dir_front)
if not os.path.exists(output_dir_side): os.makedirs(output_dir_side)

model_ids = os.listdir(chair_folder_path)
total_models = len(model_ids)
print(f"Found {total_models} models to render.")

# --- Loop through each model and render two views ---
for i, model_id in enumerate(model_ids):
    model_path = os.path.join(chair_folder_path, model_id, "models", "model_normalized.obj")

    if os.path.exists(model_path):
        print(f"Processing model {i+1}/{total_models}: {model_id}")

        mesh = pv.read(model_path)
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(mesh)
        plotter.background_color = 'white'

        # --- CORRECTED Render Front View ---
        plotter.view_xy(negative=True) # Sets the camera to the "back" (our front)
        output_file_front = os.path.join(output_dir_front, f"{model_id}.png")
        plotter.screenshot(output_file_front)

        # --- CORRECTED Render Side View ---
        plotter.view_zy(negative=True) # Sets the camera to the opposite side
        output_file_side = os.path.join(output_dir_side, f"{model_id}.png")
        plotter.screenshot(output_file_side)
        
        plotter.close()

    else:
        print(f"Warning: model_normalized.obj not found for {model_id}")

print(f"--- Script Finished! All paired images are saved. ---")