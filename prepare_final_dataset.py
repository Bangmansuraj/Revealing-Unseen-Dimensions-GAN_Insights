# File: prepare_final_dataset.py
import os
import shutil

# --- 1. CONFIGURATION (EDIT THESE TWO LINES) ---

# Path to the folder that CONTAINS your 'view_front' and 'view_side' folders.
RAW_DATA_PATH = "C:/Users/suraj ns/Downloads/major_project"

# Path where you WANT the new 'final_dataset' folder to be created.
FINAL_DATA_PATH = "C:/Users/suraj ns/downloads/major_project/final_dataset"

# --- 2. SCRIPT (DO NOT EDIT BELOW THIS LINE) ---

# Input folders are your existing folders
raw_front_dir = os.path.join(RAW_DATA_PATH, 'view_front')
raw_side_dir = os.path.join(RAW_DATA_PATH, 'view_side')

# Output folders for the clean, paired data
final_front_dir = os.path.join(FINAL_DATA_PATH, 'full_front')
final_side_dir = os.path.join(FINAL_DATA_PATH, 'full_side')
os.makedirs(final_front_dir, exist_ok=True)
os.makedirs(final_side_dir, exist_ok=True)

paired_count = 0
# Use the front-view images as the source of truth
try:
    all_front_images = os.listdir(raw_front_dir)
except FileNotFoundError:
    print(f"ERROR: Could not find the folder '{raw_front_dir}'.")
    print("Please check the 'RAW_DATA_PATH' variable in the script.")
    exit()

total_images = len(all_front_images)

print(f"Scanning {total_images} rendered images to find pairs...")

for i, img_name in enumerate(all_front_images):
    front_path = os.path.join(raw_front_dir, img_name)
    side_path = os.path.join(raw_side_dir, img_name)

    # Check if the side view partner exists in the other folder
    if os.path.exists(side_path):
        # If it exists, copy both files to the final destination
        shutil.copy(front_path, os.path.join(final_front_dir, img_name))
        shutil.copy(side_path, os.path.join(final_side_dir, img_name))
        paired_count += 1

    print(f"  Processed {i+1}/{total_images} | Found pairs: {paired_count}", end='\r')

print(f"\n--- Pairing complete! ---")
print(f"Found and copied {paired_count} valid image pairs into the '{FINAL_DATA_PATH}' folder.")