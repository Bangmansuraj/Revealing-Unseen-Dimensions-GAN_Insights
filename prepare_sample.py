import os
import shutil

print("--- Starting Dataset Sample Preparation ---")

# --- Configuration ---
# This is the location of your full dataset on your PC
SOURCE_FRONT = "C:/Users/suraj ns/Downloads/major_project/view_front"
SOURCE_SIDE = "C:/Users/suraj ns/Downloads/major_project/view_side"

# This is where the new sample folders will be created
DEST_FRONT = "C:/Users/suraj ns/Downloads/major_project/sample_front"
DEST_SIDE = "C:/Users/suraj ns/Downloads/major_project/sample_side"

# How many images you want in your sample
NUM_IMAGES = 1900

# --- Script Logic ---
# Create the new destination folders
os.makedirs(DEST_FRONT, exist_ok=True)
os.makedirs(DEST_SIDE, exist_ok=True)

# Get a sorted list of the first 200 files from the source
print(f"Getting the first {NUM_IMAGES} filenames...")
filenames_to_copy = sorted(os.listdir(SOURCE_FRONT))[:NUM_IMAGES]

print(f"Copying {len(filenames_to_copy)} matched files...")

# Loop through the list and copy the files
for filename in filenames_to_copy:
    # Copy front view
    shutil.copy(os.path.join(SOURCE_FRONT, filename), os.path.join(DEST_FRONT, filename))
    # Copy the MATCHING side view
    shutil.copy(os.path.join(SOURCE_SIDE, filename), os.path.join(DEST_SIDE, filename))

print("--- Done! Your sample dataset is ready. ---")