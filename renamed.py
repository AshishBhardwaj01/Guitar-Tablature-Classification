# import os
# import re
# import shutil

# # Directories
# CQT_IMAGES_DIR = "./cqt_images"
# RENAMED_DIR = "./cqt_images_renamed"

# # Create new directory if it doesn't exist
# os.makedirs(RENAMED_DIR, exist_ok=True)

# # Step 1: Collect and group files by base name
# file_groups = {}

# for file in sorted(os.listdir(CQT_IMAGES_DIR)):  # Ensure sorted order
#     match = re.match(r"(.*)_segment_(\d+)_(\d+)\.(\d+)\.png", file)
#     if match:
#         base_name = match.group(1)  # Extract base name before "_segment_"
#         minute = int(match.group(2))
#         second = int(match.group(3))
#         decisecond = int(match.group(4))

#         # Convert time into total seconds
#         total_time = minute * 60 + second + decisecond / 10.0

#         if base_name not in file_groups:
#             file_groups[base_name] = []

#         file_groups[base_name].append((total_time, file))

# # Step 2: Rename files and copy them to the new directory
# for base_name, files in file_groups.items():
#     # Sort files by total time
#     files.sort()

#     for index, (_, file) in enumerate(files):
#         new_filename = f"{base_name}_{index:04d}.png"  # Format as _0000, _0001, ...
#         src_path = os.path.join(CQT_IMAGES_DIR, file)
#         dest_path = os.path.join(RENAMED_DIR, new_filename)

#         shutil.copy(src_path, dest_path)  # Copy to new directory
#         print(f"Copied {file} → {new_filename}")

# print("✅ Renaming complete! Check 'cqt_images_renamed/' for updated files.")
import os
import re
import shutil
import zipfile
from google.colab import files  # If you're using Google Colab

# Directories
CQT_IMAGES_DIR = "./cqt_images"
RENAMED_DIR = "./cqt_images_renamed"
ZIP_NAME = "cqt_images_renamed.zip"

# Create new directory if it doesn't exist
os.makedirs(RENAMED_DIR, exist_ok=True)

# Step 1: Collect and group files by base name
file_groups = {}
for file in sorted(os.listdir(CQT_IMAGES_DIR)):  # Ensure sorted order
    match = re.match(r"(.*)_segment_(\d+)_(\d+)\.(\d+)\.png", file)
    if match:
        base_name = match.group(1)  # Extract base name before "_segment_"
        minute = int(match.group(2))
        second = int(match.group(3))
        decisecond = int(match.group(4))
        # Convert time into total seconds
        total_time = minute * 60 + second + decisecond / 10.0
        if base_name not in file_groups:
            file_groups[base_name] = []
        file_groups[base_name].append((total_time, file))

# Step 2: Rename files and copy them to the new directory
for base_name, files in file_groups.items():
    # Sort files by total time
    files.sort()
    for index, (_, file) in enumerate(files):
        new_filename = f"{base_name}_{index:04d}.png"  # Format as *0000, *0001, ...
        src_path = os.path.join(CQT_IMAGES_DIR, file)
        dest_path = os.path.join(RENAMED_DIR, new_filename)
        shutil.copy(src_path, dest_path)  # Copy to new directory
        print(f"Copied {file} → {new_filename}")

print("✅ Renaming complete! Check 'cqt_images_renamed/' for updated files.")

# Step 3: Create a zip file of the renamed directory
def zip_directory(directory, zip_name):
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, os.path.join(directory, '..')))
    print(f"✅ Created zip file: {zip_name}")

zip_directory(RENAMED_DIR, ZIP_NAME)

# Step 4: Download the zip file (for Google Colab)
try:
    from google.colab import files
    files.download(ZIP_NAME)
    print(f"✅ Downloading {ZIP_NAME}...")
except ImportError:
    print(f"✅ Zip file created at {ZIP_NAME}. Please download it manually.")
