import shutil
from pathlib import Path
import os
from nerfstudio.utils.io import load_from_json

json_path = Path('../transforms_test.json')
des_dir = '../gt'

def get_absolute_path(source_path, relative_path):
    # Join the source path with the relative path
    combined_path = os.path.join(source_path, relative_path)
    # Normalize the path to handle any '..' or '.' components
    absolute_path = os.path.normpath(combined_path)
    return absolute_path

def copy_image_to_destination(source_image_path, destination_directory):
    # Ensure the destination directory exists
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    # Define the destination file path
    destination_file_path = os.path.join(destination_directory, os.path.basename(source_image_path))
    # Copy the image file
    shutil.copy(source_image_path, destination_file_path)

src_dir = json_path.parent
test_meta = load_from_json(json_path)

for frame in test_meta["frames"]:
    relative_image_path = frame["file_path"]
    image_path = get_absolute_path(src_dir, relative_image_path)
    copy_image_to_destination(image_path, des_dir)