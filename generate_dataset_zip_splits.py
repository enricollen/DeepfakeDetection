import os
import zipfile
from pathlib import Path
from tqdm import tqdm  # Import tqdm for progress bars

def create_zip_files(input_folder, output_folder=None, max_size_total=10*1024):
    input_folder = Path(input_folder)
    
    if output_folder is None:
        output_folder = input_folder / "output_zips"
    else:
        output_folder = Path(output_folder)

    output_folder.mkdir(exist_ok=True)

    current_zip_size = 0
    total_zip_size = 0
    current_zip_number = 1
    current_zip_path = output_folder / f"images_{current_zip_number}.zip"

    with zipfile.ZipFile(current_zip_path, 'w') as current_zip:
        for root, dirs, files in os.walk(input_folder):
            for file in tqdm(files, desc="Creating ZIP files", unit="file"):
                file_path = Path(root) / file
                file_size = file_path.stat().st_size / (1024 * 1024)  # Convert to MB

                if current_zip_size + file_size > max_size_total:
                    # If adding the current file exceeds the total limit, create a new zip file
                    current_zip_number += 1
                    current_zip_size = 0
                    total_zip_size = 0
                    current_zip_path = output_folder / f"images_{current_zip_number}.zip"
                    current_zip = zipfile.ZipFile(current_zip_path, 'w')

                if current_zip_size + file_size > max_size_total:
                    print(f"Warning: Skipping '{file_path}' as it exceeds the total size limit.")
                    continue

                # Add the file to the current zip file
                relative_path = file_path.relative_to(input_folder)
                current_zip.write(file_path, arcname=relative_path)
                current_zip_size += file_size
                total_zip_size += file_size

    print(f"Zip files created successfully in {output_folder}")

input_folder_path = 'C:/Users/nello/Desktop/TESI/public_image_set_after_data_cleaning'
output_folder_path = 'C:/Users/nello/Desktop/TESI/ZIP'
create_zip_files(input_folder_path, output_folder_path)