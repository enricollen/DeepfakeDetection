import os
import json
import csv
import PIL
from transformers import pipeline
from tqdm import tqdm
import time

def get_image_files(directory):
    files = os.listdir(directory)
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    return [os.path.join(directory, f) for f in image_files]

def prompt():
    directory = input("Enter the directory path containing image files: ")
    save_format = input("Choose output format (JSON/CSV): ").upper()
    return directory, save_format

def save_to_json(result_list, output_file):
    with open(output_file, "w") as json_file:
        json.dump(result_list, json_file, indent=2)

def save_to_csv(result_list, output_file):
    fieldnames = ["image_name", "caption"]
    with open(output_file, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_list)

def main():
    directory, save_format = prompt()

    device = 0  # 0 = GPU, -1 = CPU

    captioner = pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-large",
        device=device,
    )

    paths = get_image_files(directory)

    batch_size = 2  
    total_inference_time = 0
    result_list = []

    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i+batch_size]

        with tqdm(total=len(batch_paths), desc=f"Processing Batch {i // batch_size + 1}/{len(paths) // batch_size}", unit="image") as pbar:
            start_time = time.time()

            try:
                captions = captioner(batch_paths, max_new_tokens=100)
            except PIL.UnidentifiedImageError as ex:
                error_entries = [{"path": path, "error": str(ex)} for path in batch_paths]
                result_list.extend(error_entries)
                pbar.update(len(batch_paths))
                continue

            end_time = time.time()
            batch_inference_time = end_time - start_time
            total_inference_time += batch_inference_time

            for path, caption in zip(batch_paths, captions):
                image_name = os.path.basename(path)
                result_list.append({"image_name": image_name, "caption": caption[0]["generated_text"]})
                pbar.update(1)

    avg_time_per_image = total_inference_time / len(paths)
    print(f"\nTotal Inference Time: {total_inference_time:.2f} seconds")
    print(f"Avg Time Per Image Caption: {avg_time_per_image:.4f} seconds")

    if save_format == "JSON":
        output_file = "2_image_captioning/captions.json"
        save_to_json(result_list, output_file)
    elif save_format == "CSV":
        output_file = "2_image_captioning/captions.csv"
        save_to_csv(result_list, output_file)
    else:
        print("Invalid output format. Please choose JSON or CSV.")

if __name__ == "__main__":
    main()