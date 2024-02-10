import os
import json
import csv
import PIL
from transformers import pipeline
from tqdm import tqdm
import time

class ImageCaptioner:
    def __init__(self, directory, save_format, device=0):
        self.directory = directory
        self.save_format = save_format
        self.device = device
        self.captioner = pipeline(
            "image-to-text",
            model="Salesforce/blip-image-captioning-base",
            device=device,
        )

    def get_image_files(self):
        files = os.listdir(self.directory)
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        return [os.path.join(self.directory, f) for f in image_files]

    def save_to_json(self, result_list, output_file):
        with open(output_file, "w") as json_file:
            json.dump(result_list, json_file, indent=2)

    def save_to_csv(self, result_list, output_file):
        fieldnames = ["image_name", "caption"]
        with open(output_file, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(result_list)

    def process_images(self, batch_size=30):
        paths = self.get_image_files()
        total_inference_time = 0
        result_list = []

        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i+batch_size]

            with tqdm(total=len(batch_paths), desc=f"Processing Batch {i // batch_size + 1}/{len(paths) // batch_size}", unit="image") as pbar:
                start_time = time.time()

                try:
                    captions = self.captioner(batch_paths, max_new_tokens=100)
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

        if self.save_format == "JSON":
            output_file = "captions.json"
            self.save_to_json(result_list, output_file)
        elif self.save_format == "CSV":
            output_file = "captions.csv"
            self.save_to_csv(result_list, output_file)
        else:
            print("Invalid output format. Please choose JSON or CSV.")

if __name__ == "__main__":
    directory = "images"
    save_format = "CSV"
    image_captioner = ImageCaptioner(directory, save_format)
    image_captioner.process_images()
