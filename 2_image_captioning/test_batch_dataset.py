import os
import json
import csv
import PIL
from transformers import pipeline
from tqdm import tqdm
import time
from torch.utils.data import Dataset, DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class ImageCaptionDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.paths = get_image_files(directory)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            image = PIL.Image.open(path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return {"path": path, "image": image}
        except PIL.UnidentifiedImageError as ex:
            return {"path": path, "error": str(ex)}

def get_image_files(directory):
    files = os.listdir(directory)
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    return [os.path.join(directory, f) for f in image_files]

def custom_collate(batch):
    paths = [item["path"] for item in batch]
    images = [item["image"] for item in batch]

    return {"paths": paths, "images": images}

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
    directory = "images"
    save_format = "JSON"
    device = 0  # 0 = GPU, -1 = CPU

    captioner = pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base",
        device=device,
    )

    transform = None  # add image transformations if needed

    dataset = ImageCaptionDataset(directory, transform=transform)
    dataloader = DataLoader(dataset, batch_size=30, shuffle=False, num_workers=4, collate_fn=custom_collate)

    total_inference_time = 0
    result_list = []

    for batch in tqdm(dataloader, desc="Processing Batches", unit="batch"):
        paths = batch["paths"]
        images = batch["images"]

        start_time = time.time()

        try:
            captions = captioner(images, max_new_tokens=100)
        except PIL.UnidentifiedImageError as ex:
            error_entries = [{"path": path, "error": str(ex)} for path in paths]
            result_list.extend(error_entries)
            continue

        end_time = time.time()
        batch_inference_time = end_time - start_time
        total_inference_time += batch_inference_time

        for path, caption in zip(paths, captions):
            image_name = os.path.basename(path)
            result_list.append({"image_name": image_name, "caption": caption[0]["generated_text"]})

    avg_time_per_image = total_inference_time / len(dataset)
    print(f"\nTotal Inference Time: {total_inference_time:.2f} seconds")
    print(f"Avg Time Per Image Caption: {avg_time_per_image:.4f} seconds")

    if save_format == "JSON":
        output_file = "captions.json"
        save_to_json(result_list, output_file)
    elif save_format == "CSV":
        output_file = "captions.csv"
        save_to_csv(result_list, output_file)
    else:
        print("Invalid output format. Please choose JSON or CSV.")

if __name__ == "__main__":
    main()