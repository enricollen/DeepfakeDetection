import pandas as pd
import os
from dotenv import load_dotenv
from transformers import pipeline
from tqdm import tqdm
import time

load_dotenv()

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class ImageCaptioner:
    def __init__(self, device=0, model="Salesforce/blip-image-captioning-base"):
        self.device = device
        self.captioner = pipeline(
            "image-to-text",
            model=model,
            device=device,
        )

    def process_images(self, image_paths, batch_size=500, error_file="error_images.txt"):
        total_inference_time = 0
        result_dict = {"image_name": [], "caption": []}

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]

            with tqdm(total=len(batch_paths),
                    desc=f"Processing Batch {i // batch_size + 1}/{len(image_paths) // batch_size}", unit="image") as pbar:
                start_time = time.time()

                try:
                    captions = self.captioner(batch_paths, max_new_tokens=100)
                except Exception as ex:
                    if not os.path.exists(error_file):
                        with open(error_file, "w"):
                            pass 

                    with open(error_file, "a") as error_file_writer:
                        error_file_writer.write("\n".join(batch_paths) + "\n")
                    pbar.update(len(batch_paths))
                    continue  # Skip to the next iteration if an exception occurs

                end_time = time.time()
                batch_inference_time = end_time - start_time
                total_inference_time += batch_inference_time

                for path, caption in zip(batch_paths, captions):
                    image_name = os.path.splitext(os.path.basename(path))[0]  # remove .jpg
                    result_dict["image_name"].append(image_name)
                    result_dict["caption"].append(caption[0]["generated_text"])
                    pbar.update(1)

        avg_time_per_image = total_inference_time / len(image_paths)
        print(f"\nTotal Inference Time: {total_inference_time:.2f} seconds")
        print(f"Avg Time Per Image Caption: {avg_time_per_image:.4f} seconds")

        result_df = pd.DataFrame(result_dict)
        return result_df

def caption_and_save_images(df, save_path):
    # Extract paths of selected pristine images
    pristine_paths = [os.path.join(DATASET_DIR, f"{image_name}.jpg") for image_name in df['id']]

    # Process and caption images
    generated_captions_df = captioner.process_images(image_paths=pristine_paths, batch_size=BATCH_SIZE)

    # Merge the original captions from df with the generated captions
    result_df = pd.merge(df, generated_captions_df, left_on="id", right_on="image_name", how="left")

    # Create a new DataFrame with 'id', 'original_caption', and 'caption'
    result_df = result_df[['id', 'clean_title', 'caption']]

    result_df = result_df.rename(columns={'clean_title': 'original_caption', 'caption': 'generated_caption'})
    result_df.to_csv(save_path, index=False, header=True, sep=',', encoding='utf-8')
    

captioner = ImageCaptioner(model="Salesforce/blip-image-captioning-large")
DATASET_DIR = os.getenv('DATASET_DIR')
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))

selected_train_pristine_csv_path = os.getenv('SELECTED_TRAIN_PRISTINE_CSV_PATH')
selected_train_pristine = pd.read_csv(selected_train_pristine_csv_path, sep=',')

caption_and_save_images(selected_train_pristine, "csv/training_pristine_captioned.csv")