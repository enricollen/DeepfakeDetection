import os
import torch
import time
from diffusers import StableDiffusionPipeline
import pandas as pd
from torch.cuda.amp import autocast
from dotenv import load_dotenv

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

load_dotenv()

NUM_IMAGES = int(os.getenv("NUM_IMAGES"))
START_INDEX = int(os.getenv("START_INDEX"))
END_INDEX = int(os.getenv("END_INDEX"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
NUM_INFERENCE_STEPS = int(os.getenv("NUM_INFERENCE_STEPS"))
INPUT_CSV_PATH = os.getenv("INPUT_CSV_PATH")
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER")
MODEL = os.getenv("MODEL")
PROMPT_COLUMN_NAME = os.getenv("PROMPT_COLUMN_NAME")
IDS_COLUMN_NAME = os.getenv("IDS_COLUMN_NAME")
NEGATIVE_PROMPTS = os.getenv("NEGATIVE_PROMPTS")

class Txt2ImgGenerator:
    def __init__(self, model_name, input_csv_path=INPUT_CSV_PATH, output_folder=OUTPUT_FOLDER, batch_size=BATCH_SIZE, num_inference_steps=NUM_INFERENCE_STEPS):
        self.model = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            local_files_only=False,
            use_safetensors=True,
            safety_checker = None,
            requires_safety_checker = False
        )
        self.model.enable_vae_slicing()
        self.model = self.model.to("cuda")
        self.model_name = 'SD' if model_name == "stabilityai/stable-diffusion-2" else "DL"

        self.prompts, self.images_ids = self.read_prompts_from_csv(input_csv_path, PROMPT_COLUMN_NAME, IDS_COLUMN_NAME)
        self.prompts = [prompt + ", photo, real" for prompt in self.prompts]

        self.batch_size = batch_size
        self.num_inference_steps = num_inference_steps
        self.output_folder = output_folder
        self.generated_image_info = []  # list to store information for the final CSV to generate

        os.makedirs(self.output_folder, exist_ok=True)

        # enable Automatic Mixed Precision (AMP)
        self.scaler = torch.cuda.amp.GradScaler()

    def read_prompts_from_csv(self, csv_file_path, prompts_column_name, ids_column_name):
        df = pd.read_csv(csv_file_path)
        prompts = df[prompts_column_name].tolist()
        image_ids = df[ids_column_name].tolist()
        return prompts, image_ids

    def divide_chunks(self, lst1, lst2, chunk_size):
        for i in range(0, len(lst1), chunk_size):
            yield lst1[i:i + chunk_size], lst2[i:i + chunk_size]

    def generate_images(self, start_index, end_index):
        
        prompts_slice = self.prompts[start_index:end_index]
        images_ids_slice = self.images_ids[start_index:end_index]

        total_images = len(prompts_slice)
        batches = list(self.divide_chunks(prompts_slice, images_ids_slice, self.batch_size))

        print("Total prompts:", total_images)
        counter = 0

        start_time = time.time()

        # negative prompts must match the batch size of prompts
        #negative_prompts_batch = [NEGATIVE_PROMPTS] * self.batch_size

        for index, (batch_prompts, batch_image_ids) in enumerate(batches):
            print("Processing batch {}/{}".format(index + 1, len(batches)))

            negative_prompts_batch = [NEGATIVE_PROMPTS] * len(batch_prompts)  # Match batch size

            # generate images for the batch of prompts
            with autocast(enabled=True):
                generated_images = self.model(batch_prompts, negative_prompt=negative_prompts_batch, num_inference_steps=self.num_inference_steps)["images"]

            # save the generated images and update generated_image_info
            for i in range(len(batch_prompts)):
                image_id = batch_image_ids[i].split("_")[0]
                fake_id = self.model_name + "_fake_" + image_id
                dst_path = os.path.join(self.output_folder, fake_id + ".jpg")
                counter += 1
                generated_images[i].save(dst_path)
                self.generated_image_info.append({'id': image_id, 'fake_id': fake_id})

            batch_end_time = time.time()
            batch_time = batch_end_time - start_time
            minutes, seconds = divmod(batch_time, 60)
            # print("Generation Time: {:.0f} m {:.2f} s".format(minutes, seconds))

        total_end_time = time.time()
        total_time = total_end_time - start_time
        minutes, seconds = divmod(total_time, 60)
        print("Total Generation Time: {:.0f} m {:.2f} s".format(minutes, seconds))

        # specify directory for saving the CSV
        folder_name = os.path.basename(self.output_folder)
        destination_dir = os.path.join(self.output_folder, f'{folder_name}_{self.model_name}.csv')

        # Convert generated_image_info to DataFrame and save as new CSV
        generated_info_df = pd.DataFrame(self.generated_image_info)
        input_df = pd.read_csv(INPUT_CSV_PATH)
        result_df = pd.merge(input_df.iloc[START_INDEX:END_INDEX], generated_info_df, on='id', how='left')
        result_df['class'] = 'fake' # from 'pristine' now they will be 'fake'

        result_df.to_csv(destination_dir, index=False)

        print("\nGeneration completed.")


if __name__ == "__main__":
    SD = Txt2ImgGenerator(
        model_name=MODEL,
        input_csv_path=INPUT_CSV_PATH,
        output_folder=OUTPUT_FOLDER,
        batch_size=BATCH_SIZE,
        num_inference_steps=NUM_INFERENCE_STEPS
    )

    SD.generate_images(START_INDEX, END_INDEX)
