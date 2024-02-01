import os
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline
import pandas as pd
from torch.cuda.amp import autocast  # FOR FP16
from dotenv import load_dotenv

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

load_dotenv()

BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
NUM_INFERENCE_STEPS = int(os.getenv("NUM_INFERENCE_STEPS"))

class Txt2ImgGenerator:
    def __init__(self, model_name, output_folder, csv_file_path=None, column_name=None, prompts_file_path=None, batch_size=BATCH_SIZE, num_inference_steps=NUM_INFERENCE_STEPS):
        self.model = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            local_files_only=False,
            use_safetensors=True
        )
        self.model.enable_vae_slicing()
        self.model = self.model.to("cuda")

        if prompts_file_path!=None:
            self.prompts = self.read_prompts_from_file(prompts_file_path)
        else:
            self.prompts = self.read_prompts_from_csv(csv_file_path, column_name)

        self.prompts = ["photo of " + prompt for prompt in self.prompts]

        self.batch_size = batch_size
        self.num_inference_steps = num_inference_steps
        self.output_folder = output_folder

        os.makedirs(self.output_folder, exist_ok=True)

        # enable Automatic Mixed Precision (AMP)
        self.scaler = torch.cuda.amp.GradScaler()

    def read_prompts_from_file(self, file_path):
        with open(file_path, "r") as file:
            prompts = [line.strip() for line in file.readlines()]
        return prompts
    
    def read_prompts_from_csv(self, csv_file_path, column_name):
        df = pd.read_csv(csv_file_path)
        prompts = df[column_name].tolist()
        return prompts

    def divide_chunks(self, lst, chunk_size):
        for i in range(0, len(lst), chunk_size):
            yield lst[i:i + chunk_size]

    def generate_images(self):
        total_images = len(self.prompts[:10])
        batches = list(self.divide_chunks(self.prompts[:10], self.batch_size))
        print("Total prompts:", total_images)
        counter = 0

        start_time = time.time()

        for index, batch_prompts in enumerate(batches):
            print("Processing batch {}/{}".format(index + 1, len(batches)))

            # create directories for each prompt
            for prompt in batch_prompts:
                prompt_dir = os.path.join(self.output_folder, prompt)
                os.makedirs(prompt_dir, exist_ok=True)

            # generate images for the batch of prompts
            with autocast(enabled=True):  # FOR FP16
                generated_images = self.model(batch_prompts, num_inference_steps=self.num_inference_steps)["images"]

            # save the generated images
            for i in range(len(batch_prompts)):
                dst_path = os.path.join(self.output_folder, batch_prompts[i], "fake" + str(counter) + ".png")
                counter += 1
                generated_images[i].save(dst_path)

            batch_end_time = time.time()
            batch_time = batch_end_time - start_time
            minutes, seconds = divmod(batch_time, 60)
            print("Batch generation Time: {:.0f} m {:.2f} s".format(minutes, seconds))

        total_end_time = time.time()
        total_time = total_end_time - start_time
        minutes, seconds = divmod(total_time, 60)
        print("Total Generation Time: {:.0f} m {:.2f} s".format(minutes, seconds))

        print("\nGeneration completed.")

    def create_image_grid(self, images, titles, rows, cols, figsize=(15, 15)):
        fig, axs = plt.subplots(rows, cols, figsize=figsize)
        axs = axs.flatten()

        for img, title, ax in zip(images, titles, axs):
            ax.imshow(np.array(img))
            ax.set_title(title)
            ax.axis('off')

        plt.show()

if __name__ == "__main__":

    """SD = Txt2ImgGenerator(
        model_name="stabilityai/stable-diffusion-2", #"dreamlike-art/dreamlike-diffusion-1.0"
        output_folder="/home/enriconello/DeepFakeDetection/Thesis/3_image_generation/test/generated_images",
        prompts_file_path="/home/enriconello/DeepFakeDetection/Thesis/3_image_generation/test/prompts.txt"
    )"""

    SD = Txt2ImgGenerator(
        model_name="stabilityai/stable-diffusion-2",
        output_folder="/home/enriconello/DeepFakeDetection/Thesis/3_image_generation/test/generated_images",
        csv_file_path="../2_image_captioning/csv/validation_pristine_captioned.csv",
        column_name="generated_caption"
    )

    SD.generate_images()