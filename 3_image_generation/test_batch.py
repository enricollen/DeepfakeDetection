import os
from diffusers import StableDiffusionPipeline
import torch
import time
from dotenv import load_dotenv

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

load_dotenv()

BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
NUM_INFERENCE_STEPS = int(os.getenv("NUM_INFERENCE_STEPS"))

def read_prompts_from_file(file_path):
    with open(file_path, "r") as file:
        prompts = [line.strip() for line in file.readlines()]
    return prompts

def process_batch(pipe, prompts):
    with torch.no_grad():
        # Process the batch of prompts
        results = pipe(prompts, num_inference_steps=NUM_INFERENCE_STEPS).images
    return results

model_id = "dreamlike-art/dreamlike-diffusion-1.0" #"dreamlike-art/dreamlike-diffusion-1.0"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

all_prompts = read_prompts_from_file("prompts.txt")
all_prompts = ["photo of " + prompt for prompt in all_prompts]

num_batches = len(all_prompts) // BATCH_SIZE

start_time = time.time()
counter = 0

output_directory = "generated_images"
os.makedirs(output_directory, exist_ok=True)

# Process each batch
for i in range(num_batches + 1):
    start_idx = i * BATCH_SIZE
    end_idx = (i + 1) * BATCH_SIZE
    batch_prompts = all_prompts[start_idx:end_idx]

    if not batch_prompts:
        continue

    # Process the batch
    batch_results = process_batch(pipe, batch_prompts)

    for prompt, result_image in zip(batch_prompts, batch_results):
        # Use the prompt to create a safe filename
        prompt_for_filename = prompt.replace(" ", "_").replace("/", "_").replace(":", "_").replace(",", "_").replace(".", "_")

        dst_path = f"generated_images/{prompt_for_filename}.png"
        result_image.save(dst_path)
        counter += 1


end_time = time.time()
generation_time = end_time - start_time
minutes, seconds = divmod(generation_time, 60)

print("Generation Time: {:.0f} m {:.2f} s".format(minutes, seconds))
