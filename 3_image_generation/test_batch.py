from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import time

# Function to process a batch of prompts
def process_batch(pipe, prompts):
    with torch.no_grad():
        # Process the batch of prompts
        results = pipe(prompts).images
    return results

model_id = "dreamlike-art/dreamlike-diffusion-1.0"

# Initialize the pipeline
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Batch size
batch_size = 2

# Example list of prompts
all_prompts = [
    "cutest baby cow ive seen, style realistic",
    "a dog sitting on the beach with a stick in its mouth, style realistic",
    "a rabbit in a field with a strange hat, style realistic",
    "a woman in a red shirt with a red hat, style realistic",
]

# Calculate the number of batches
num_batches = len(all_prompts) // batch_size

start_time = time.time()
counter = 0

# Process each batch
for i in range(num_batches + 1):
    start_idx = i * batch_size
    end_idx = (i + 1) * batch_size
    batch_prompts = all_prompts[start_idx:end_idx]

    # Skip empty batches
    if not batch_prompts:
        continue

    # Process the batch
    batch_results = process_batch(pipe, batch_prompts)

    # Save the generated images
    for j, result_image in enumerate(batch_results):
        dst_path = f"output_batch_{i}_image_{j}.png"
        result_image.save(dst_path)
        counter += 1

end_time = time.time()
generation_time = end_time - start_time
minutes, seconds = divmod(generation_time, 60)

print("Generation Time: {:.0f} m {:.2f} s".format(minutes, seconds))
