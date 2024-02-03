from PIL import Image
import torch as th
import os
import pandas as pd
import time
from dotenv import load_dotenv

from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)

load_dotenv()

NUM_IMAGES = int(os.getenv("NUM_IMAGES"))
START_INDEX = int(os.getenv("START_INDEX"))
END_INDEX = int(os.getenv("END_INDEX"))
NUM_INFERENCE_STEPS = os.getenv("NUM_INFERENCE_STEPS")
GUIDANCE_SCALE = float(os.getenv("GUIDANCE_SCALE"))
INPUT_CSV_PATH = os.getenv("INPUT_CSV_PATH")
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER")
PROMPT_COLUMN_NAME = os.getenv("PROMPT_COLUMN_NAME")
IDS_COLUMN_NAME = os.getenv("IDS_COLUMN_NAME")
NEGATIVE_PROMPTS = os.getenv("NEGATIVE_PROMPTS")


def save_images(batch: th.Tensor, output_path, image_ids):
    scaled = ((batch + 1) * 127.5).round().clamp(0, 255).to(th.uint8).cpu()
    reshaped = scaled.permute(0, 2, 3, 1)

    os.makedirs(output_path, exist_ok=True)

    for index, (image, image_id) in enumerate(zip(reshaped, image_ids)):
        img = Image.fromarray(image.numpy())
        filename = f"GL_fake_{image_id}.jpg"
        img.save(os.path.join(output_path, filename))


def read_prompts_from_csv(csv_file_path):
    df = pd.read_csv(csv_file_path)
    prompts = df[PROMPT_COLUMN_NAME].tolist()
    image_ids = df[IDS_COLUMN_NAME].tolist()
    return prompts, image_ids


def model_fn(x_t, ts, **kwargs):
    half = x_t[: len(x_t) // 2]
    combined = th.cat([half, half], dim=0)
    model_out = model(combined, ts, **kwargs)
    eps, rest = model_out[:, :3], model_out[:, 3:]
    cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
    half_eps = uncond_eps + GUIDANCE_SCALE * (cond_eps - uncond_eps)
    eps = th.cat([half_eps, half_eps], dim=0)
    return th.cat([eps, rest], dim=1)


has_cuda = th.cuda.is_available()
device = th.device('cpu' if not has_cuda else 'cuda')

# base model.
options = model_and_diffusion_defaults()
options['use_fp16'] = has_cuda
options['timestep_respacing'] = NUM_INFERENCE_STEPS
model, diffusion = create_model_and_diffusion(**options)
model.eval()
if has_cuda:
    model.convert_to_fp16()
model.to(device)
model.load_state_dict(load_checkpoint('base', device))

# upsampler model.
options_up = model_and_diffusion_defaults_upsampler()
options_up['use_fp16'] = has_cuda
options_up['timestep_respacing'] = 'fast27' 
model_up, diffusion_up = create_model_and_diffusion(**options_up)
model_up.eval()
if has_cuda:
    model_up.convert_to_fp16()
model_up.to(device)
model_up.load_state_dict(load_checkpoint('upsample', device))


def txt2img(START_INDEX, END_INDEX):
    prompts, image_ids = read_prompts_from_csv(INPUT_CSV_PATH)
    prompts = prompts[START_INDEX:END_INDEX]
    prompts = [prompt + ", photo, real" for prompt in prompts]
    image_ids = image_ids[START_INDEX:END_INDEX]

    # Sampling parameters
    batch_size = END_INDEX - START_INDEX

    # Tune this parameter to control the sharpness of 256x256 images.
    # A value of 1.0 is sharper, but sometimes results in grainy artifacts.
    upsample_temp = 0.997

    ##############################
    # Sample from the base model #
    ##############################

    rows = len(prompts)

    start_time = time.time()

    generated_images = []

    for index, prompt in enumerate(prompts):
        # Create the text tokens to feed to the model.
        tokens = model.tokenizer.encode(prompt)
        tokens, mask = model.tokenizer.padded_tokens_and_mask(
            tokens, options['text_ctx']
        )

        # Create the classifier-free guidance tokens (empty)
        full_batch_size = batch_size * 2
        uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask(
            [], options['text_ctx']
        )

        # Pack the tokens together into model kwargs.
        model_kwargs = dict(
            tokens=th.tensor(
                [tokens] * batch_size + [uncond_tokens] * batch_size, device=device
            ),
            mask=th.tensor(
                [mask] * batch_size + [uncond_mask] * batch_size,
                dtype=th.bool,
                device=device,
            ),
        )

        # Sample from the base model.
        model.del_cache()
        samples = diffusion.p_sample_loop(
            model_fn,
            (full_batch_size, 3, options["image_size"], options["image_size"]),
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]
        model.del_cache()

        ##############################
        # Upsample the 64x64 samples #
        ##############################

        tokens = model_up.tokenizer.encode(prompt)
        tokens, mask = model_up.tokenizer.padded_tokens_and_mask(
            tokens, options_up['text_ctx']
        )

        # Create the model conditioning dict.
        model_kwargs = dict(
            # Low-res image to upsample.
            low_res=((samples+1)*127.5).round()/127.5 - 1,

            # Text tokens
            tokens=th.tensor(
                [tokens] * batch_size, device=device
            ),
            mask=th.tensor(
                [mask] * batch_size,
                dtype=th.bool,
                device=device,
            ),
        )

        # Sample from the base model.
        model_up.del_cache()
        up_shape = (batch_size, 3, options_up["image_size"], options_up["image_size"])
        up_samples = diffusion_up.ddim_sample_loop(
            model_up,
            up_shape,
            noise=th.randn(up_shape, device=device) * upsample_temp,
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]
        model_up.del_cache()

        # Save the output
        save_images(up_samples, OUTPUT_FOLDER, image_ids)

        generated_images.extend([(f"GL_fake_{image_id}", image_id) for image_id in image_ids])

        if index % 100:
            print("Generated", str(index), "/", str(rows))

    end_time = time.time()

    # Calculate the total time
    total_time_seconds = end_time - start_time

    # Convert total time to hours, minutes, and seconds
    total_hours, remainder = divmod(total_time_seconds, 3600)
    total_minutes, total_seconds = divmod(remainder, 60)

    print(f"Total Generation Time: {int(total_hours)}h {int(total_minutes)}m {total_seconds:.2f}s")

    print("\nGeneration completed.")

    # Combine generated image information with the original DataFrame
    generated_df = pd.DataFrame(generated_images, columns=['fake_id', 'original_id'])
    original_df = pd.read_csv(INPUT_CSV_PATH)
    original_df = original_df[START_INDEX:END_INDEX]

    # Merge the original DataFrame with the generated DataFrame
    merged_df = pd.merge(original_df, generated_df, left_on=IDS_COLUMN_NAME, right_on='original_id', how='left')

    # Drop the 'original_id' column
    merged_df = merged_df.drop(columns=['original_id'])
    
    merged_df['class'] = "fake"

    # Save the merged DataFrame to a new CSV file
    merged_df.to_csv('generated_images.csv', index=False)

txt2img(START_INDEX, END_INDEX)