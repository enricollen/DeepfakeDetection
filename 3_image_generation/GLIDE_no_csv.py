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

#os.environ["CUDA_VISIBLE_DEVICES"] = "5"

class GLIDE:
    def __init__(self) -> None:
        
        load_dotenv()

        self.START_INDEX = int(os.getenv("START_INDEX"))
        self.END_INDEX = int(os.getenv("END_INDEX"))
        self.NUM_INFERENCE_STEPS = os.getenv("NUM_INFERENCE_STEPS")
        self.GUIDANCE_SCALE = float(os.getenv("GUIDANCE_SCALE"))
        self.INPUT_CSV_PATH = os.getenv("INPUT_CSV_PATH")
        self.OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER")
        self.PROMPT_COLUMN_NAME = os.getenv("PROMPT_COLUMN_NAME")
        self.IDS_COLUMN_NAME = os.getenv("IDS_COLUMN_NAME")
        self.NEGATIVE_PROMPTS = os.getenv("NEGATIVE_PROMPTS")

        self.load_models()


    def save_images(self, batch: th.Tensor, output_path, image_ids):
        scaled = ((batch + 1) * 127.5).round().clamp(0, 255).to(th.uint8).cpu()
        reshaped = scaled.permute(0, 2, 3, 1)

        os.makedirs(output_path, exist_ok=True)

        for (reshaped_image, image_id) in zip(reshaped, image_ids):
            img = Image.fromarray(reshaped_image.numpy())
            filename = f"GL_fake_{image_id}.jpg"
            img.save(os.path.join(output_path, filename))



    def read_prompts_from_csv(self, csv_file_path):
        df = pd.read_csv(csv_file_path)
        prompts = df[self.PROMPT_COLUMN_NAME].tolist()
        image_ids = df[self.IDS_COLUMN_NAME].tolist()
        return prompts, image_ids


    def model_fn(self, x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = th.cat([half, half], dim=0)
        model_out = self.model_base(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + self.GUIDANCE_SCALE * (cond_eps - uncond_eps)
        eps = th.cat([half_eps, half_eps], dim=0)
        return th.cat([eps, rest], dim=1)

    def load_models(self):
        has_cuda = th.cuda.is_available()
        self.device = th.device('cpu' if not has_cuda else 'cuda')

        # base model.
        self.options = model_and_diffusion_defaults()
        self.options['use_fp16'] = has_cuda
        self.options['timestep_respacing'] = self.NUM_INFERENCE_STEPS
        self.model_base, self.diffusion = create_model_and_diffusion(**self.options)
        self.model_base.eval()
        if has_cuda:
            self.model_base.convert_to_fp16()
        self.model_base.to(self.device)
        self.model_base.load_state_dict(load_checkpoint('base', self.device))

        # upsampler model.
        self.options_up = model_and_diffusion_defaults_upsampler()
        self.options_up['use_fp16'] = has_cuda
        self.options_up['timestep_respacing'] = 'fast27' 
        self.model_up, self.diffusion_up = create_model_and_diffusion(**self.options_up)
        self.model_up.eval()
        if has_cuda:
            self.model_up.convert_to_fp16()
        self.model_up.to(self.device)
        self.model_up.load_state_dict(load_checkpoint('upsample', self.device))


    def txt2img(self):
        prompts, image_ids = self.read_prompts_from_csv(self.INPUT_CSV_PATH)
        prompts = prompts[self.START_INDEX:self.END_INDEX]
        prompts = [prompt + ", photo" for prompt in prompts]
        image_ids = image_ids[self.START_INDEX:self.END_INDEX]

        # Sampling parameters
        batch_size = self.END_INDEX - self.START_INDEX

        # Tune this parameter to control the sharpness of 256x256 images.
        # A value of 1.0 is sharper, but sometimes results in grainy artifacts.
        upsample_temp = 0.998

        ##############################
        # Sample from the base model #
        ##############################

        rows = len(prompts)

        start_time = time.time()

        for prompt, image_id in zip(prompts, image_ids):
            
            print(f"Generating for prompt: {prompt}")
            # Create the text tokens to feed to the model.
            tokens = self.model_base.tokenizer.encode(prompt)
            tokens, mask = self.model_base.tokenizer.padded_tokens_and_mask(
                tokens, self.options['text_ctx']
            )

            # Create the classifier-free guidance tokens (empty)
            full_batch_size = batch_size * 2
            uncond_tokens, uncond_mask = self.model_base.tokenizer.padded_tokens_and_mask(
                [], self.options['text_ctx']
            )

            # Pack the tokens together into model kwargs.
            model_kwargs = dict(
                tokens=th.tensor(
                    [tokens] * batch_size + [uncond_tokens] * batch_size, device=self.device
                ),
                mask=th.tensor(
                    [mask] * batch_size + [uncond_mask] * batch_size,
                    dtype=th.bool,
                    device=self.device,
                ),
            )

            # Sample from the base model.
            self.model_base.del_cache()
            samples = self.diffusion.p_sample_loop(
                self.model_fn,
                (full_batch_size, 3, self.options["image_size"], self.options["image_size"]),
                device=self.device,
                clip_denoised=True,
                progress=True,
                model_kwargs=model_kwargs,
                cond_fn=None,
            )[:batch_size]
            self.model_base.del_cache()

            ##############################
            # Upsample the 64x64 samples #
            ##############################

            tokens = self.model_up.tokenizer.encode(prompt)
            tokens, mask = self.model_up.tokenizer.padded_tokens_and_mask(
                tokens, self.options_up['text_ctx']
            )

            # Create the model conditioning dict.
            model_kwargs = dict(
                # Low-res image to upsample.
                low_res=((samples+1)*127.5).round()/127.5 - 1,

                # Text tokens
                tokens=th.tensor(
                    [tokens] * batch_size, device=self.device
                ),
                mask=th.tensor(
                    [mask] * batch_size,
                    dtype=th.bool,
                    device=self.device,
                ),
            )

            # Sample from the base model.
            self.model_up.del_cache()
            up_shape = (batch_size, 3, self.options_up["image_size"], self.options_up["image_size"])
            up_samples = self.diffusion_up.ddim_sample_loop(
                self.model_up,
                up_shape,
                noise=th.randn(up_shape, device=self.device) * upsample_temp,
                device=self.device,
                clip_denoised=True,
                progress=True,
                model_kwargs=model_kwargs,
                cond_fn=None,
            )[:batch_size]
            self.model_up.del_cache()

            # Save the output
            self.save_images(up_samples, self.OUTPUT_FOLDER, [image_id])

            print(f"Generated image name: GL_fake_{image_id}")

        end_time = time.time()

        # Calculate the total time
        total_time_seconds = end_time - start_time

        # Convert total time to hours, minutes, and seconds
        total_hours, remainder = divmod(total_time_seconds, 3600)
        total_minutes, total_seconds = divmod(remainder, 60)

        print(f"Total Generation Time: {int(total_hours)}h {int(total_minutes)}m {total_seconds:.2f}s")

        print("\nGeneration completed.")



if __name__ == "__main__":
    model = GLIDE()

    # Set the batch size for each iteration
    batch_size = int(os.getenv("BATCH_SIZE"))

    start_index = model.START_INDEX
    end_index = model.END_INDEX

    while start_index < end_index:
        model.START_INDEX = start_index
        model.END_INDEX = min(start_index + batch_size, end_index)

        # Call the txt2img method for the current batch
        model.txt2img()

        # Update start_index for the next iteration
        start_index += batch_size