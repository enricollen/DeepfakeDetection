import torch
import requests
from PIL import Image
import time
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to("cuda")

#img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
#raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
raw_image = Image.open("/home/enriconello/DeepFakeDetection/dataset_small/1a1h5j.jpg")

# conditional image captioning
text = "a picture of"
inputs = processor(raw_image, text, return_tensors="pt").to("cuda", torch.float16)

#out = model.generate(**inputs)
#print(processor.decode(out[0], skip_special_tokens=True))

# unconditional image captioning
#inputs = processor(raw_image, return_tensors="pt").to("cuda", torch.float16)

start = time.time()
out = model.generate(**inputs)
end = time.time()
inference_time = end - start

print(f"Inference time: {inference_time:.4f} seconds")
print(processor.decode(out[0], skip_special_tokens=True))
