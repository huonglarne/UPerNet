import requests
import torch
from PIL import Image
from transformers import UperNetForSemanticSegmentation, AutoImageProcessor


model = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-swin-large")

url = "https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg"
image = Image.open(requests.get(url, stream=True).raw)
image 

processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-swin-large")
pixel_values = processor(image, return_tensors="pt").pixel_values
print(pixel_values.shape)

with torch.no_grad():
  outputs = model(pixel_values)
  print(outputs.logits.shape)