import argparse

parser = argparse.ArgumentParser(description="Program with optimization flags")

# Define optional flags with short aliases
parser.add_argument("--accuracy", action="store_true", help="Enable accuracy-focused optimizations")
parser.add_argument("--autocast", action="store_true", help="Enable automatic casting and use BF16")
parser.add_argument("--dynamic", action="store_true", help="Enable dynamic compilation")
parser.add_argument("--compile", action="store_true", help="Use torch compiler")
parser.add_argument("--fp32", action="store_true", help="Default mode")

# Parse arguments from the command line
#args = parser.parse_args(['--fp32', "--dynamic", "--fastmath"])
args = parser.parse_args()

# Set flags based on parsed arguments
accuracy_enabled = args.accuracy
autocast_enabled = args.autocast
dynamic_enabled = args.dynamic
compile_enabled = args.compile
fp32_enabled = args.fp32


import torch
import time
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# you can specify the revision tag if you don't want the timm dependency
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
orig_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

encoded_input = processor(images=image, return_tensors="pt")
outputs = orig_model(**encoded_input)

# convert outputs (bounding boxes and class logits) to COCO API
# let's only keep detections with score > 0.9
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
            f"Detected {orig_model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
    )

# Output should be
# Detected remote with confidence 0.998 at location [40.16, 70.81, 175.55, 117.98]
# Detected remote with confidence 0.996 at location [333.24, 72.55, 368.33, 187.66]
# Detected couch with confidence 0.995 at location [-0.02, 1.15, 639.73, 473.76]
# Detected cat with confidence 0.999 at location [13.24, 52.05, 314.02, 470.93]
# Detected cat with confidence 0.999 at location [345.4, 23.85, 640.37, 368.72]

def bench(model, input, n=20):
  with torch.no_grad():
    # Warmup
    for _ in range(10):
      model(**input)
    start = time.time()
    for _ in range(n):
      model(**input)
    end = time.time()
    return((end-start)*1000)/n

import json
data = []  # Initialize an empty list to hold JSON data

if fp32_enabled:
  avg_time = bench(orig_model, encoded_input)
  data.append({"FP32": f"{avg_time:.2f} ms"})

if dynamic_enabled:
  model = torch.ao.quantization.quantize_dynamic(orig_model,{torch.nn.Linear},dtype=torch.qint8)
  avg_time = bench(model, encoded_input)
  data.append({"Dyn Quant": f"{avg_time:.2f} ms"})

if autocast_enabled:
  with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
    avg_time = bench(orig_model, encoded_input)
    data.append({"Autocast": f"{avg_time:.2f} ms"})

if compile_enabled:
  model = torch.compile(orig_model, backend="inductor")
  avg_time = bench(model, encoded_input)
  data.append({"TorchCompile": f"{avg_time:.2f} ms"})

# Convert the list to a JSON string
json_array = json.dumps(data, indent=4)  # Add indentation for readability (optional)

# Print the JSON array
print(json_array)

