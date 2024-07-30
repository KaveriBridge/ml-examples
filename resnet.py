import argparse

parser = argparse.ArgumentParser(description="Program with optimization flags")

# Define optional flags with short aliases
parser.add_argument("--accuracy", action="store_true", help="Enable accuracy-focused optimizations")
parser.add_argument("--autocast", action="store_true", help="Enable automatic casting and use BF16")
parser.add_argument("--dynamic", action="store_true", help="Enable dynamic compilation")
parser.add_argument("--compile", action="store_true", help="Use torch compiler")
parser.add_argument("--fp32", action="store_true", help="Default mode")

# Parse arguments from the command line
#args = parser.parse_args(['--fp32'])
args = parser.parse_args()

import torch
import time

from transformers import AutoImageProcessor, ResNetForImageClassification
from datasets import load_dataset

dataset = load_dataset("huggingface/cats-image",trust_remote_code=True)
image = dataset["test"]["image"][0]

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

def bench(model, input, n=100):
# import torch.backends.mkldnn
# with torch.no_grad(), torch.backends.mkldnn.flags(enabled=False):
  with torch.no_grad():
    # Warmup
    for _ in range(10):
      model(**input)
    start = time.time()
    for _ in range(n):
      model(**input)
    end = time.time()
    return((end-start)*1000)/n


from transformers import AutoImageProcessor, ResNetForImageClassification
processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
orig_model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
encoded_input = processor(image, return_tensors="pt")

import json
data = []  # Initialize an empty list to hold JSON data

if (args.fp32):
  with torch.backends.mkldnn.flags(enabled=False):
    avg_time = bench(orig_model, encoded_input)
    data.append({"FP32": f"{avg_time:.2f} ms"})

if (args.dynamic):
  model = torch.ao.quantization.quantize_dynamic(orig_model,{torch.nn.Linear},dtype=torch.qint8)
  avg_time = bench(model, encoded_input)
  data.append({"Dyn Quant": f"{avg_time:.2f} ms"})

if (args.autocast):
  with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
    avg_time = bench(orig_model, encoded_input)
    data.append({"Autocast": f"{avg_time:.2f} ms"})

if (args.compile):
  model = torch.compile(orig_model, backend="inductor")
  avg_time = bench(model, encoded_input)
  data.append({"TorchCompile": f"{avg_time:.2f} ms"})

# Convert the list to a JSON string
json_array = json.dumps(data, indent=4)  # Add indentation for readability (optional)

# Print the JSON array
print(json_array)
