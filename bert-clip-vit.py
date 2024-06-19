import argparse

parser = argparse.ArgumentParser(description="Program with optimization flags")

# Define optional flags with short aliases
parser.add_argument("--accuracy", action="store_true", help="Enable accuracy-focused optimizations")
parser.add_argument("--autocast", action="store_true", help="Enable automatic casting and use BF16")
parser.add_argument("--dynamic", action="store_true", help="Enable dynamic compilation")
parser.add_argument("--compile", action="store_true", help="Use torch compiler")
parser.add_argument("--fp32", action="store_true", help="Default mode")
parser.add_argument("--bertbase", action="store_true", help=" Run BERT base")
parser.add_argument("--bertlarge", action="store_true", help=" Run BERT large")
parser.add_argument("--clipvit", action="store_true", help="Run VIT")

# Parse arguments from the command line
#args = parser.parse_args(['--fp32', '--bertlarge'])
args = parser.parse_args()

import torch
import time

def bench(model, input, n=100):
  with torch.no_grad():
    # Warmup
    for _ in range(10):
      model(**input)
    start = time.time()
    for _ in range(n):
      model(**input)
    end = time.time()
    return((end-start)*1000)/n

if (args.bertbase):
  from transformers import BertTokenizer, BertModel
  device = torch.device("cpu")
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  orig_model = BertModel.from_pretrained('bert-base-uncased').to(device=device)
  orig_model.eval()
  text = "Replace me with any text you'd like. " * 14
  print(f"Sequence length: {len(text)}")
  encoded_input = tokenizer(text, return_tensors='pt').to(device=device)

if (args.bertlarge):
  from transformers import BertTokenizer, BertModel
  device = torch.device("cpu")
  tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
  orig_model = BertModel.from_pretrained('bert-large-uncased').to(device=device)
  orig_model.eval()
  text = "Replace me with any text you'd like. " * 14
  print(f"Sequence length: {len(text)}")
  encoded_input = tokenizer(text, return_tensors='pt').to(device=device)

if (args.clipvit):
  from transformers import CLIPProcessor, CLIPModel
  orig_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
  processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

  from PIL import Image
  import requests
  url = "http://images.cocodataset.org/val2017/000000039769.jpg"
  image = Image.open(requests.get(url, stream=True).raw)
  encoded_input = processor(text=["a photo of a puma", "a photo of a donkey"], images=image, return_tensors="pt", padding=True)

import json
data = []  # Initialize an empty list to hold JSON data

if (args.fp32):
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
