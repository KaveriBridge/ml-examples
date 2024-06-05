import argparse

parser = argparse.ArgumentParser(description="Program with optimization flags")

# Define optional flags with short aliases
parser.add_argument("--accuracy", action="store_true", help="Enable accuracy-focused optimizations")
parser.add_argument("--autocast", action="store_true", help="Enable automatic casting and use BF16")
parser.add_argument("--dynamic", action="store_true", help="Enable dynamic compilation")
parser.add_argument("--compile", action="store_true", help="Use torch compiler")
parser.add_argument("--fp32", action="store_true", help="Default mode")
parser.add_argument("--base", action="store_true", help="Run T5-base")
parser.add_argument("--small", action="store_true", help="Run T5-small")

# Parse arguments from the command line
#args = parser.parse_args(['--fp32', "--dynamic", "--fastmath"])
args = parser.parse_args()

# Set flags based on parsed arguments
accuracy_enabled = args.accuracy
autocast_enabled = args.autocast
dynamic_enabled = args.dynamic
compile_enabled = args.compile
fp32_enabled = args.fp32
base_enabled = args.base
small_enabled = args.small

# Your program logic here, using the enabled flags
import torch
import time

from transformers import T5Tokenizer, T5ForConditionalGeneration
# Define model and tokenizer
if(base_enabled == true):
  model_name = "t5-base"
if(small_enabled == true):
  model_name = "t5-small"

tokenizer = T5Tokenizer.from_pretrained(model_name)
orig_model = T5ForConditionalGeneration.from_pretrained(model_name)

# English sentence to translate
text = "This is a sentence to be translated to French."

def bench(model, tokenizer, text, n=20):
  with torch.no_grad():

    # Warmup
    for _ in range(1):
      output_ids = model.generate(tokenizer.encode(text, return_tensors="pt"))
      translated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
      # Output should be "Cette phrase doit être traduite en français" for t5-base
      # Output should be "Cette sentence est une sentence qui est traduit en français." for t5-small
      print(f"Translated Text: {translated_text}")

    start = time.time()
    for _ in range(n):
      model.generate(tokenizer.encode(text, return_tensors="pt"))
    end = time.time()
    return((end-start)*1000)/n

import json
data = []  # Initialize an empty list to hold JSON data

if fp32_enabled:
  avg_time = bench(orig_model, tokenizer, text)
  data.append({"FP32": f"{avg_time:.2f} ms"})

if dynamic_enabled:
  model = torch.ao.quantization.quantize_dynamic(orig_model,{torch.nn.Linear},dtype=torch.qint8)
  avg_time = bench(model, tokenizer, text)
  data.append({"Dyn Quant": f"{avg_time:.2f} ms"})

if autocast_enabled:
  with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
    avg_time = bench(orig_model, tokenizer, text)
    data.append({"Autocast": f"{avg_time:.2f} ms"})

if compile_enabled:
  model = torch.compile(orig_model, backend="inductor")
  avg_time = bench(model, tokenizer, text)
  data.append({"TorchCompile": f"{avg_time:.2f} ms"})

# Convert the list to a JSON string
json_array = json.dumps(data, indent=4)  # Add indentation for readability (optional)

# Print the JSON array
print(json_array)
