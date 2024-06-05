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

# Set flags based on parsed arguments
accuracy_enabled = args.accuracy
autocast_enabled = args.autocast
dynamic_enabled = args.dynamic
compile_enabled = args.compile
fp32_enabled = args.fp32


import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_default_device("cpu")

orig_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.float32, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

encoded_input = tokenizer('''def print_prime(n):
   """
   Print all primes between 1 and n
   """''', return_tensors="pt", return_attention_mask=False)

outputs = orig_model.generate(**encoded_input, max_length=200)
text = tokenizer.batch_decode(outputs)[0]
print(text)

# Output should be
# ...

def bench(model, input, n=4):
  with torch.no_grad():
    # Warmup
    for _ in range(2):
      model.generate(**input, max_length=200)
    start = time.time()
    for _ in range(n):
      model.generate(**input, max_length=200)
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



