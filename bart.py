import argparse

parser = argparse.ArgumentParser(description="Program with optimization flags")

# Define optional flags with short aliases
parser.add_argument("--accuracy", action="store_true", help="Enable accuracy-focused optimizations")
parser.add_argument("--autocast", action="store_true", help="Enable automatic casting and use BF16")
parser.add_argument("--dynamic", action="store_true", help="Enable dynamic compilation")
parser.add_argument("--compile", action="store_true", help="Use torch compiler")
parser.add_argument("--fp32", action="store_true", help="Default mode")
parser.add_argument("--distilBART", action="store_true", help=" Run distilBART")
parser.add_argument("--bart", action="store_true", help=" Run BART")
parser.add_argument("--print", action="store_true", help=" print result")


# Parse arguments from the command line
#args = parser.parse_args(['--distilBART','--print', '--autocast'])
args = parser.parse_args()


import torch
import time
from transformers import pipeline, BartForConditionalGeneration, BartTokenizer

def bench(model,tokenizer, text, n=10):
  summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

  if(args.print):
        summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
        print(summary[0]['summary_text'])

  with torch.no_grad():
    # Warmup
    for _ in range(10):
      summarizer(text, max_length=50, min_length=25, do_sample=False)
    start = time.time()
    for _ in range(n):
      summarizer(text, max_length=50, min_length=25, do_sample=False)
    end = time.time()
    return((end-start)*1000)/n

text = """
Since the launch of the original Raspberry Pi in 2012, Arm and Raspberry Pi have shared a vision to make computing accessible for all and delivered critical solutions for the Internet of Things (IoT) developer community. Over the course of this long-term partnership, Raspberry Pi has sold over 60 million Arm-based units to date, lowering barriers to innovation so that anyone, anywhere – from hobbyists to academics to professional developers – can learn, experience and create new solutions. Our collaboration continues to go from strength to strength, with today’s launch of Raspberry Pi Pico 2 (RP2350) marking an exciting next step in this mission..
"""

if (args.distilBART):
  from transformers import BertTokenizer, BertModel
  device = torch.device("cpu")
  orig_model = BartForConditionalGeneration.from_pretrained("sshleifer/distilbart-cnn-12-6")
  tokenizer = BartTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
  orig_model.eval()

if (args.bart):
  from transformers import BertTokenizer, BertModel
  device = torch.device("cpu")
  orig_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device=device)
  tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
  orig_model.eval()


import json
data = []  # Initialize an empty list to hold JSON data

if (args.fp32):
  avg_time = bench(orig_model, tokenizer, text)
  data.append({"FP32": f"{avg_time:.2f} ms"})

if (args.dynamic):
  model = torch.ao.quantization.quantize_dynamic(orig_model,{torch.nn.Linear},dtype=torch.qint8)
  avg_time = bench(model, tokenizer, text)
  data.append({"Dyn Quant": f"{avg_time:.2f} ms"})

if (args.autocast):
  with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
    avg_time = bench(orig_model, tokenizer, text)
    data.append({"Autocast": f"{avg_time:.2f} ms"})

if (args.compile):
  model = torch.compile(orig_model, backend="inductor")
  avg_time = bench(model, tokenizer, text)
  data.append({"TorchCompile": f"{avg_time:.2f} ms"})

# Convert the list to a JSON string
json_array = json.dumps(data, indent=4)  # Add indentation for readability (optional)

# Print the JSON array
print(json_array)
