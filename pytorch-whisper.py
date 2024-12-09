#!curl -o OSR_us_000_0010_8k.wav https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0010_8k.wav

import argparse

parser = argparse.ArgumentParser(description="Program with optimization flags")

# Define optional flags with short aliases
parser.add_argument("--accuracy", action="store_true", help="Enable accuracy-focused optimizations")
parser.add_argument("--autocast", action="store_true", help="Enable automatic casting and use BF16")
parser.add_argument("--dynamic", action="store_true", help="Enable dynamic compilation")
parser.add_argument("--compile", action="store_true", help="Use torch compiler")
parser.add_argument("--fp32", action="store_true", help="Default mode")
parser.add_argument("--print", action="store_true", help=" print result")


# Parse arguments from the command line
args = parser.parse_args(['--print', '--dynamic', '--compile'])
#args = parser.parse_args()


import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np
import soundfile as sf
import time as time

# Set device to CPU
torch_dtype = torch.float32
device="cpu"

import requests

def bench(model, processor, audio_input, n=10):

  # Create the pipeline
  pipe = pipeline(
      "automatic-speech-recognition",
      model=model,
      tokenizer=processor.tokenizer,
      feature_extractor=processor.feature_extractor,
      torch_dtype=torch_dtype,
      device=device
  )

  if(args.print):
        result = pipe(audio_input)
        print("Transcription:", result["text"])

  with torch.no_grad():
    # Warmup
    for _ in range(10):
      result = pipe(audio_input)
    start = time.time()

    # Benchmark
    for _ in range(n):
      result = pipe(audio_input)
    end = time.time()
    return((end-start)*1000)/n


# Load the model and processor
model_id = "openai/whisper-large-v3-turbo"
orig_model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch.float32)
orig_model.to("cpu")
orig_model.eval()
processor = AutoProcessor.from_pretrained(model_id)

# Create an empty audio file (1 second of silence)
#empty_audio = np.zeros((16000,), dtype=np.float32)  # 16000 samples for 1 second at 16kHz
#sf.write("empty_audio.wav", empty_audio, 16000)

# Load the empty audio file
#audio_input, _ = sf.read("empty_audio.wav")

audio_input, _ = sf.read("OSR_us_000_0010_8k.wav")

import json
data = []  # Initialize an empty list to hold JSON data

if (args.fp32):
  avg_time = bench(orig_model, processor, audio_input, 10)
  data.append({"FP32": f"{avg_time:.2f} ms"})

if (args.dynamic):
  model = torch.ao.quantization.quantize_dynamic(orig_model,{torch.nn.Linear},dtype=torch.qint8)
  avg_time = bench(model, processor, audio_input, 10)
  data.append({"Dyn Quant": f"{avg_time:.2f} ms"})

if (args.autocast):
  with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
    avg_time = bench(orig_model, processor, audio_input, 10)
    data.append({"Autocast": f"{avg_time:.2f} ms"})

if (args.compile):
  model = torch.compile(orig_model, backend="inductor")
  avg_time = bench(model, processor, audio_input, 10)
  data.append({"TorchCompile": f"{avg_time:.2f} ms"})

# Convert the list to a JSON string
json_array = json.dumps(data, indent=4)  # Add indentation for readability (optional)

# Print the JSON array
print(json_array)
