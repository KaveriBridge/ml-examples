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
parser.add_argument("--tiny", action="store_true", help="Whisper Tiny Model")
parser.add_argument("--small", action="store_true", help="Whisper Small Model")
parser.add_argument("--medium", action="store_true", help="Whisper Medium Model")
parser.add_argument("--large", action="store_true", help="Whisper Large Model")
parser.add_argument("--turbo", action="store_true", help="Whisper Large Turbo Model")


# Parse arguments from the command line
#args = parser.parse_args(['--print', '--dynamic', '--compile'])
args = parser.parse_args()

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np
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
      #chunk_length_s=30,
      #batch_size=16,  # batch size for inference - set based on your device
      torch_dtype=torch_dtype,
      device=device
  )

  if(args.print):
        result = pipe(audio_input, generate_kwargs={"return_timestamps": True})
        print("Transcription:", result["text"])

  with torch.no_grad():
    # Warmup
    for _ in range(10):
      result = pipe(audio_input, generate_kwargs={"return_timestamps": True})
    start = time.time()

    # Benchmark
    for _ in range(n):
      result = pipe(audio_input, generate_kwargs={"return_timestamps": True})
    end = time.time()
    return((end-start)*1000)/n


# Load the model and processor
model_id = "openai/whisper-large-v3-turbo"
if (args.tiny):
  model_id = "openai/whisper-tiny"
elif (args.small):  
  model_id = "openai/whisper-small"
elif (args.medium):
  model_id = "openai/whisper-medium"
elif (args.large):
  model_id = "openai/whisper-large-v3"
elif (args.turbo):
  model_id = "openai/whisper-large-v3-turbo"

orig_model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch.float32, use_safetensors=True)
orig_model.to("cpu")
orig_model.eval()
processor = AutoProcessor.from_pretrained(model_id)

# Create an empty audio file (1 second of silence) and load it
#import soundfile as sf
#empty_audio = np.zeros((16000,), dtype=np.float32)  # 16000 samples for 1 second at 16kHz
#sf.write("empty_audio.wav", empty_audio, 16000)
#audio_input, _ = sf.read("empty_audio.wav")

import librosa
audio_input, _ = librosa.load('OSR_us_000_0010_8k.wav', sr=16000, mono=True)

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
  with torch.autocast(device_type="cpu", dtype=torch.float16):
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
