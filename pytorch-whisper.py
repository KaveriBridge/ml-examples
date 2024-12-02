import time
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Load the processor and model
model_name = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Ensure the model is on the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load example audio input
# Replace this with a real WAV/FLAC file or audio input array
import numpy as np
audio_length = 30  # seconds
sample_rate = 16000
audio = np.random.randn(audio_length * sample_rate).astype(np.float32)

# Tokenize the audio input
inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
input_features = inputs.input_features.to(device)

# Define the transcription function
def transcribe(input_features):
    # Generate tokens
    predicted_ids = model.generate(input_features, max_length=128)
    # Decode tokens into text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription

# Benchmarking
def benchmark_transcription(func, input_features, iterations=10):
    start_time = time.time()
    for _ in range(iterations):
        _ = func(input_features)
    end_time = time.time()
    avg_time = (end_time - start_time) / iterations
    print(f"Average inference time over {iterations} runs: {avg_time:.4f} seconds")

# Warm-up the model (for JIT or GPU initialization latency)
_ = transcribe(input_features)

# Benchmark without any additional optimizations
print("Benchmarking regular inference:")
benchmark_transcription(transcribe, input_features)

# Optionally benchmark with torch.compile (if PyTorch 2.0+ is available)
if torch.__version__ >= "2.0":
    print("Benchmarking with torch.compile optimization:")
    optimized_model = torch.compile(model)
    def transcribe_optimized(input_features):
        predicted_ids = optimized_model.generate(input_features, max_length=128)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return transcription

    benchmark_transcription(transcribe_optimized, input_features)
