#!pip install transformers datasets flax jax jaxlib
import jax
import jax.numpy as jnp
from transformers import FlaxAutoModelForImageClassification, AutoFeatureExtractor

import time
from datasets import load_dataset
from PIL import Image

# Load tokenizer and model
model_name = "google/vit-base-patch16-224"
model = FlaxAutoModelForImageClassification.from_pretrained(model_name)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

# Define a prediction function
def predict(image):
    inputs = feature_extractor(images=image, return_tensors="np")
    logits = model(**inputs).logits
    predicted_class_idx = jnp.argmax(logits, axis=-1)
    return model.config.id2label[predicted_class_idx.item()]

# Load an example image
dataset = load_dataset("huggingface/cats-image",trust_remote_code=True)
image = dataset["test"]["image"][0]

# Predict the class of the example image
predicted_class = predict(image)
print(f"Predicted class: {predicted_class}") # Should be Predicted class: Egyptian cat

# Benchmark without JIT
start_time = time.time()
for _ in range(100):  # Number of iterations
    _ = predict(image)
end_time = time.time()
print(f"Average inference time (no JIT): {(end_time - start_time) / 100:.4f} seconds")

