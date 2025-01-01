import time
from transformers import AutoTokenizer, FlaxAutoModelForSequenceClassification
import jax
import jax.numpy as jnp

# Load tokenizer and model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = FlaxAutoModelForSequenceClassification.from_pretrained(model_name)

# Define a prediction function
def predict(input_ids, attention_mask):
    # Convert inputs to JAX arrays with correct dtype
    input_ids = jnp.array(input_ids, dtype=jnp.int32) 
    attention_mask = jnp.array(attention_mask, dtype=jnp.int32)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits # Pass inputs by name
    probs = jax.nn.softmax(logits, axis=-1)
    return probs

# JIT-compile the prediction function for performance
predict_jit = jax.jit(predict)

# Example input text
texts = ["I love using JAX for machine learning!"] * 32  # Adjust for batch size
inputs = tokenizer(texts, return_tensors="jax", padding=True, truncation=True)

# Benchmark without JIT
start_time = time.time()
for _ in range(100):  # Number of iterations
    _ = predict(inputs.input_ids, inputs.attention_mask)
end_time = time.time()
print(f"Average inference time (no JIT): {(end_time - start_time) / 100:.4f} seconds")

# Benchmark with JIT
start_time = time.time()
for _ in range(100):  # Number of iterations
    _ = predict_jit(inputs.input_ids, inputs.attention_mask)
end_time = time.time()
print(f"Average inference time (with JIT): {(end_time - start_time) / 100:.4f} seconds")
