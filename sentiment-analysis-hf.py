import time
import numpy as np
import torch
from transformers import pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def benchmark(pipe, input, runs=1000):
    # Warmup
    for i in range(100):
        result = pipe(input)
    # Benchmark runs
    times = []
    for i in range(runs):
        start = time.time()
        result = pipe(input)
        end = time.time()
        times.append(end - start)
    # Calculate mean and 99 percentile values
    mean = np.mean(times) * 1000
    p99 = np.percentile(times, 99) * 1000
    return "{:.1f}".format(mean), "{:.1f}".format(p99)


short_review = "I'm extremely satisfied with my new Ikea Kallax; It's an excellent storage solution for our kids. A definite must have."
long_review = "We were in search of a storage solution for our kids, and their desire to personalize their storage units led us to explore various options. After careful consideration, we decided on the Ikea Kallax system. It has proven to be an ideal choice for our needs. The flexibility of the Kallax design allows for extensive customization. Whether itâ€™s choosing vibrant colors, adding inserts for specific items, or selecting different finishes, the possibilities are endless. We appreciate that it caters to our kidsâ€™ preferences and encourages their creativity. Overall, the boys are thrilled with the outcome. A great value for money."

models = ["distilbert-base-uncased", "bert-base-uncased", "roberta-base"]

for model in models:
    pipe = pipeline("sentiment-analysis", model=model, device=device)
    result = benchmark(pipe, long_review)
    print(f"{model} long sentence: {result}")
