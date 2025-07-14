# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.pipelines import pipeline
from time import time


tokenizer = AutoTokenizer.from_pretrained("models/")
model = AutoModelForSequenceClassification.from_pretrained("models/")

pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device="cuda")

t = time()
result = pipe("He never went out without a book under his arm")
print(result)
print(f"Time takes: {time() - t}") # Time takes: 0.3239879608154297