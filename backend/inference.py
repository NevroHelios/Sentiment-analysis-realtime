from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
from time import time
from transformers.pipelines import pipeline

onnx_model_dir = "models\\onnx"

model = ORTModelForSequenceClassification.from_pretrained(onnx_model_dir)
tokenizer = AutoTokenizer.from_pretrained(onnx_model_dir)

pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device="cuda") # type: ignore

text_for_inference = "This movie is absolutely fantastic!"

t = time()
result = pipe(text_for_inference)
print(f"Inference result for '{text_for_inference}': {result}")
print(f"Time taken: {time() - t}")
