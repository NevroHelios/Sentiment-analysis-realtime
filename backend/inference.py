from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
from time import time
from transformers.pipelines import pipeline
from pathlib import Path
from typing import Dict, Any, List

onnx_model_dir = Path("./models/onnx/")
assert onnx_model_dir.exists(), "ONNX model directory does not exist."

model = ORTModelForSequenceClassification.from_pretrained(onnx_model_dir)
tokenizer = AutoTokenizer.from_pretrained(onnx_model_dir)

pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device="cpu") # type: ignore

def get_gentiment(text_for_inference: str) -> Dict[str, Any]:
    t = time()
    result = pipe(text_for_inference)
    time_taken = time() - t
    try:
        assert isinstance(result, List) and len(result) == 1
        result[0]['time_taken'] = time_taken
        return result[0]
    except Exception as e:
        return {"error": e}

if __name__ == "__main__":
    result = get_gentiment("hello world")
    # print(f"Time taken: {result.get('time_taken', 0)}") # Time taken: 0.007698
    print(f"Result: {result}") # Result: {'label': 'NEGATIVE', 'score': 0.9998, 'time_taken': 0.007698}