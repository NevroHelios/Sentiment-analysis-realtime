from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
from time import time
from transformers.pipelines import pipeline
from transformers.pipelines.base import Pipeline
from typing import Dict, Any, List
import os

def load_pipe():
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    onnx_model_dir = os.path.join(models_dir, "onnx")
    assert os.path.exists(onnx_model_dir), "ONNX model directory does not exist."

    model = ORTModelForSequenceClassification.from_pretrained(onnx_model_dir)
    try:
        tokenizer = AutoTokenizer.from_pretrained(onnx_model_dir)
        try:
            tokenizer = AutoTokenizer.from_pretrained(models_dir)
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer from {onnx_model_dir} or {models_dir}.") from e
    except:
        pass

    pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device="cpu") # type: ignore

    return pipe

def get_sentiment(text_for_inference: str, pipe: Pipeline) -> Dict[str, Any]:
    t = time()
    result = pipe(text_for_inference)
    time_taken = time() - t
    try:
        assert isinstance(result, List) and len(result) == 1
        result[0]['time_taken'] = round(time_taken * 1000, 2)
        return result[0]
    except Exception as e:
        return {"error": e}

if __name__ == "__main__":
    result = get_sentiment("hello world", load_pipe())
    # print(f"Time taken: {result.get('time_taken', 0)}") # Time taken: 0.007698
    print(f"Result: {result}") # Result: {'label': 'NEGATIVE', 'score': 0.9998, 'time_taken': 0.007698}
