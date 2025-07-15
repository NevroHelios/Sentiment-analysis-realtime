import os
from pathlib import Path
import argparse
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

parent_dir = Path(os.getcwd()).parent
model_dir = os.path.join(parent_dir, 'models')
onnx_model_dir = os.path.join(model_dir, 'onnx')
os.makedirs(onnx_model_dir, exist_ok=True)



model = ORTModelForSequenceClassification.from_pretrained(model_dir, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

try:
    model.save_pretrained(onnx_model_dir)
    tokenizer.save_pretrained(onnx_model_dir)
    print(f"Model and tokenizer saved to {onnx_model_dir}")

except Exception as e:
    print(f"An error occurred while saving the model and tokenizer: {e}")
