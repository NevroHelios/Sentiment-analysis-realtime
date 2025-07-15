import os
from pathlib import Path
import argparse
from optimum.onnxruntime import ORTModelForSequenceClassification

parent_dir = Path(os.getcwd()).parent
model_dir = os.path.join(parent_dir, 'models', 'finetuned')
onnx_model_dir = os.path.join(parent_dir, 'models', 'onnx')

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default=model_dir)
parser.add_argument("--onnx_model_dir", type=str, default=onnx_model_dir)
args = parser.parse_args()

if not os.path.exists(args.model_dir):
    print(f"Model directory {args.model_dir} does not exist. Using the default model directory.")
    args.model_dir = model_dir


def convert_and_save(model_dir: str | Path = args.model_dir, output_dir: str | Path = args.onnx_model_dir):
    model = ORTModelForSequenceClassification.from_pretrained(model_dir, export=True)
    # tokenizer = AutoTokenizer.from_pretrained(model_dir) # tokenizer remains the same, no need to export it

    try:
        model.save_pretrained(output_dir)
        print(f"Model and tokenizer saved to {output_dir}")

    except Exception as e:
        print(f"An error occurred while saving the model and tokenizer: {e}")

if __name__ == "__main__":
    convert_and_save()