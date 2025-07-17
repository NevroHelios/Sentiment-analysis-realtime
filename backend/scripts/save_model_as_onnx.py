import os
from pathlib import Path
from optimum.onnxruntime import ORTModelForSequenceClassification

parent_dir = Path(os.getcwd()).parent
model_dir = os.path.join(parent_dir, 'models', 'finetuned')
onnx_model_dir = os.path.join(parent_dir, 'models', 'onnx')


if not os.path.exists(model_dir):
    print(f"Model directory {model_dir} does not exist. Using the default model directory.")
    model_dir = model_dir


def convert_and_save(model_dir: str | Path = model_dir, output_dir: str | Path = onnx_model_dir):
    model = ORTModelForSequenceClassification.from_pretrained(model_dir, export=True)
    # tokenizer = AutoTokenizer.from_pretrained(model_dir) # tokenizer remains the same, no need to export it

    try:
        model.save_pretrained(output_dir)
        print(f"Model and tokenizer saved to {output_dir}")

    except Exception as e:
        print(f"An error occurred while saving the model and tokenizer: {e}")

if __name__ == "__main__":
    convert_and_save()