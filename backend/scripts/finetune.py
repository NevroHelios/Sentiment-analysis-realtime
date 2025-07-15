import os
from pathlib import Path
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from transformers.trainer import Trainer
# from transformers.training_args import TrainingArguments
import numpy as np
from torch.optim import AdamW
from transformers.optimization import get_scheduler
from torch.utils.data import DataLoader

from utils import train

# set random seed
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

parent_dir = Path(os.getcwd()).parent
data_dir = os.path.join(parent_dir, "data")
model_dir = os.path.join(parent_dir, "models")
log_dir = os.path.join(parent_dir, "models", "logs")
output_dir = os.path.join(parent_dir, "models", "finetuned")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, tokenizer) -> None:
        super().__init__()
        self.data = []
        self.tokenizer = tokenizer
        
        self._load_data(data_path)

    def _load_data(self, data_path: str):
        with open(data_path, 'r') as file:
            for line in file:
                item = json.loads(line)
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokenized_inputs = {
            key: torch.tensor(val) for key, val in self.tokenizer(self.data[idx]['text'], truncation=True, padding="max_length", max_length=128).items()
        } 
        tokenized_inputs['labels'] = torch.tensor(int(self.data[idx]['label']), dtype=torch.long)
        return tokenized_inputs


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model.")
    parser.add_argument("--data", type=str, default="data.jsonl", help="Data file name. Put the data file inside the `data` directory or `the absolute path`.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--model_dir", type=str, default=model_dir, help="Directory where the model is stored.")
    parser.add_argument("--output_dir", type=str, default=output_dir, help="Directory to save the fine-tuned model.")
    parser.add_argument("--log_dir", type=str, default=log_dir, help="Directory to save training logs.")
    parser.add_argument("--learning_rate", '--lr', type=float, default=5e-5, help="Learning rate for training.")
    parser.add_argument("--device", type=str, default=device, help="Device to use for training (e.g., 'cuda' or 'cpu').")

    args = parser.parse_args()
    # datafile
    data_path = os.path.join(data_dir, args.data)
    if not os.path.exists(data_path):
        if os.path.exists(args.data): # check if the absolute path is provided
            data_path = args.data
        else:
            raise FileNotFoundError(f"Data file {data_path} does not exist. Please provide a valid path.")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2)  

    train_dataset = CustomDataset(data_path, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = args.epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    results = train(
        num_training_steps=num_training_steps,
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        schedular=lr_scheduler,
        epochs=args.epochs,
        clip_grad_norm=1.0,
        checkpoint_dir=args.output_dir,
        device=device
    )

    print(f"Training completed. Results: {results}")

if __name__ == "__main__":
    main()
