import os
from pathlib import Path
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

parent_dir = Path(os.getcwd()).parent
data_dir = os.path.join(parent_dir, "data")
model_dir = os.path.join(parent_dir, "models")
log_dir = os.path.join(parent_dir, "models", "logs")
output_dir = os.path.join(parent_dir, "models", "finetuned")

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, tokenizer) -> None:
        super().__init__()
        self.data = []
        self.tokenizer = tokenizer
        with open(data_path, 'r') as file:
            for line in file:
                item = json.loads(line)
                self.data.append({
                    'input_ids': torch.tensor(self.tokenizer(item['text'], truncation=True, padding="max_length", max_length=128)['input_ids']),
                    'attention_mask': torch.tensor(self.tokenizer(item['text'], truncation=True, padding="max_length", max_length=128)['attention_mask']),
                    'label': torch.tensor(int(item['label']))
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model.")
    parser.add_argument("--data", type=str, default="data.jsonl", help="Data file name. Put the data file inside the `data` directory or `the absolute path`.")
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

    dataset = CustomDataset(data_path, tokenizer)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir=log_dir,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

if __name__ == "__main__":
    main()
