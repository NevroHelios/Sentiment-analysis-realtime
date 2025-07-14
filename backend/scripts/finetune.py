import os
from pathlib import Path
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

data_dir = os.path.join(Path(os.getcwd()).parent, "data")


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model.")
    parser.add_argument("--data", type=str, default="data.jsonl", help="Data file name. Put the data file inside the `data` directory or `the absolute path`.")
    args = parser.parse_args()
    # datafile
    data_file = os.path.join(data_dir, args.data)
    if not os.path.exists(data_file):
        if os.path.exists(args.data): # check if the absolute path is provided
            data_file = args.data
        else:
            raise FileNotFoundError(f"Data file {data_file} does not exist. Please provide a valid path.")
    
    
    with open(data_file, 'r') as file: # read the data
        data = [json.loads(line) for line in file]

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)

    # tokenize the data
    tokenized_data = []
    for item in data:
        tmp = {
            key: torch.tensor(val) for key, val in tokenizer(item['text'], truncation=True, padding="max_length", max_length=128).items()
        }
        tmp['label'] = torch.tensor(int(item['label']))
        tokenized_data.append(tmp)
    
    dataset = CustomDataset(tokenized_data)
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

if __name__ == "__main__":
    main()
