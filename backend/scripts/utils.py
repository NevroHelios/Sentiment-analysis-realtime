import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

def train(
        model: torch.nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        schedular,
        epochs: int,
        clip_grad_norm: float,
        device: str | None,
        num_training_steps: int,
        checkpoint_dir: str | Path,
):
    if not device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metrices = {
        "loss": [],
        "accuracy": [],
    }
    best_loss = float('inf')

    api_key = os.getenv('WANDB_API_KEY')
    assert api_key is not None
    wandb.login(key=api_key)
    wandb.init(project="sent-clf finetuning", name="fine run")
    wandb.watch(model, log="all")

    progress_bar = tqdm(range(num_training_steps), desc="Training Progress")
    model.to(device)
    for epoch in range(epochs):
        model.train()
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            loss = outputs.loss
            loss.backward()
            
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            
            optimizer.step()
            schedular.step()
            optimizer.zero_grad()

            metrices["loss"].append(loss.item())
            metrices["accuracy"].append((outputs.logits.argmax(dim=-1) == batch['labels']).float().mean().item())
            wandb.log({
                'Loss': loss.item(),
                'Accuracy': (outputs.logits.argmax(dim=-1) == batch['labels']).float().mean().item(),
                'Step': progress_bar.n,
            })

            if loss.item() < best_loss:
                best_loss = loss.item()
                model.save_pretrained(checkpoint_dir) # type: ignore

            progress_bar.update(1)

    wandb.finish()
    return metrices
