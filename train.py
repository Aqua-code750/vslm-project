import os
import sys
import math
import json
import time
from typing import Tuple, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import Mog1, Mog1Config
from dataset import SubwordTokenizer, InstructionSFTDataset, PretrainTextDataset
from data_builder import INSTRUCTION_DATA, generate_datasets

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def build_tokenizer_from_instructions(instruction_data: List[Dict[str, str]], vocab_limit: int = 2048) -> SubwordTokenizer:
    full_text = ""
    for item in instruction_data:
        full_text += f"{item.get('instruction', '')}\n{item.get('response', '')}\n"
    tokenizer = SubwordTokenizer(text_corpus=full_text, vocab_limit=vocab_limit)
    return tokenizer


def configure_optimizers(model: Mog1, lr: float = 1e-3, weight_decay: float = 0.1, betas: Tuple[float, float] = (0.9, 0.95)) -> torch.optim.Optimizer:
    """
    Separates parameters into decay (2D weight matrices) and no-decay (1D norms and biases).
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas, eps=1e-8)
    return optimizer


def get_lr_schedule(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Linear warmup followed by Cosine Annealing decay."""
    if step < warmup_steps:
        return max_lr * (step + 1) / (warmup_steps + 1)
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


@torch.no_grad()
def evaluate_loss(model: Mog1, dataloader: DataLoader, device: str) -> Tuple[float, float]:
    """Computes cross entropy loss and perplexity on evaluation dataloader."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        _, loss, _ = model(x, targets=y)
        # Count non-masked tokens (target != -100)
        non_masked = (y != -100).sum().item()
        if non_masked > 0:
            total_loss += loss.item() * non_masked
            total_tokens += non_masked

    avg_loss = total_loss / max(1, total_tokens) if total_tokens > 0 else float('inf')
    perplexity = math.exp(min(avg_loss, 20.0))  # Prevent numerical overflow
    return avg_loss, perplexity


def train_instruction_model(
    train_data_path: str = "data/train_instructions.json",
    val_data_path: str = "data/val_instructions.json",
    epochs: int = 50,
    batch_size: int = 8,
    max_lr: float = 1.2e-3,
    min_lr: float = 1e-4,
    block_size: int = 256,
    save_path: str = "vslm_checkpoint.pt"
) -> Dict[str, Any]:
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Starting Mog1 Instruction Model Training on {device}...", flush=True)

    # Ensure dataset files exist
    if not os.path.exists(train_data_path) or not os.path.exists(val_data_path):
        train_data_path, val_data_path = generate_datasets()

    with open(train_data_path, "r", encoding="utf-8") as f:
        train_samples = json.load(f)
    with open(val_data_path, "r", encoding="utf-8") as f:
        val_samples = json.load(f)

    # Build tokenizer on combined corpus
    all_samples = train_samples + val_samples
    tokenizer = build_tokenizer_from_instructions(all_samples, vocab_limit=2048)

    # Create SFT datasets with target loss masking
    train_dataset = InstructionSFTDataset(train_samples, tokenizer, block_size=block_size)
    val_dataset = InstructionSFTDataset(val_samples, tokenizer, block_size=block_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    config = Mog1Config(
        vocab_size=tokenizer.vocab_size,
        n_embd=288,
        n_head=6,
        n_layer=6,
        d_ffn=768,
        block_size=block_size,
        dropout=0.08,
        tie_weights=True
    )

    model = Mog1(config).to(device)
    optimizer = configure_optimizers(model, lr=max_lr, weight_decay=0.15)

    total_steps = len(train_loader) * epochs
    warmup_steps = max(5, int(total_steps * 0.1))

    print(f"Model Parameters: {model.get_num_params():,} | Vocab Size: {tokenizer.vocab_size} | Total Steps: {total_steps}", flush=True)

    best_val_loss = float('inf')
    best_checkpoint = None

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        steps_in_epoch = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # Update learning rate with warmup + cosine decay
            lr = get_lr_schedule(global_step, warmup_steps, total_steps, max_lr, min_lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.zero_grad()
            _, loss, _ = model(x, targets=y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            steps_in_epoch += 1
            global_step += 1

        avg_train_loss = epoch_loss / max(1, steps_in_epoch)

        # Evaluate on validation set
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            val_loss, val_ppl = evaluate_loss(model, val_loader, device)
            print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} | LR: {lr:.6f}", flush=True)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "stoi": tokenizer.stoi,
                    "itos": tokenizer.itos,
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                    "epoch": epoch,
                    "is_best": True,
                    "timestamp": time.time()
                }
                torch.save(best_checkpoint, "vslm_checkpoint_best.pt")
                torch.save(best_checkpoint, save_path)
                print(f"  ⭐ Saved new BEST checkpoint (Val Loss: {best_val_loss:.4f}, PPL: {val_ppl:.2f}) -> 'vslm_checkpoint_best.pt'", flush=True)

        # Save latest checkpoint
        latest_checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": config,
            "stoi": tokenizer.stoi,
            "itos": tokenizer.itos,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_ppl": val_ppl,
            "epoch": epoch,
            "is_best": False,
            "timestamp": time.time()
        }
        torch.save(latest_checkpoint, "vslm_checkpoint_latest.pt")

    elapsed = time.time() - start_time
    print(f"\n✅ Training Complete in {elapsed:.2f}s! Best Validation Loss: {best_val_loss:.4f}", flush=True)
    return {
        "best_val_loss": best_val_loss,
        "elapsed_time": elapsed,
        "epochs": epochs,
        "save_path": save_path
    }


def one_shot_train(save_path: str = "vslm_checkpoint.pt"):
    """Quick high-speed training run for instant pretraining."""
    return train_instruction_model(epochs=35, batch_size=8, save_path=save_path)


if __name__ == "__main__":
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 45
    train_instruction_model(epochs=epochs)
