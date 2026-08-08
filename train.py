import os
import sys
import torch
from torch.utils.data import DataLoader
from model import Mog1, Mog1Config
from dataset import SubwordTokenizer, TextDataset

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

DEFAULT_KNOWLEDGE_PATH = "knowledge_base.txt"

def train_mog1(
    data_path: str = DEFAULT_KNOWLEDGE_PATH,
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 2.5e-3,
    save_path: str = "mog1_checkpoint.pt",
    target_loss: float = 0.04
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training Mog1 Smart AI Model on device: {device}", flush=True)

    if data_path and os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"Loaded knowledge base from '{data_path}'. Size: {len(text)} characters.", flush=True)
    else:
        text = "User: Hello\nMog1: I am Mog1 AI model."

    if len(text) < 50000:
        text_corpus = text * 5
    else:
        text_corpus = text

    tokenizer = SubwordTokenizer(text_corpus=text_corpus)
    encoded_tokens = tokenizer.encode(text_corpus)
    encoded = torch.tensor(encoded_tokens, dtype=torch.long)

    print(f"Knowledge Base Tokens: {len(encoded)}. Vocab size: {tokenizer.vocab_size}", flush=True)

    block_size = 64
    dataset = TextDataset(encoded, block_size=block_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    config = Mog1Config(
        vocab_size=tokenizer.vocab_size,
        n_embd=256,
        n_head=8,
        n_layer=4,
        block_size=block_size,
        dropout=0.05
    )
    model = Mog1(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Mog1 Smart Architecture initialized with {param_count:,} parameters.", flush=True)
    model.train()

    avg_loss = float('inf')

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        steps = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        scheduler.step()
        avg_loss = total_loss / steps if steps > 0 else 0.0

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs or avg_loss <= target_loss:
            print(f"Epoch {epoch:02d}/{epochs} - Loss: {avg_loss:.4f}", flush=True)

        if avg_loss <= target_loss and epoch >= 20:
            print(f"Target loss <= {target_loss} achieved at epoch {epoch}.", flush=True)
            break

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'use_tiktoken': getattr(tokenizer, 'use_tiktoken', False),
        'stoi': getattr(tokenizer, 'stoi', None),
        'itos': getattr(tokenizer, 'itos', None),
    }
    torch.save(checkpoint, save_path)
    torch.save(checkpoint, "vslm_checkpoint.pt")
    print(f"Mog1 Smart AI Model saved to '{save_path}' & 'vslm_checkpoint.pt' (Final Loss: {avg_loss:.4f}).", flush=True)
    return avg_loss

if __name__ == "__main__":
    epochs = 60
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        epochs = int(sys.argv[1])
    train_mog1(epochs=epochs)
