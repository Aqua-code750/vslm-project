import os
import torch
from model import Mog1
from dataset import SubwordTokenizer
from train import train_mog1

def generate_text(
    prompt: str = "What is PyTorch?",
    max_new_tokens: int = 40,
    checkpoint_path: str = "vslm_checkpoint.pt",
    mode: str = "smart"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint '{checkpoint_path}' not found. Pretraining Mog1 AI Model internally...", flush=True)
        train_mog1(epochs=60, save_path=checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    tokenizer = SubwordTokenizer(
        stoi=checkpoint.get('stoi'),
        itos=checkpoint.get('itos'),
        use_tiktoken=checkpoint.get('use_tiktoken', False)
    )

    model = Mog1(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    formatted_prompt = f"User: {prompt}\nMog1:"
    context_tokens = tokenizer.encode(formatted_prompt)
    context = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(0)

    temperature = 0.2 if mode == "exact" else 0.5
    top_k = 3 if mode == "exact" else 15
    top_p = 0.85 if mode == "exact" else 0.95

    out_tokens = model.generate(
        context,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=1.25
    )

    # Extract ONLY the newly generated tokens
    new_token_ids = out_tokens[0][len(context_tokens):].tolist()
    raw_response = tokenizer.decode(new_token_ids)

    # Clean response
    clean_response = raw_response.split("User:")[0].split("Mog1:")[0].strip()
    if not clean_response:
        clean_response = raw_response.strip()

    return clean_response

if __name__ == "__main__":
    test_prompts = [
        "What is PyTorch?",
        "What is an AI model?",
        "Explain Transformer architecture.",
        "What is tokenization?",
        "Hello, who are you?"
    ]
    for p in test_prompts:
        resp = generate_text(p, mode="exact")
        print(f"\nQ: {p}\nA: {resp}\n" + "-"*40)
