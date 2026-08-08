import os
import sys
import re
import urllib.request
import json
import torch
from model import Mog1
from dataset import SubwordTokenizer
from train import train_mog1

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def fetch_online_knowledge(query: str) -> str:
    try:
        clean_q = re.sub(r'[^\w\s]', '', query).strip()
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(clean_q)}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mog1AI/1.0'})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
            if 'extract' in data and data['extract']:
                sentences = data['extract'].split('. ')
                return sentences[0] + '.' if sentences else data['extract']
    except Exception:
        pass
    return ""

def generate_text(
    prompt: str = "What is PyTorch?",
    max_new_tokens: int = 50,
    checkpoint_path: str = "vslm_checkpoint.pt",
    mode: str = "free",
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 30
) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint '{checkpoint_path}' not found. Pretraining Mog1 AI Model internally...", flush=True)
        train_mog1(epochs=30, save_path=checkpoint_path)

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

    formatted_prompt = f"User: {prompt.strip()}\nMog1:"
    context_tokens = tokenizer.encode(formatted_prompt)
    context = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(0)

    if mode == "exact":
        temp, tk, tp = 0.2, 3, 0.85
    elif mode == "smart":
        temp, tk, tp = 0.5, 15, 0.95
    else: # "free"
        temp, tk, tp = temperature, top_k, top_p

    out_tokens = model.generate(
        context,
        max_new_tokens=max_new_tokens,
        temperature=temp,
        top_k=tk,
        top_p=tp,
        repetition_penalty=1.25
    )

    new_token_ids = out_tokens[0][len(context_tokens):].tolist()
    raw_response = tokenizer.decode(new_token_ids)
    clean_response = raw_response.split("User:")[0].split("Mog1:")[0].strip()

    if not clean_response or len(clean_response) < 3:
        fact = fetch_online_knowledge(prompt)
        if fact:
            clean_response = fact
        elif not clean_response:
            clean_response = f"{prompt.strip()} is an interesting topic in technology and AI."

    return clean_response

if __name__ == "__main__":
    test_prompts = [
        "Write a creative story about space exploration.",
        "What is PyTorch?",
        "What is 5 times 5?",
        "Hello, tell me something interesting!"
    ]
    for p in test_prompts:
        resp = generate_text(p, mode="free")
        print(f"\nUser: {p}\nMog1: {resp}\n" + "-"*50)
