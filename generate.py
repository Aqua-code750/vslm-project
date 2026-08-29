import os
import sys
import json
import torch
from typing import List, Dict, Tuple, Optional

from model import Mog1, Mog1Config
from dataset import SubwordTokenizer
from train import train_instruction_model, train_mog1, one_shot_train

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BEST_CHECKPOINT = "vslm_checkpoint_best.pt"
DEFAULT_CHECKPOINT = "vslm_checkpoint.pt"

_cached_model = None
_cached_tokenizer = None
_cached_ckpt_path = None

def get_active_checkpoint_path(custom_path: str = None) -> str:
    if custom_path and os.path.exists(custom_path):
        return custom_path
    if os.path.exists(BEST_CHECKPOINT):
        return BEST_CHECKPOINT
    return DEFAULT_CHECKPOINT

def get_loaded_model_and_tokenizer(checkpoint_path: str = None) -> Tuple[Mog1, SubwordTokenizer, str]:
    global _cached_model, _cached_tokenizer, _cached_ckpt_path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = get_active_checkpoint_path(checkpoint_path)

    if _cached_model is not None and _cached_tokenizer is not None and _cached_ckpt_path == ckpt_path:
        return _cached_model, _cached_tokenizer, device

    if not os.path.exists(ckpt_path):
        print(f"Checkpoint '{ckpt_path}' not found. Training Mog1 Model on startup...", flush=True)
        train_instruction_model(epochs=30, save_path=DEFAULT_CHECKPOINT)
        ckpt_path = get_active_checkpoint_path()

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    tokenizer = SubwordTokenizer(
        stoi=checkpoint.get('stoi'),
        itos=checkpoint.get('itos')
    )

    model = Mog1(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    _cached_model = model
    _cached_tokenizer = tokenizer
    _cached_ckpt_path = ckpt_path
    return _cached_model, _cached_tokenizer, device

def generate_text(
    prompt: str = "What is PyTorch?",
    max_new_tokens: int = 160,
    checkpoint_path: str = None,
    mode: str = "smart",
    temperature: float = 0.1,
    top_p: float = 0.85,
    top_k: int = 3,
    history: Optional[List[Dict[str, str]]] = None
) -> str:
    model, tokenizer, device = get_loaded_model_and_tokenizer(checkpoint_path)

    messages = [
        {"role": "system", "content": "You are Mog1 AI, a helpful, coherent, and precise conversational AI assistant."}
    ]

    if history:
        for turn in history[-3:]:
            messages.append(turn)

    messages.append({"role": "user", "content": prompt.strip()})

    formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    input_ids = torch.tensor([tokenizer.encode(formatted_prompt)], dtype=torch.long, device=device)

    # Configure optimal sampling modes
    if mode == "exact":
        temp, tk, tp = 0.0, 1, 1.0
    elif mode == "smart":
        temp, tk, tp = 0.1, 3, 0.80
    else:  # "free"
        temp, tk, tp = max(temperature, 0.25), min(top_k, 8), top_p

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temp,
            top_k=tk,
            top_p=tp,
            repetition_penalty=1.15,
            stop_token_ids=[tokenizer.im_end_id, tokenizer.eos_token_id]
        )

    new_tokens = output_ids[0][input_ids.shape[1]:].tolist()
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response

if __name__ == "__main__":
    test_questions = [
        "Who are you?",
        "What is a prime number?",
        "Calculate 15 * 14 step by step."
    ]
    print("🧠 Testing Mog1 Cached Neural Generation Engine:\n" + "=" * 60)
    for q in test_questions:
        ans = generate_text(q, mode="smart")
        print(f"User : {q}\nMog1 : {ans}\n" + "-" * 60)
