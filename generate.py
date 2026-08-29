import os
import sys
import json
import torch

from model import Mog1, Mog1Config
from dataset import SubwordTokenizer
from train import train_instruction_model, train_mog1, one_shot_train

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BEST_CHECKPOINT = "vslm_checkpoint_best.pt"
DEFAULT_CHECKPOINT = "vslm_checkpoint.pt"

def get_active_checkpoint_path(custom_path: str = None) -> str:
    if custom_path and os.path.exists(custom_path):
        return custom_path
    if os.path.exists(BEST_CHECKPOINT):
        return BEST_CHECKPOINT
    return DEFAULT_CHECKPOINT

def generate_text(
    prompt: str = "What is PyTorch?",
    max_new_tokens: int = 60,
    checkpoint_path: str = None,
    mode: str = "smart",
    temperature: float = 0.15,
    top_p: float = 0.85,
    top_k: int = 5
) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = get_active_checkpoint_path(checkpoint_path)

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

    messages = [
        {"role": "system", "content": "You are Mog1 AI, a helpful, coherent, and precise conversational AI assistant."},
        {"role": "user", "content": prompt.strip()}
    ]
    formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    input_ids = torch.tensor([tokenizer.encode(formatted_prompt)], dtype=torch.long, device=device)

    # Configure optimal sampling modes for small language models
    if mode == "exact":
        temp, tk, tp = 0.0, 1, 1.0
    elif mode == "smart":
        temp, tk, tp = 0.1, 3, 0.80
    else:  # "free"
        temp, tk, tp = max(temperature, 0.2), min(top_k, 10), top_p

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
        "What is PyTorch?",
        "Calculate 15 * 14 step by step.",
        "What is Rotary Position Embedding (RoPE)?"
    ]
    print("🧠 Testing Mog1 Neural Generation Engine:\n" + "=" * 60)
    for q in test_questions:
        ans = generate_text(q, mode="smart")
        print(f"User : {q}\nMog1 : {ans}\n" + "-" * 60)
