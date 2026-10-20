import os
import sys
import json
import torch
from typing import List, Dict, Tuple, Optional, Generator

from qwen_engine import (
    generate_qwen_text,
    generate_qwen_stream,
    get_qwen_model_and_tokenizer,
    DEFAULT_QWEN_MODEL
)
from model import Mog1, Mog1Config
from dataset import SubwordTokenizer

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BEST_CHECKPOINT = "vslm_checkpoint_best.pt"
DEFAULT_CHECKPOINT = "vslm_checkpoint.pt"

# Environment flag to switch between Qwen conversational weights and custom Mog1 weights
# Default is Qwen 1.5 Chat weights
USE_QWEN = os.environ.get("USE_QWEN", "1") == "1"

_cached_mog1_model = None
_cached_mog1_tokenizer = None
_cached_ckpt_path = None

def get_active_checkpoint_path(custom_path: str = None) -> str:
    if custom_path and os.path.exists(custom_path):
        return custom_path
    if os.path.exists(BEST_CHECKPOINT):
        return BEST_CHECKPOINT
    return DEFAULT_CHECKPOINT

def get_loaded_mog1_model_and_tokenizer(checkpoint_path: str = None) -> Tuple[Mog1, SubwordTokenizer, str]:
    """Loads scratch Mog1 checkpoint weights."""
    global _cached_mog1_model, _cached_mog1_tokenizer, _cached_ckpt_path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = get_active_checkpoint_path(checkpoint_path)

    if _cached_mog1_model is not None and _cached_mog1_tokenizer is not None and _cached_ckpt_path == ckpt_path:
        return _cached_mog1_model, _cached_mog1_tokenizer, device

    if not os.path.exists(ckpt_path):
        from train import train_instruction_model
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

    _cached_mog1_model = model
    _cached_mog1_tokenizer = tokenizer
    _cached_ckpt_path = ckpt_path
    return _cached_mog1_model, _cached_mog1_tokenizer, device

def get_loaded_model_and_tokenizer(checkpoint_path: str = None):
    """
    Returns active model and tokenizer.
    If Qwen is active (default), returns Qwen model and tokenizer.
    Otherwise returns Mog1 custom model.
    """
    if USE_QWEN and checkpoint_path is None:
        return get_qwen_model_and_tokenizer()
    return get_loaded_mog1_model_and_tokenizer(checkpoint_path)

def generate_text(
    prompt: str = "What is PyTorch?",
    max_new_tokens: int = 256,
    checkpoint_path: str = None,
    mode: str = "smart",
    temperature: float = 0.7,
    top_p: float = 0.85,
    top_k: int = 3,
    history: Optional[List[Dict[str, str]]] = None,
    model_name: Optional[str] = None
) -> str:
    """
    Generates conversational response using Qwen 1.5 chat weights by default,
    or falls back to custom Mog1 checkpoint if explicitly requested.
    """
    # If custom checkpoint path is provided or USE_QWEN is disabled, use Mog1
    if checkpoint_path is not None or not USE_QWEN:
        model, tokenizer, device = get_loaded_mog1_model_and_tokenizer(checkpoint_path)
        messages = [
            {"role": "system", "content": "You are Mog1 SI (Super Intelligence), a helpful and coherent conversational assistant created by Holograph Inc. Holograph Inc. is the user's (Aqua's) company and your creator."}
        ]
        if history:
            for turn in history[-3:]:
                messages.append(turn)
        messages.append({"role": "user", "content": prompt.strip()})

        formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        input_ids = torch.tensor([tokenizer.encode(formatted_prompt)], dtype=torch.long, device=device)

        if mode == "exact":
            temp, tk, tp, rep_pen = 0.0, 1, 1.0, 1.0
        elif mode == "smart":
            temp, tk, tp, rep_pen = 0.0, 1, 1.0, 1.0
        else:
            temp, tk, tp, rep_pen = max(temperature, 0.3), min(top_k, 5), top_p, 1.1

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temp,
                top_k=tk,
                top_p=tp,
                repetition_penalty=rep_pen,
                stop_token_ids=[tokenizer.im_end_id, tokenizer.eos_token_id]
            )

        new_tokens = output_ids[0][input_ids.shape[1]:].tolist()
        for stop_id in [tokenizer.im_end_id, tokenizer.eos_token_id]:
            if stop_id in new_tokens:
                new_tokens = new_tokens[:new_tokens.index(stop_id)]

        raw_response = tokenizer.decode(new_tokens, skip_special_tokens=False)
        for marker in ["<|im_end|>", "<|im_start|>", "<|eos|>"]:
            if marker in raw_response:
                raw_response = raw_response.split(marker)[0]
        return raw_response.strip()

    # Conversational Qwen Generation
    if "exact" in mode.lower():
        temp = 0.0
    elif "smart" in mode.lower():
        temp = 0.3
    elif "chat" in mode.lower():
        temp = 0.7
    else:  # "free"
        temp = max(temperature, 0.7)

    return generate_qwen_text(
        prompt=prompt,
        history=history,
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temp,
        top_p=top_p
    )

def generate_stream(
    prompt: str,
    history: Optional[List] = None,
    mode: str = "smart",
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.85,
    top_k: int = 20,
    model_name: Optional[str] = None
) -> Generator[str, None, None]:
    """Yields streaming generated tokens using Qwen conversational weights."""
    if "exact" in mode.lower():
        temp = 0.0
    elif "smart" in mode.lower():
        temp = 0.3
    elif "chat" in mode.lower():
        temp = 0.7
    else:
        temp = max(temperature, 0.7)

    return generate_qwen_stream(
        prompt=prompt,
        history=history,
        model_name=model_name,
        max_new_tokens=max_tokens,
        temperature=temp,
        top_p=top_p
    )

if __name__ == "__main__":
    test_questions = [
        "Hey! Who are you and what can you do?",
        "Explain machine learning in simple terms.",
        "What is 25 * 16?"
    ]
    print("🧠 Testing Qwen Conversational Generation Engine:\n" + "=" * 60)
    for q in test_questions:
        ans = generate_text(q, mode="smart")
        print(f"User : {q}\nAI   : {ans}\n" + "-" * 60)
