import os
import sys
import threading
from typing import List, Dict, Optional, Generator, Tuple
import torch

# Ensure UTF-8 output on Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Available models
SUPPORTED_MODELS = [
    "Qwen/Qwen1.5-0.5B-Chat",
    "Qwen/Qwen1.5-1.8B-Chat",
    "Qwen/Qwen1.5-7B-Chat"
]

DEFAULT_QWEN_MODEL = os.environ.get("QWEN_MODEL", "Qwen/Qwen1.5-0.5B-Chat")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_cached_model = None
_cached_tokenizer = None
_cached_model_name = None
_lock = threading.Lock()

def get_qwen_model_and_tokenizer(model_name: Optional[str] = None):
    """
    Loads and caches the Qwen model and tokenizer.
    Uses bfloat16 on CUDA or float32 on CPU for optimal speed and memory efficiency.
    """
    global _cached_model, _cached_tokenizer, _cached_model_name
    from transformers import AutoModelForCausalLM, AutoTokenizer

    target_model = model_name or DEFAULT_QWEN_MODEL

    with _lock:
        if _cached_model is not None and _cached_tokenizer is not None and _cached_model_name == target_model:
            return _cached_model, _cached_tokenizer, DEVICE

        print(f"⚡ Loading Qwen conversational model '{target_model}' on {DEVICE}...", flush=True)

        tokenizer = AutoTokenizer.from_pretrained(
            target_model,
            trust_remote_code=True
        )

        # Always use bfloat16 on CPU and GPU to cut RAM usage by 50% (prevents Streamlit Cloud 1GB OOM crash)
        model_dtype = torch.bfloat16

        model = AutoModelForCausalLM.from_pretrained(
            target_model,
            dtype=model_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(DEVICE)
        model.eval()

        _cached_model = model
        _cached_tokenizer = tokenizer
        _cached_model_name = target_model
        print(f"✅ Qwen model '{target_model}' successfully loaded into memory!", flush=True)
        return _cached_model, _cached_tokenizer, DEVICE

def format_chat_prompt(
    prompt: str,
    tokenizer,
    history: Optional[List] = None,
    system_prompt: str = "You are Mog1 AI (powered by Qwen 1.5), a helpful, highly knowledgeable, and friendly conversational assistant."
) -> str:
    """Formats multi-turn messages into Qwen official chat template."""
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        for turn in history[-6:]:
            if isinstance(turn, dict) and "role" in turn and "content" in turn:
                messages.append({"role": turn["role"], "content": turn["content"]})
            elif isinstance(turn, (list, tuple)) and len(turn) == 2:
                u, b = turn
                if u:
                    messages.append({"role": "user", "content": str(u)})
                if b:
                    messages.append({"role": "assistant", "content": str(b)})

    messages.append({"role": "user", "content": prompt.strip()})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def generate_qwen_text(
    prompt: str,
    history: Optional[List] = None,
    model_name: Optional[str] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.8,
    system_prompt: str = "You are Mog1 AI (powered by Qwen 1.5), a helpful, highly knowledgeable, and friendly conversational assistant."
) -> str:
    """Non-streaming text generation for CLI and automated scripts."""
    model, tokenizer, device = get_qwen_model_and_tokenizer(model_name)
    chat_text = format_chat_prompt(prompt, tokenizer, history=history, system_prompt=system_prompt)
    model_inputs = tokenizer([chat_text], return_tensors="pt").to(device)

    do_sample = temperature > 0.05
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.no_grad():
        generated_ids = model.generate(model_inputs.input_ids, **gen_kwargs)
        input_len = model_inputs.input_ids.shape[1]
        response_ids = generated_ids[0][input_len:]
        response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

    return response

def generate_qwen_stream(
    prompt: str,
    history: Optional[List] = None,
    model_name: Optional[str] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.8,
    system_prompt: str = "You are Mog1 AI (powered by Qwen 1.5), a helpful, highly knowledgeable, and friendly conversational assistant."
) -> Generator[str, None, None]:
    """Streaming text generation for web interfaces (Gradio / Streamlit)."""
    from transformers import TextIteratorStreamer

    model, tokenizer, device = get_qwen_model_and_tokenizer(model_name)
    chat_text = format_chat_prompt(prompt, tokenizer, history=history, system_prompt=system_prompt)
    model_inputs = tokenizer([chat_text], return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    do_sample = temperature > 0.05
    gen_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    accumulated = ""
    for new_text in streamer:
        accumulated += new_text
        yield accumulated

    thread.join()
