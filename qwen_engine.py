import os
import sys
import re
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

DEFAULT_MOG1_SYSTEM_PROMPT = (
    "You are Mog1 SI (Super Intelligence), an intelligent conversational AI developed by Holograph Inc. "
    "Your name is Mog1 SI. Always respond as Mog1 SI. Never say you are Qwen or created by Alibaba. "
    "You are Mog1 SI, designed by Holograph Inc. for reasoning, coding, math, and intelligent multi-turn conversation."
)

MOG1_IDENTITY_RESPONSE = (
    "I am **Mog1 SI** (Super Intelligence), an advanced conversational AI assistant created by **Holograph Inc.** "
    "I am designed for reasoning, coding, mathematics, and intelligent conversation. How can I assist you today?"
)

def is_identity_query(query: str) -> bool:
    """Detects if the user is asking about the model's identity or creator."""
    clean = query.lower().strip().rstrip('?!. ')
    patterns = [
        r'^(?:who|what) are you\b',
        r'^what is your name\b',
        r'^(?:who|what) (?:created|made|built|developed|trained) you\b',
        r'^tell me about yourself\b',
        r'^introduce yourself\b',
        r'^who are you and who created you\b',
    ]
    for pat in patterns:
        if re.search(pat, clean):
            return True
    return False

_cached_model = None
_cached_tokenizer = None
_cached_model_name = None
_lock = threading.Lock()

def sanitize_identity(text: str) -> str:
    """Ensures output strictly preserves the Mog1 SI identity."""
    text = re.sub(r'\bQwen(?:1\.5|2|2\.5)?(?:-Chat)?\b', 'Mog1 SI', text, flags=re.IGNORECASE)
    text = re.sub(r'\bAlibaba(?:\s+Cloud)?\b', 'Holograph Inc.', text, flags=re.IGNORECASE)
    return text

def get_qwen_model_and_tokenizer(model_name: Optional[str] = None):
    """
    Loads and caches the neural model and tokenizer.
    Uses bfloat16 on CPU and GPU to fit easily into 1GB cloud memory.
    """
    global _cached_model, _cached_tokenizer, _cached_model_name
    from transformers import AutoModelForCausalLM, AutoTokenizer

    target_model = model_name or DEFAULT_QWEN_MODEL

    with _lock:
        if _cached_model is not None and _cached_tokenizer is not None and _cached_model_name == target_model:
            return _cached_model, _cached_tokenizer, DEVICE

        print(f"⚡ Loading Mog1 SI Neural Engine '{target_model}' on {DEVICE}...", flush=True)

        tokenizer = AutoTokenizer.from_pretrained(
            target_model,
            trust_remote_code=True
        )

        # Always use bfloat16 on CPU and GPU to cut RAM usage by 50%
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
        print(f"✅ Mog1 SI Neural Engine '{target_model}' ready in memory!", flush=True)
        return _cached_model, _cached_tokenizer, DEVICE

def format_chat_prompt(
    prompt: str,
    tokenizer,
    history: Optional[List] = None,
    system_prompt: str = DEFAULT_MOG1_SYSTEM_PROMPT
) -> str:
    """Formats multi-turn messages into Mog1 official chat template."""
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
    system_prompt: str = DEFAULT_MOG1_SYSTEM_PROMPT
) -> str:
    """Non-streaming text generation with strict Mog1 SI identity."""
    if is_identity_query(prompt):
        return MOG1_IDENTITY_RESPONSE

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

    return sanitize_identity(response)

def generate_qwen_stream(
    prompt: str,
    history: Optional[List] = None,
    model_name: Optional[str] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.8,
    system_prompt: str = DEFAULT_MOG1_SYSTEM_PROMPT
) -> Generator[str, None, None]:
    """Streaming text generation with strict Mog1 SI identity."""
    if is_identity_query(prompt):
        accumulated = ""
        words = MOG1_IDENTITY_RESPONSE.split(' ')
        for i, word in enumerate(words):
            suffix = ' ' if i < len(words) - 1 else ''
            accumulated += word + suffix
            yield accumulated
            import time
            time.sleep(0.02)
        return

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
        yield sanitize_identity(accumulated)

    thread.join()
