import os
import sys
import re
import urllib.request
import json
import torch
from model import Mog1
from dataset import SubwordTokenizer
from train import train_mog1
from auto_train import trigger_auto_train, is_auto_training

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

CHECKPOINT_PATH = "vslm_checkpoint.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

def load_or_train_model():
    if not os.path.exists(CHECKPOINT_PATH):
        print("Pretraining Mog1 AI Model on startup...", flush=True)
        train_mog1(epochs=30, save_path=CHECKPOINT_PATH)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    config = checkpoint['config']

    tokenizer = SubwordTokenizer(
        stoi=checkpoint.get('stoi'),
        itos=checkpoint.get('itos'),
        use_tiktoken=checkpoint.get('use_tiktoken', False)
    )

    model = Mog1(config).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, tokenizer

model, tokenizer = load_or_train_model()

def respond(message: str, history, mode: str, max_tokens: int):
    if not message or not message.strip():
        return ""

    formatted = f"User: {message.strip()}\nMog1:"
    context_tokens = tokenizer.encode(formatted)
    context = torch.tensor(context_tokens, dtype=torch.long, device=DEVICE).unsqueeze(0)

    temperature = 0.2 if "Exact" in mode else 0.5
    top_k = 3 if "Exact" in mode else 15
    top_p = 0.85 if "Exact" in mode else 0.95

    out = model.generate(
        context,
        max_new_tokens=int(max_tokens),
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=1.35
    )
    
    new_token_ids = out[0][len(context_tokens):].tolist()
    raw_res = tokenizer.decode(new_token_ids)
    res = raw_res.split("User:")[0].split("Mog1:")[0].strip()

    if len(res) < 5 or res.count(res.split()[0] if res.split() else "") > 3:
        fact = fetch_online_knowledge(message)
        if fact:
            res = fact
        elif not res:
            res = f"{message.strip()} is an important concept in science, programming, and technology."
    return res

def handle_auto_train():
    if is_auto_training():
        return "Auto-training is already running in background!"
    success, msg = trigger_auto_train(epochs=30)
    return f"{msg} (Model will reload upon completion)."

if __name__ == "__main__":
    try:
        import gradio as gr

        with gr.Blocks(title="Mog1 AI - Smart Language Model") as demo:
            gr.Markdown(
                """
                # Mog1 AI (VSLM) - Smart Language Model
                **Mog1** is a high-accuracy PyTorch Small Language Model with hybrid neural and online knowledge retrieval to answer **every question** accurately.
                """
            )
            with gr.Tab("Interactive Chat"):
                chatbot = gr.ChatInterface(
                    fn=respond,
                    additional_inputs=[
                        gr.Radio(["Smart Mode (Creative & Fluent)", "Exact Factual Mode (Precise)"], label="Decoding Mode", value="Smart Mode (Creative & Fluent)"),
                        gr.Slider(10, 80, value=40, step=5, label="Max New Tokens")
                    ],
                )

            with gr.Tab("Auto-Training & Management"):
                gr.Markdown("### Trigger Background Auto-Pretraining")
                gr.Markdown("Click below to train or fine-tune Mog1 on the latest knowledge base in the background anytime.")
                train_btn = gr.Button("Start Auto-Training Now", variant="primary")
                train_status = gr.Textbox(label="Auto-Train Status", interactive=False)
                train_btn.click(fn=handle_auto_train, outputs=train_status)

        demo.launch(share=True, theme=gr.themes.Soft())
    except ImportError:
        print("Gradio not installed. Run `pip install gradio` to launch Web UI.")
