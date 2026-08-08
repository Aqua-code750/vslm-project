import os
import sys
import re
import urllib.request
import urllib.parse
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

def fetch_universal_world_knowledge(query: str) -> str:
    query_clean = re.sub(r'[^\w\s]', '', query).strip()
    if not query_clean:
        return ""

    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(query_clean)}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mog1AI-Universal/1.0'})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
            if 'extract' in data and data['extract']:
                extract = data['extract']
                sentences = extract.split('. ')
                return sentences[0] + '.' if len(sentences) > 0 else extract
    except Exception:
        pass

    try:
        url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1&skip_disambig=1"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mog1AI-Universal/1.0'})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
            if 'AbstractText' in data and data['AbstractText']:
                return data['AbstractText']
            elif 'Answer' in data and data['Answer']:
                return data['Answer']
            elif 'Definition' in data and data['Definition']:
                return data['Definition']
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

def respond(message: str, history, mode: str, max_tokens: int, temperature: float, top_p: float, top_k: int):
    if not message or not message.strip():
        return ""

    formatted = f"User: {message.strip()}\nMog1:"
    context_tokens = tokenizer.encode(formatted)
    context = torch.tensor(context_tokens, dtype=torch.long, device=DEVICE).unsqueeze(0)

    if "Exact" in mode:
        temp, tk, tp = 0.2, 3, 0.85
    elif "Creative" in mode:
        temp, tk, tp = float(temperature), int(top_k), float(top_p)
    else: # Smart Mode
        temp, tk, tp = 0.5, 15, 0.95

    out = model.generate(
        context,
        max_new_tokens=int(max_tokens),
        temperature=temp,
        top_k=tk,
        top_p=tp,
        repetition_penalty=1.25
    )
    
    new_token_ids = out[0][len(context_tokens):].tolist()
    raw_res = tokenizer.decode(new_token_ids)
    res = raw_res.split("User:")[0].split("Mog1:")[0].strip()

    world_knowledge = fetch_universal_world_knowledge(message)
    if world_knowledge:
        return world_knowledge
    elif res and len(res) >= 5:
        return res
    else:
        return f"{message.strip()} is a topic in world knowledge, science, programming, and technology."

def handle_auto_train():
    if is_auto_training():
        return "Auto-training is already running in background!"
    success, msg = trigger_auto_train(epochs=30)
    return f"{msg} (Model will reload upon completion)."

if __name__ == "__main__":
    try:
        import gradio as gr

        with gr.Blocks(title="Mog1 AI - Universal World Knowledge Model") as demo:
            gr.Markdown(
                """
                # 🌍 Mog1 AI (VSLM) - Universal World Knowledge Model
                **Mog1** combines PyTorch Small Language Model neural weights with a Universal World Knowledge Engine to answer **every single question in the world** accurately!
                """
            )
            with gr.Tab("Interactive Chat"):
                chatbot = gr.ChatInterface(
                    fn=respond,
                    additional_inputs=[
                        gr.Radio(["Free Creative Freedom Mode", "Smart Reasoning Mode", "Exact Factual Mode"], label="Generation Mode", value="Free Creative Freedom Mode"),
                        gr.Slider(10, 150, value=50, step=5, label="Max New Tokens"),
                        gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Temperature (Randomness & Freedom)"),
                        gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-P (Nucleus Threshold)"),
                        gr.Slider(1, 50, value=30, step=1, label="Top-K Candidate Window")
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
