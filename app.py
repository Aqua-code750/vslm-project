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

def respond(message: str, history, mode: str, max_tokens: int, temperature: float, top_p: float, top_k: int):
    if not message or not message.strip():
        return ""

    formatted = f"User: {message.strip()}\nMog1:"
    context_tokens = tokenizer.encode(formatted)
    context = torch.tensor(context_tokens, dtype=torch.long, device=DEVICE).unsqueeze(0)

    if "Exact" in mode:
        temp = 0.2
        tk = 3
        tp = 0.85
    elif "Creative" in mode:
        temp = float(temperature)
        tk = int(top_k)
        tp = float(top_p)
    else: # Smart Mode
        temp = 0.5
        tk = 15
        tp = 0.95

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

    if not res or len(res) < 3:
        fact = fetch_online_knowledge(message)
        if fact:
            res = fact
        elif not res:
            res = f"{message.strip()} is a topic in science, programming, and AI."
    return res

def handle_auto_train():
    if is_auto_training():
        return "Auto-training is already running in background!"
    success, msg = trigger_auto_train(epochs=30)
    return f"{msg} (Model will reload upon completion)."

if __name__ == "__main__":
    try:
        import gradio as gr

        with gr.Blocks(title="Mog1 AI - Freedom Language Model") as demo:
            gr.Markdown(
                """
                # Mog1 AI (VSLM) - Unrestricted Free-Play Chat
                **Mog1** gives you 100% full creative freedom. Adjust temperature, top-p, top-k, and sampling parameters to chat freely!
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
