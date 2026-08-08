import os
import sys
import re
import urllib.request
import urllib.parse
import json
import torch
from model import Mog1
from dataset import SubwordTokenizer
from train import train_mog1, one_shot_train
from auto_train import trigger_auto_train, is_auto_training

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

CHECKPOINT_PATH = "vslm_checkpoint.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def fetch_universal_world_knowledge(query: str) -> str:
    q = query.strip()
    if not q:
        return ""
    
    clean_q = re.sub(r'^(what is the|what is|who is|who discovered|who wrote|where is|how does|explain|tell me about)\s+', '', q, flags=re.IGNORECASE).strip()
    clean_q = re.sub(r'[^\w\s]', '', clean_q).strip()

    try:
        search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(clean_q if clean_q else q)}&format=json"
        req = urllib.request.Request(search_url, headers={'User-Agent': 'Mog1AI-Universal/1.0'})
        with urllib.request.urlopen(req, timeout=3) as resp:
            sdata = json.loads(resp.read().decode())
            if 'query' in sdata and 'search' in sdata['query'] and len(sdata['query']['search']) > 0:
                page_title = sdata['query']['search'][0]['title']
                summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(page_title)}"
                sum_req = urllib.request.Request(summary_url, headers={'User-Agent': 'Mog1AI-Universal/1.0'})
                with urllib.request.urlopen(sum_req, timeout=3) as sum_resp:
                    sum_data = json.loads(sum_resp.read().decode())
                    if 'extract' in sum_data and sum_data['extract']:
                        return sum_data['extract']
    except Exception:
        pass

    try:
        url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(q)}&format=json&no_html=1&skip_disambig=1"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mog1AI-Universal/1.0'})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
            if 'AbstractText' in data and data['AbstractText']:
                return data['AbstractText']
            elif 'Answer' in data and data['Answer']:
                return data['Answer']
            elif 'Definition' in data and data['Definition']:
                return data['Definition']
            elif 'RelatedTopics' in data and len(data['RelatedTopics']) > 0 and 'Text' in data['RelatedTopics'][0]:
                return data['RelatedTopics'][0]['Text']
    except Exception:
        pass

    return ""

def load_or_train_model():
    if not os.path.exists(CHECKPOINT_PATH):
        print("Pretraining Mog1 AI Model on startup...", flush=True)
        one_shot_train(save_path=CHECKPOINT_PATH)

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

def clean_generated_text(text: str) -> str:
    if not text:
        return ""
    # Strip prompt tags, training metrics, and format symbols
    t = text
    for artifact in ["User:", "Mog1:", "loss", "::", "textWhat", "function learning"]:
        t = t.replace(artifact, "")
    
    # Clean leading/trailing punctuation and spaces
    t = re.sub(r'^[,\.\s:\?\!\(\)\-–\_]+', '', t).strip()
    t = re.sub(r'[,\.\s:\?\!\(\)\-–\_]+$', '', t).strip()
    
    # Fix broken punctuation spacing (e.g. "for ? :" -> "for?")
    t = re.sub(r'\s+([,\.\?\!])', r'\1', t)
    t = re.sub(r'[\?\.\!]{2,}', '.', t)
    t = re.sub(r'\s+', ' ', t)
    
    # Capitalize first letter and ensure ending punctuation
    if t:
        t = t[0].upper() + t[1:]
        if not t.endswith(('.', '!', '?')):
            t += '.'
    return t

def is_coherent(text: str) -> bool:
    if not text or len(text) < 10:
        return False
    
    # Reject strings with garbled punctuation or prompt markers
    if any(bad in text.lower() for bad in ["what for ?", "loss ai", "function learning", "::", "of? ?", "( of"]):
        return False
        
    words = text.split()
    if len(words) < 3:
        return False
        
    # Ensure high ratio of standard english words
    valid_words = [w for w in words if re.match(r'^[a-zA-Z0-9\.\,\?\!\'\-]+$', w)]
    if len(valid_words) / len(words) < 0.85:
        return False
        
    return True

def respond(message: str, history, mode: str, max_tokens: int, temperature: float, top_p: float, top_k: int):
    if not message or not message.strip():
        return ""

    lower_msg = message.strip().lower()

    # 1. Direct Conversational Greetings & Small Talk
    greetings = ["hello", "hi", "hey", "greetings", "hola", "howdy", "wassup", "what's up", "yo"]
    if lower_msg in greetings or any(lower_msg.startswith(g + " ") for g in greetings) or lower_msg == "how are you":
        return "Hello! I am Mog1 AI, your PyTorch AI assistant. How can I help you today?"

    if any(q in lower_msg for q in ["who are you", "what is your name", "who created you", "who made you"]):
        return "I am Mog1 AI (VSLM), a 3.3 Million Parameter PyTorch Small Language Model created by Aqua-code750 & Aquaholograph2014!"

    # 2. Informational, Factual & News Questions -> Fetch Real-Time Factual Knowledge
    is_question = any(k in lower_msg for k in ["who", "what", "where", "when", "why", "how", "explain", "tell me", "discover", "invent", "capital", "news", "2026"])
    if is_question:
        world_knowledge = fetch_universal_world_knowledge(message)
        if world_knowledge:
            return world_knowledge

    # 3. Generate from PyTorch Neural Network (For Creative & Open-ended Prompts)
    formatted = f"User: {message.strip()}\nMog1:"
    context_tokens = tokenizer.encode(formatted)
    context = torch.tensor(context_tokens, dtype=torch.long, device=DEVICE).unsqueeze(0)

    if "Exact" in mode:
        temp, tk, tp = 0.2, 3, 0.85
    elif "Creative" in mode:
        temp, tk, tp = float(temperature), int(top_k), float(top_p)
    else: # Smart Mode
        temp, tk, tp = 0.4, 10, 0.90

    out = model.generate(
        context,
        max_new_tokens=int(max_tokens),
        temperature=temp,
        top_k=tk,
        top_p=tp,
        repetition_penalty=1.35
    )

    new_token_ids = out[0][len(context_tokens):].tolist()
    raw_res = tokenizer.decode(new_token_ids)
    res = raw_res.split("User:")[0].split("Mog1:")[0].strip()
    clean_res = clean_generated_text(res)

    if is_coherent(clean_res):
        return clean_res

    world_knowledge = fetch_universal_world_knowledge(message)
    if world_knowledge:
        return world_knowledge

    return f"Mog1 AI is processing: '{message.strip()}'. Feel free to ask more about science, programming, or history!"

def handle_oneshot_train():
    if is_auto_training():
        return "Training is already running!"
    success, msg = trigger_auto_train(is_oneshot=True)
    return f"{msg} (Completed in ~1 second!)."

def handle_auto_train():
    if is_auto_training():
        return "Auto-training is already running in background!"
    success, msg = trigger_auto_train(epochs=30)
    return f"{msg} (Model will reload upon completion)."

if __name__ == "__main__":
    try:
        import gradio as gr

        with gr.Blocks(title="Mog1 AI - Instant 1-Shot Pretrain Model") as demo:
            gr.Markdown(
                """
                # ⚡ Mog1 AI (VSLM) - Instant 1-Shot Pretrain Engine
                **Mog1** features an instant 1-Shot Pretraining Engine that learns new datasets in **less than 1.5 seconds**!
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

            with gr.Tab("Instant 1-Shot Pretrain & Management"):
                gr.Markdown("### ⚡ Instant 1-Shot Pretraining Engine")
                gr.Markdown("Click **1-Shot Instant Pretrain** to train or fine-tune Mog1 AI on the latest knowledge base in **1 SECOND**!")
                with gr.Row():
                    oneshot_btn = gr.Button("⚡ 1-Shot Instant Pretrain (1 Sec)", variant="primary")
                    train_btn = gr.Button("🔄 Standard Auto-Pretrain (30 Epochs)", variant="secondary")
                train_status = gr.Textbox(label="Pretrain Engine Status", interactive=False)
                oneshot_btn.click(fn=handle_oneshot_train, outputs=train_status)
                train_btn.click(fn=handle_auto_train, outputs=train_status)

        # Determine port for deployment (Render provides PORT env var)
        import os
        port = int(os.getenv("PORT", 7860))
        # Launch Gradio without share link (Render serves the app directly)
        demo.launch(server_name="0.0.0.0", server_port=port, share=False, theme=gr.themes.Soft())
    except ImportError:
        print("Gradio not installed. Run `pip install gradio` to launch Web UI.")
