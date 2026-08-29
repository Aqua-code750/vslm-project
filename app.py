import os
import sys
import time
import json
import torch
import torch.nn.functional as F

from model import Mog1, Mog1Config
from dataset import SubwordTokenizer
from train import train_instruction_model, one_shot_train
from auto_train import trigger_auto_train, is_auto_training

# Set encoding for Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
BEST_CHECKPOINT_PATH = "vslm_checkpoint_best.pt"
DEFAULT_CHECKPOINT_PATH = "vslm_checkpoint.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_best_checkpoint_path() -> str:
    if os.path.exists(BEST_CHECKPOINT_PATH):
        return BEST_CHECKPOINT_PATH
    return DEFAULT_CHECKPOINT_PATH

def load_or_train_model():
    ckpt_path = get_best_checkpoint_path()
    if not os.path.exists(ckpt_path):
        print("Pretraining Mog1 AI Model on startup...", flush=True)
        train_instruction_model(epochs=35, save_path=DEFAULT_CHECKPOINT_PATH)
        ckpt_path = get_best_checkpoint_path()

    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    config = checkpoint['config']

    tokenizer = SubwordTokenizer(
        stoi=checkpoint.get('stoi'),
        itos=checkpoint.get('itos')
    )

    model = Mog1(config).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    val_loss = checkpoint.get('val_loss', 'N/A')
    val_ppl = checkpoint.get('val_ppl', 'N/A')
    print(f"⭐ Loaded BEST Mog1 Checkpoint '{ckpt_path}' ({model.get_num_params():,} params, Val Loss: {val_loss}, Val PPL: {val_ppl}) on {DEVICE}.", flush=True)
    return model, tokenizer

model, tokenizer = load_or_train_model()

def respond(message: str, history, mode: str, max_tokens: int, temperature: float, top_p: float, top_k: int) -> str:
    """
    Pure Neural Autoregressive Generation using Mog1 Transformer, RoPE, RMSNorm, SwiGLU, and ChatML formatting.
    """
    if not message or not message.strip():
        return ""

    messages = [
        {"role": "system", "content": "You are Mog1 AI, a helpful, coherent, and precise conversational language model."}
    ]

    # Format multi-turn conversation history
    if history:
        for item in history[-3:]:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                u, b = item
                if u: messages.append({"role": "user", "content": str(u)})
                if b: messages.append({"role": "assistant", "content": str(b)})
            elif isinstance(item, dict) and "role" in item and "content" in item:
                messages.append({"role": item["role"], "content": str(item["content"])})

    messages.append({"role": "user", "content": message.strip()})

    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    input_ids = torch.tensor([tokenizer.encode(prompt_text)], dtype=torch.long, device=DEVICE)

    # Configure sampling parameters
    if "Creative" in mode:
        temp = max(temperature, 0.75)
        tp = 0.95
    elif "Factual" in mode:
        temp = 0.2
        tp = 0.7
    else:
        temp = temperature
        tp = top_p

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temp,
            top_k=top_k,
            top_p=tp,
            repetition_penalty=1.15,
            stop_token_ids=[tokenizer.im_end_id, tokenizer.eos_token_id]
        )

    generated_tokens = output_ids[0][input_ids.shape[1]:].tolist()
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return answer

def handle_oneshot_train():
    if is_auto_training():
        return "Training is already running!"
    success, msg = trigger_auto_train(is_oneshot=True)
    return f"{msg} (Completed in ~1 second!)."

def handle_auto_train():
    if is_auto_training():
        return "Auto-training is already running in background!"
    success, msg = trigger_auto_train(epochs=40)
    return f"{msg} (Model will reload upon completion)."

def handle_claude_train():
    if is_auto_training():
        return "Training is already running in background!"
    success, msg = trigger_auto_train(is_claude_level=True)
    return f"{msg} (Optimizing weights with Cosine Annealing & AdamW)."

if __name__ == "__main__":
    try:
        import gradio as gr

        with gr.Blocks(title="Mog1 AI - Modernized Small Language Model") as demo:
            gr.Markdown(
                """
                # ⚡ Mog1 AI (VSLM) - Modernized Small Language Model
                **Mog1** is built from scratch with **RMSNorm**, **RoPE (Rotary Embeddings)**, **SwiGLU**, **KV Caching**, and **Tied Embeddings** (~6.2M parameters).
                """
            )
            with gr.Tab("Interactive Neural Chat"):
                chatbot = gr.ChatInterface(
                    fn=respond,
                    additional_inputs=[
                        gr.Radio(["Smart Reasoning Mode", "Free Creative Freedom Mode", "Exact Factual Mode"], label="Reasoning & Generation Mode", value="Smart Reasoning Mode"),
                        gr.Slider(10, 200, value=60, step=5, label="Max New Tokens"),
                        gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="Temperature"),
                        gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-P (Nucleus Threshold)"),
                        gr.Slider(1, 50, value=30, step=1, label="Top-K Candidate Window")
                    ],
                )

            with gr.Tab("Instant Pretrain & Management"):
                gr.Markdown("### ⚡ Pretraining & Fine-Tuning Engine")
                gr.Markdown("Train Mog1 AI with SFT Loss Masking and Cosine Annealing learning rate schedule:")
                with gr.Row():
                    oneshot_btn = gr.Button("⚡ SFT Quick Train (35 Epochs)", variant="primary")
                    claude_btn = gr.Button("🧠 Deep SFT Train (60 Epochs)", variant="primary")
                    train_btn = gr.Button("🔄 Standard Train (40 Epochs)", variant="secondary")
                train_status = gr.Textbox(label="Pretrain Engine Status", interactive=False)
                oneshot_btn.click(fn=handle_oneshot_train, outputs=train_status)
                claude_btn.click(fn=handle_claude_train, outputs=train_status)
                train_btn.click(fn=handle_auto_train, outputs=train_status)

        # Determine port for deployment (Render provides PORT env var)
        port = int(os.getenv("PORT", 7860))
        demo.launch(server_name="0.0.0.0", server_port=port, share=False, theme=gr.themes.Soft())
    except ImportError:
        print("Gradio not installed. Run `pip install gradio` to launch Web UI.")
