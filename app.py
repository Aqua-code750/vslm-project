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
from qwen_engine import generate_qwen_stream, DEFAULT_QWEN_MODEL

# Set encoding for Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BEST_CHECKPOINT_PATH = "vslm_checkpoint_best.pt"
DEFAULT_CHECKPOINT_PATH = "vslm_checkpoint.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_mog1_model = None
_mog1_tokenizer = None

def get_best_checkpoint_path() -> str:
    if os.path.exists(BEST_CHECKPOINT_PATH):
        return BEST_CHECKPOINT_PATH
    return DEFAULT_CHECKPOINT_PATH

def load_or_train_mog1():
    global _mog1_model, _mog1_tokenizer
    if _mog1_model is not None and _mog1_tokenizer is not None:
        return _mog1_model, _mog1_tokenizer

    ckpt_path = get_best_checkpoint_path()
    if not os.path.exists(ckpt_path):
        print("Pretraining Mog1 AI Model on startup...", flush=True)
        train_instruction_model(epochs=35, save_path=DEFAULT_CHECKPOINT_PATH)
        ckpt_path = get_best_checkpoint_path()

    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    config = checkpoint['config']

    _mog1_tokenizer = SubwordTokenizer(
        stoi=checkpoint.get('stoi'),
        itos=checkpoint.get('itos')
    )

    _mog1_model = Mog1(config).to(DEVICE)
    _mog1_model.load_state_dict(checkpoint['model_state_dict'])
    _mog1_model.eval()
    return _mog1_model, _mog1_tokenizer

def respond_stream(
    message: str,
    history,
    mode: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    engine_choice: str = "Qwen 1.5 0.5B Chat"
):
    """
    Real-time streaming token generator for instant UI typing.
    Routes to Qwen 1.5 Chat weights or custom Mog1 checkpoint based on user selection.
    """
    if not message or not message.strip():
        yield ""
        return

    # Route to Qwen engine when in Chat Mode or Qwen is selected
    if "Chat" in mode or "Qwen" in engine_choice:
        model_name = "Qwen/Qwen1.5-1.8B-Chat" if "1.8B" in engine_choice else "Qwen/Qwen1.5-0.5B-Chat"
        temp = 0.0 if "Exact" in mode else (0.3 if "Smart" in mode else (0.7 if "Chat" in mode else max(temperature, 0.7)))
        for partial in generate_qwen_stream(
            prompt=message,
            history=history,
            model_name=model_name,
            max_new_tokens=max_tokens,
            temperature=temp,
            top_p=top_p
        ):
            yield partial
        return

    # Fallback to Mog1 scratch model
    model, tokenizer = load_or_train_mog1()
    messages = [
        {"role": "system", "content": "You are Mog1 SI (Super Intelligence), a helpful and coherent conversational assistant created by Holograph Inc."}
    ]

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

    if "Creative" in mode or "Free" in mode:
        temp = max(temperature, 0.4)
        tp = min(top_p, 0.90)
        tk = max(top_k, 5)
        rep_pen = 1.1
    elif "Factual" in mode or "Exact" in mode:
        temp, tp, tk, rep_pen = 0.0, 1.0, 1, 1.0
    else:
        temp, tp, tk, rep_pen = 0.0, 1.0, 1, 1.0

    accumulated_tokens = []
    stop_ids = {tokenizer.im_end_id, tokenizer.eos_token_id}

    for token_id in model.generate_stream(
        input_ids,
        max_new_tokens=max_tokens,
        temperature=temp,
        top_k=tk,
        top_p=tp,
        repetition_penalty=rep_pen,
        stop_token_ids=list(stop_ids)
    ):
        if token_id in stop_ids:
            break
        accumulated_tokens.append(token_id)
        raw_text = tokenizer.decode(accumulated_tokens, skip_special_tokens=False)
        for marker in ["<|im_end|>", "<|im_start|>", "<|eos|>"]:
            if marker in raw_text:
                raw_text = raw_text.split(marker)[0]
        yield raw_text.strip()

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

        with gr.Blocks(title="Mog1 AI - Conversational Neural Assistant") as demo:
            gr.Markdown(
                """
                # ⚡ Mog1 AI (Powered by Qwen 1.5 Conversational Weights)
                Conversational Intelligence upgraded with **Qwen 1.5 Chat** weights, featuring full multi-turn dialog, RoPE, RMSNorm, and real-time streaming inference.
                """
            )
            with gr.Tab("Interactive Neural Chat"):
                chatbot = gr.Chatbot(label="Mog1 Conversation", height=420)
                with gr.Row():
                    msg_input = gr.Textbox(placeholder="Ask Mog1 anything (e.g. 'Hey bro!', 'Explain how transformers work', 'Write a Python function')...", scale=8, show_label=False)
                    send_btn = gr.Button("Send 🚀", scale=2, variant="primary")

                with gr.Accordion("⚙️ Engine & Generation Settings", open=True):
                    with gr.Row():
                        engine_selector = gr.Dropdown(
                            ["Qwen 1.5 0.5B Chat", "Qwen 1.5 1.8B Chat", "Mog1 VSLM (Custom Checkpoint)"],
                            value="Qwen 1.5 0.5B Chat",
                            label="Neural Engine Weights"
                        )
                        mode_radio = gr.Radio(
                            ["💬 Chat Mode (Conversational)", "⚡ Smart Reasoning Mode", "🎨 Free Creative Freedom Mode", "🎯 Exact Factual Mode"],
                            label="Reasoning & Chat Mode",
                            value="💬 Chat Mode (Conversational)"
                        )
                    with gr.Row():
                        max_tok_slider = gr.Slider(16, 512, value=160, step=16, label="Max New Tokens")
                        temp_slider = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")
                        top_p_slider = gr.Slider(0.1, 1.0, value=0.85, step=0.05, label="Top-P")
                        top_k_slider = gr.Slider(1, 50, value=20, step=1, label="Top-K")
                    clear_btn = gr.Button("🗑️ Clear Chat History")

                def user_turn(user_msg, chat_hist):
                    if not user_msg or not user_msg.strip():
                        return "", chat_hist or []
                    chat_hist = chat_hist or []
                    return "", chat_hist + [[user_msg, ""]]

                def bot_turn(chat_hist, mode, max_tokens, temperature, top_p, top_k, engine_choice):
                    if not chat_hist:
                        return
                    user_msg = chat_hist[-1][0]
                    history_prior = chat_hist[:-1]
                    for partial_res in respond_stream(user_msg, history_prior, mode, max_tokens, temperature, top_p, top_k, engine_choice):
                        chat_hist[-1][1] = partial_res
                        yield chat_hist

                msg_input.submit(user_turn, [msg_input, chatbot], [msg_input, chatbot]).then(
                    bot_turn, [chatbot, mode_radio, max_tok_slider, temp_slider, top_p_slider, top_k_slider, engine_selector], [chatbot]
                )
                send_btn.click(user_turn, [msg_input, chatbot], [msg_input, chatbot]).then(
                    bot_turn, [chatbot, mode_radio, max_tok_slider, temp_slider, top_p_slider, top_k_slider, engine_selector], [chatbot]
                )
                clear_btn.click(lambda: [], outputs=[chatbot])

            with gr.Tab("VSLM Pretrain & Management"):
                gr.Markdown("### ⚡ Custom Scratch Model Fine-Tuning")
                gr.Markdown("Train the lightweight scratch Mog1 model with SFT Loss Masking and Cosine Annealing:")
                with gr.Row():
                    oneshot_btn = gr.Button("⚡ SFT Quick Train (35 Epochs)", variant="primary")
                    claude_btn = gr.Button("🧠 Deep SFT Train (60 Epochs)", variant="primary")
                    train_btn = gr.Button("🔄 Standard Train (40 Epochs)", variant="secondary")
                train_status = gr.Textbox(label="Pretrain Engine Status", interactive=False)
                oneshot_btn.click(fn=handle_oneshot_train, outputs=train_status)
                claude_btn.click(fn=handle_claude_train, outputs=train_status)
                train_btn.click(fn=handle_auto_train, outputs=train_status)

        # Determine port for deployment with automatic fallback if occupied
        port = int(os.getenv("PORT", 7860))
        try:
            demo.launch(server_name="0.0.0.0", server_port=port, share=False, theme=gr.themes.Soft())
        except OSError:
            print(f"⚠️ Port {port} is occupied. Automatically searching for next available open port...", flush=True)
            demo.launch(server_name="0.0.0.0", server_port=None, share=False, theme=gr.themes.Soft())
    except ImportError:
        print("Gradio not installed. Run `pip install gradio` to launch Web UI.")
