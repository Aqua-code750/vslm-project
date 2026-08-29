import os
import sys
import torch
from generate import generate_text, get_loaded_model_and_tokenizer
from auto_train import trigger_auto_train, is_auto_training

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    print("=" * 60)
    print("🚀 Mog1 AI (VSLM) Interactive Neural Terminal Chat")
    print("Commands:")
    print("  /smart      : Smart Reasoning Mode (Default - Focused & Coherent)")
    print("  /exact      : Factual Precision Mode (Greedy Argmax)")
    print("  /free       : Free Creative Mode")
    print("  /clear      : Reset conversation history")
    print("  /auto-train : Trigger background model training")
    print("  /exit       : Quit chat")
    print("=" * 60)

    # Preload model weights on startup
    print("⚡ Loading Mog1 Neural Weights...", flush=True)
    get_loaded_model_and_tokenizer()
    print("✅ Model Ready!\n" + "-" * 60)

    mode = "smart"
    history = []

    while True:
        try:
            prompt = input(f"\nUser [{mode}]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat. Goodbye!")
            break

        if not prompt:
            continue

        if prompt.lower() in ["/exit", "exit", "quit"]:
            print("Goodbye!")
            break
        elif prompt.lower() == "/clear":
            history = []
            print("🧹 Conversation history cleared.")
            continue
        elif prompt.lower() == "/free":
            mode = "free"
            print("🎨 Switched to Free Creative Mode.")
            continue
        elif prompt.lower() == "/smart":
            mode = "smart"
            print("💡 Switched to Smart Reasoning Mode.")
            continue
        elif prompt.lower() == "/exact":
            mode = "exact"
            print("🎯 Switched to Exact Factual Mode.")
            continue
        elif prompt.lower() == "/auto-train":
            if is_auto_training():
                print("⏳ Auto-training is already running in background.")
            else:
                success, msg = trigger_auto_train(epochs=30)
                print(f"🚀 {msg}")
            continue

        res = generate_text(prompt, max_new_tokens=160, mode=mode, history=history)
        print(f"Mog1: {res}")

        # Append to history
        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": res})

        # Keep last 4 turns
        if len(history) > 8:
            history = history[-8:]

if __name__ == "__main__":
    main()
