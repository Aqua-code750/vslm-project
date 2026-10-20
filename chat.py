import os
import sys
import torch
from generate import generate_stream, generate_text, get_loaded_model_and_tokenizer
from qwen_engine import DEFAULT_QWEN_MODEL, SUPPORTED_MODELS

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    active_model = DEFAULT_QWEN_MODEL
    print("=" * 65)
    print("🚀 Mog1 AI Interactive Terminal Chat (Powered by Qwen 1.5)")
    print(f"📦 Active Weights: {active_model}")
    print("Commands:")
    print("  /chat       : Chat Mode (Default - Full Conversational Neural Assistant)")
    print("  /smart      : Smart Reasoning Mode (Balanced & Coherent)")
    print("  /exact      : Precise Factual Mode (Greedy Argmax)")
    print("  /free       : Creative Freedom Mode")
    print("  /model      : Switch model (0.5B / 1.8B / 7B)")
    print("  /clear      : Reset conversation history")
    print("  /exit       : Quit chat")
    print("=" * 65)

    print("⚡ Initializing Qwen Weights...", flush=True)
    try:
        get_loaded_model_and_tokenizer()
        print("✅ Qwen Conversational Engine Ready!\n" + "-" * 65)
    except Exception as e:
        print(f"⚠️ Preload notice: {e}")

    mode = "chat"
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
        elif prompt.lower() == "/chat":
            mode = "chat"
            print("💬 Switched to Chat Mode (Full Conversational Qwen).")
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
        elif prompt.lower().startswith("/model"):
            parts = prompt.split()
            if len(parts) > 1:
                target = parts[1].strip()
                if "1.8" in target:
                    active_model = "Qwen/Qwen1.5-1.8B-Chat"
                elif "7" in target:
                    active_model = "Qwen/Qwen1.5-7B-Chat"
                else:
                    active_model = "Qwen/Qwen1.5-0.5B-Chat"
                print(f"🔄 Switching model weights to {active_model}...")
            else:
                print(f"Currently active model: {active_model}")
                print(f"Available options: {', '.join(SUPPORTED_MODELS)}")
            continue

        print("Mog1: ", end="", flush=True)
        full_res = ""
        try:
            last_len = 0
            for partial in generate_stream(prompt, history=history, mode=mode, model_name=active_model):
                new_chunk = partial[last_len:]
                sys.stdout.write(new_chunk)
                sys.stdout.flush()
                last_len = len(partial)
                full_res = partial
            print()
        except Exception as e:
            # Fallback to non-streaming if streamer encounters issue
            full_res = generate_text(prompt, history=history, mode=mode, model_name=active_model)
            print(full_res)

        # Append to history
        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": full_res})

        # Keep last 8 turns
        if len(history) > 16:
            history = history[-16:]

if __name__ == "__main__":
    main()
