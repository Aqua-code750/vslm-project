import os
import sys
import torch
from model import Mog1
from dataset import SubwordTokenizer
from train import train_mog1
from auto_train import trigger_auto_train, is_auto_training

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def run_chat(checkpoint_path: str = "vslm_checkpoint.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint '{checkpoint_path}' not found. Pretraining Mog1 AI Model internally...", flush=True)
        train_mog1(epochs=60, save_path=checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    tokenizer = SubwordTokenizer(
        stoi=checkpoint.get('stoi'),
        itos=checkpoint.get('itos'),
        use_tiktoken=checkpoint.get('use_tiktoken', False)
    )

    model = Mog1(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("\n==================================================")
    print("      Mog1 AI Model - Interactive Free-Play Chat  ")
    print("==================================================")
    print("Commands:")
    print(" - Type '/smart' for Smart Reasoning Mode (Default)")
    print(" - Type '/exact' for Exact Factual Mode")
    print(" - Type '/auto-train' to trigger background model pretraining")
    print(" - Type 'exit' to quit.\n")

    current_mode = "smart"

    while True:
        try:
            user_input = input(f"User [{current_mode.upper()}]: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting Mog1 CLI. Goodbye!")
                break
            if user_input.lower() == "/smart":
                current_mode = "smart"
                print("Switched to Smart Mode (creative data reasoning).")
                continue
            if user_input.lower() == "/exact":
                current_mode = "exact"
                print("Switched to Exact Factual Mode (precise answers).")
                continue
            if user_input.lower() == "/auto-train":
                success, msg = trigger_auto_train(epochs=50)
                print(f"{msg}")
                continue

            prompt_formatted = f"User: {user_input}\nMog1:"
            context_tokens = tokenizer.encode(prompt_formatted)
            context = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(0)

            temp = 0.2 if current_mode == "exact" else 0.5
            top_k = 3 if current_mode == "exact" else 15
            top_p = 0.85 if current_mode == "exact" else 0.95

            out_tokens = model.generate(
                context,
                max_new_tokens=50,
                temperature=temp,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=1.25
            )

            new_token_ids = out_tokens[0][len(context_tokens):].tolist()
            raw_response = tokenizer.decode(new_token_ids)
            clean_response = raw_response.split("User:")[0].split("Mog1:")[0].strip()

            if not clean_response:
                clean_response = raw_response.strip()

            print(f"Mog1: {clean_response}\n")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting Mog1 CLI.")
            break

if __name__ == "__main__":
    run_chat()
