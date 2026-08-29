import os
import sys
import threading
import time
from train import train_instruction_model, one_shot_train

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

_auto_train_thread = None
_auto_train_lock = threading.Lock()
_is_training = False

def _run_training(epochs: int = 40, is_oneshot: bool = False):
    global _is_training
    try:
        with _auto_train_lock:
            _is_training = True

        if is_oneshot:
            print("⚡ Starting SFT Quick Pretraining...", flush=True)
            res = one_shot_train()
            print(f"⚡ SFT Quick Training finished in {res['elapsed_time']:.2f}s! Best Val Loss: {res['best_val_loss']:.4f}", flush=True)
        else:
            print(f"Starting background instruction training for {epochs} epochs...", flush=True)
            res = train_instruction_model(epochs=epochs)
            print(f"Instruction training finished! Best Val Loss: {res['best_val_loss']:.4f}", flush=True)
    except Exception as e:
        print(f"Error during training: {e}", flush=True)
    finally:
        with _auto_train_lock:
            _is_training = False

def trigger_auto_train(epochs: int = 40, is_oneshot: bool = False, is_claude_level: bool = False):
    global _auto_train_thread, _is_training
    with _auto_train_lock:
        if _is_training:
            return False, "Training is already in progress."

    actual_epochs = 60 if is_claude_level else epochs
    _auto_train_thread = threading.Thread(target=_run_training, args=(actual_epochs, is_oneshot), daemon=True)
    _auto_train_thread.start()
    if is_oneshot:
        mode_str = "SFT Quick Training (35 Epochs)"
    elif is_claude_level:
        mode_str = "Deep SFT Training (60 Epochs, Cosine Annealing)"
    else:
        mode_str = f"Standard Training ({epochs} Epochs)"
    return True, f"Started {mode_str}!"

def is_auto_training() -> bool:
    with _auto_train_lock:
        return _is_training

if __name__ == "__main__":
    success, msg = trigger_auto_train(is_oneshot=True)
    print(msg)
    time.sleep(3)
