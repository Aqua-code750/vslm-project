import os
import sys
import threading
import time
from train import train_mog1, one_shot_train

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

_auto_train_thread = None
_auto_train_lock = threading.Lock()
_is_training = False

def _run_training(epochs: int = 30, is_oneshot: bool = False):
    global _is_training
    try:
        with _auto_train_lock:
            _is_training = True

        if is_oneshot:
            print("⚡ Starting 1-Shot Instant Pretraining...", flush=True)
            loss, elapsed = one_shot_train()
            print(f"⚡ 1-Shot Instant Pretraining finished in {elapsed:.2f}s! Loss: {loss:.4f}", flush=True)
        else:
            print("Starting background auto-pretraining...", flush=True)
            loss = train_mog1(epochs=epochs)
            print(f"Auto-pretraining finished! Loss: {loss:.4f}", flush=True)
    except Exception as e:
        print(f"Error during pretraining: {e}", flush=True)
    finally:
        with _auto_train_lock:
            _is_training = False

def trigger_auto_train(epochs: int = 30, is_oneshot: bool = False):
    global _auto_train_thread, _is_training
    with _auto_train_lock:
        if _is_training:
            return False, "Training is already in progress."

    _auto_train_thread = threading.Thread(target=_run_training, args=(epochs, is_oneshot), daemon=True)
    _auto_train_thread.start()
    mode_str = "1-Shot Instant Pretraining" if is_oneshot else "Background Auto-Pretraining"
    return True, f"Started {mode_str}!"

def is_auto_training() -> bool:
    with _auto_train_lock:
        return _is_training

if __name__ == "__main__":
    success, msg = trigger_auto_train(is_oneshot=True)
    print(msg)
    time.sleep(3)
