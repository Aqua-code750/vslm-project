import os
import sys
import time
import threading
from train import train_mog1

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

KNOWLEDGE_FILE = "knowledge_base.txt"
CHECKPOINT_FILE = "vslm_checkpoint.pt"

_is_training = False
_lock = threading.Lock()

def trigger_auto_train(epochs: int = 50, force: bool = False):
    global _is_training
    with _lock:
        if _is_training and not force:
            print("Auto-train already in progress...")
            return False, "Auto-training is already running in background."
        _is_training = True

    def _worker():
        global _is_training
        try:
            print("[Auto-Train Engine] Starting automatic pretraining background task...", flush=True)
            loss = train_mog1(data_path=KNOWLEDGE_FILE, epochs=epochs)
            print(f"[Auto-Train Engine] Task completed successfully with final loss {loss:.4f}!", flush=True)
        except Exception as e:
            print(f"[Auto-Train Engine] Training failed: {e}", flush=True)
        finally:
            with _lock:
                _is_training = False

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return True, "Auto-training launched in background successfully."

def is_auto_training():
    return _is_training

if __name__ == "__main__":
    print("Launching Mog1 Auto-Train Engine...")
    success, msg = trigger_auto_train(epochs=60)
    print(msg)
    while is_auto_training():
        time.sleep(2)
    print("Auto-training complete.")
