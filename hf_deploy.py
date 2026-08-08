import os
import sys
import time
from huggingface_hub import HfApi, create_repo, login

TOKEN = os.getenv("HF_TOKEN")
REPO_ID = "Aquaholograph2014/mog1-ai-vslm"

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def deploy():
    if TOKEN:
        login(token=TOKEN)
    api = HfApi()

    print(f"Ensuring Hugging Face Space '{REPO_ID}' exists...", flush=True)
    try:
        url = create_repo(
            repo_id=REPO_ID,
            token=TOKEN,
            repo_type="space",
            space_sdk="gradio",
            exist_ok=True
        )
        print(f"Space repository confirmed: {url}", flush=True)
    except Exception as e:
        print(f"Space creation info: {e}", flush=True)

    files_to_upload = [
        "README.md", "app.py", "model.py", "dataset.py", "train.py",
        "generate.py", "chat.py", "auto_train.py", "requirements.txt",
        "knowledge_base.txt"
    ]

    for filename in files_to_upload:
        if os.path.exists(filename):
            print(f"Uploading '{filename}' to {REPO_ID}...", flush=True)
            uploaded = False
            for attempt in range(5):
                try:
                    api.upload_file(
                        path_or_fileobj=filename,
                        path_in_repo=filename,
                        repo_id=REPO_ID,
                        repo_type="space",
                        token=TOKEN
                    )
                    uploaded = True
                    print(f"  ✓ {filename} uploaded.", flush=True)
                    break
                except Exception as err:
                    print(f"  Attempt {attempt+1} failed ({err}). Retrying in 3s...", flush=True)
                    time.sleep(3)
            if not uploaded:
                print(f"❌ Failed to upload {filename}.", flush=True)

    print(f"\n✅ Hugging Face Space Deployment Complete!", flush=True)
    print(f"🔗 View Space: https://huggingface.co/spaces/{REPO_ID}", flush=True)

if __name__ == "__main__":
    deploy()
