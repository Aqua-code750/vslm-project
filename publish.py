import os
import sys
import subprocess

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

DEFAULT_HF_REPO = "Aquaholograph2014/mog1-ai-vslm"
DEFAULT_GITHUB_REPO = "https://github.com/Aqua-code750/vslm-project.git"

def publish_to_huggingface(repo_id: str = DEFAULT_HF_REPO, token: str = None):
    print(f"Deploying to Hugging Face Space: {repo_id}...")
    if token:
        hf_git_url = f"https://Aquaholograph2014:{token}@huggingface.co/spaces/{repo_id}"
    else:
        hf_git_url = f"https://huggingface.co/spaces/{repo_id}"

    try:
        subprocess.run(["git", "init"], check=False)
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", "Deploy Mog1 AI VSLM to Hugging Face Space"], check=False)
        subprocess.run(["git", "remote", "remove", "hf"], check=False)
        subprocess.run(["git", "remote", "add", "hf", hf_git_url], check=True)
        res = subprocess.run(["git", "push", "hf", "main", "--force"], check=False)
        if res.returncode == 0:
            print("✅ Successfully published to Hugging Face Spaces!")
            print(f"🔗 View Space: https://huggingface.co/spaces/{repo_id}")
        else:
            print("⚠️ Hugging Face push notice: Ensure Space is created at https://huggingface.co/new-space")
    except Exception as e:
        print(f"❌ HF push failed: {e}")

def publish_to_github(remote_url: str = DEFAULT_GITHUB_REPO):
    print(f"Syncing repository to GitHub remote: {remote_url}...")
    try:
        subprocess.run(["git", "init"], check=False)
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", "Upgrade Mog1 AI VSLM with auto-train and smart knowledge base"], check=False)
        subprocess.run(["git", "branch", "-M", "main"], check=False)
        subprocess.run(["git", "remote", "remove", "origin"], check=False)
        subprocess.run(["git", "remote", "add", "origin", remote_url], check=True)
        res = subprocess.run(["git", "push", "-u", "origin", "main", "--force"], check=False)
        if res.returncode == 0:
            print("✅ Successfully published to GitHub!")
        else:
            print("❌ GitHub push failed.")
    except Exception as e:
        print(f"❌ GitHub push failed: {e}")

if __name__ == "__main__":
    print("Mog1 AI Publishing Suite (Aqua-code750 & Aquaholograph2014)")
    
    if len(sys.argv) > 1:
        action = sys.argv[1].lower()
        if action == "hf":
            token = sys.argv[2] if len(sys.argv) > 2 else None
            publish_to_huggingface(DEFAULT_HF_REPO, token)
        elif action == "github":
            repo_url = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_GITHUB_REPO
            publish_to_github(repo_url)
    else:
        print(f"  GitHub URL : {DEFAULT_GITHUB_REPO}")
        print(f"  HF Space   : {DEFAULT_HF_REPO}")
