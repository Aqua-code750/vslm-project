# Implementation Plan

## Deploy Mog1 AI as a permanently live web app on a free cloud platform (Render.com)

**Goal**: Move the Gradio app from Hugging Face (which has daily CPU limits) to a free cloud hosting service that can run 24/7 with a custom subdomain (e.g., `mog1.onrender.com`).

### User Review Required

> [!IMPORTANT]
> The following steps will create a new CI/CD pipeline on Render.com. You will need an active Render.com account and must grant the repository `Aqua-code750/vslm-project` access to Render. The free tier limits the service to **500 MB RAM** and **750 CPU‑hours/month**, which is sufficient for a lightweight Gradio inference.

> [!WARNING]
> The app will be publicly accessible. Ensure you do **not** expose any secret tokens in the repository (e.g., the Hugging Face API token currently hard‑coded in `app.py`).

### Open Questions

1. **Preferred platform** – You selected a free cloud platform; should we target **Render.com** (recommended) or another service like **Fly.io**?
2. **Custom sub‑domain** – Do you have an existing domain you’d like to point to the Render service, or is the default `*.onrender.com` acceptable?
3. **Environment variables** – Do you need any additional secrets (e.g., a new Hugging Face token for model loading) to be stored in Render’s environment?

### Proposed Changes

#### 1. Add Render deployment configuration
- **[NEW] `render.yaml`** – Defines the build and start commands for Render.
- **[MODIFY] `app.py`** – Adjust the Gradio launch to read the port from `os.getenv("PORT")` and bind to `0.0.0.0` (required by Render).
- **[NEW] `.gitignore` entry** – Ensure the local `venv/` folder is ignored if present.

---
#### File Details

**[NEW] `render.yaml`**
```yaml
services:
  - type: web
    name: mog1-ai
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: python app.py
    plan: free
```

**[MODIFY] `app.py` (excerpt)**
```python
import os
# ... existing imports ...

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    # Share=False because Render provides its own public URL
    iface = gr.Interface(fn=chat, ... )
    iface.launch(server_name="0.0.0.0", server_port=port, share=False)
```

---
#### 2. Push changes to GitHub (already linked to Render)
- After committing the new files, the GitHub Pages site will stay as a static wrapper, while Render serves the live Gradio interface at `https://mog1.onrender.com`.
- No further manual steps are required besides linking the repo in the Render dashboard.

### Verification Plan

- **Automated**: Run `python -c "import app; print('Import ok')"` locally to ensure syntax is valid after modifications.
- **Manual**:
  1. After pushing, open the Render dashboard, add the repository, and trigger a deploy.
  2. Verify the app loads at the provided Render URL and that the chat works.
  3. Confirm the 1‑click “Restart Space” banner on the GitHub Pages site still points to the Hugging Face space (optional).

---
**Next Step**: Await your approval and answers to the open questions before applying the changes.
