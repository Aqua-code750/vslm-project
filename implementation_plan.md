# Implementation Plan

## Goal
Replace the Python‑based Gradio backend with a pure‑JavaScript (client‑side) inference pipeline so the AI runs directly in the browser without a Python server.

---

## User Review Required

> [!IMPORTANT]
> This change is a **major architectural rewrite**. It will:
> - Remove the `app.py` Gradio server.
> - Add a new JavaScript inference engine (ONNX/TensorFlow.js) that runs in the browser.
> - Require exporting the PyTorch model to an ONNX file and converting it to a TF.js format.
> - Increase the bundle size of the GitHub Pages site (≈ 10‑20 MB) and may affect load time.
> - Necessitate a modern browser with WebGL / WebGPU support.

> [!WARNING]
> The free GitHub Pages hosting has a 100 MB site limit. Ensure the exported model stays within that limit, or host the model file on a CDN (e.g., Hugging Face Hub) and load it dynamically.

---

## Open Questions

1. **Model size** – The current checkpoint `vslm_checkpoint.pt` is ~ 200 MB. Do you want to:
   - **Quantize** it to a smaller 8‑bit version (~ 50 MB) before conversion?
   - **Host the model file on a CDN** (e.g., Hugging Face) to keep the GitHub Pages bundle small?
2. **Target inference library** – Which JS runtime do you prefer?
   - `onnxruntime-web` (runs ONNX directly in the browser).
   - `tfjs` (requires conversion to TensorFlow.js format).
3. **Performance expectations** – Running a 3.3 M‑parameter model client‑side will be slower than the server version. Is that acceptable, or should we keep a lightweight fallback to the Python server for heavy queries?
4. **User interaction** – Should the UI keep the same chat layout, merely swapping the backend call to a JavaScript `predict()` function?

---

## Proposed Changes

### 1. Export & Convert the Model
- **[NEW] `export_model.py`** – Script that loads `vslm_checkpoint.pt`, exports to ONNX, optionally quantizes, and saves `model.onnx`.
- **[NEW] `convert_to_tfjs.sh`** (optional) – Uses `tensorflowjs_converter` to turn `model.onnx` into TensorFlow.js `model.json` + shard files.

### 2. Add JavaScript Inference Engine
- **[NEW] `js_infer.js`** – Wrapper around `onnxruntime-web` (or TF.js) that loads the model, tokenizes input (using a lightweight BPE tokenizer in JS), runs inference, and returns the generated text.
- **[NEW] `tokenizer.js`** – Minimal sub‑word tokenizer mirroring the Python `SubwordTokenizer` (stores `stoi`/`itos` JSON). Loaded from a static JSON file.
- **[MODIFY] `index.html`** – Replace the `fetchSmartResponse` function’s fallback to call `js_infer.js` when the query is not a direct fact. Remove the Gradio iframe embed (since there is no longer a Gradio server).
- **[NEW] `model/` directory** – Holds `model.onnx` (or `model.json` + shards) and `vocab.json` for the tokenizer.

### 3. Remove Python Server Files (optional, keep for fallback)
- **[DELETE] `app.py`** – No longer needed for the client‑side version (can be kept in a `fallback/` folder if you want a hybrid approach).
- **[DELETE] `publish.py`** – Not required for pure static deployment.

### 4. Update Build / Deployment
- **[MODIFY] `render.yaml`** – Not needed for pure static hosting; we’ll keep only GitHub Pages.
- **[MODIFY] `README.md`** – Add instructions on how to rebuild the model and redeploy.

---

## Verification Plan

### Automated Tests
- Run `python export_model.py` locally to ensure the ONNX file is generated without errors.
- Run `node js_infer.js "Who discovered gravity?"` (using a tiny Node script) to verify inference works.

### Manual Checks
1. Deploy the updated static site to GitHub Pages.
2. Open the live URL and ask a few questions (e.g., “Who invented buffet?”, “Who discovered gravity?”). Verify the answers appear quickly and match expectations.
3. Open the browser dev‑tools → Network tab and confirm the model file is loaded (size < 100 MB) and that inference runs without errors.
4. Test on a low‑end device (e.g., mobile) to gauge performance.

---

**Next Step**
- Please review the open questions and confirm the preferred inference library (ONNX Runtime Web vs. TensorFlow.js), model hosting strategy, and whether you’d like a fallback Python server.
- Once approved, I’ll create the necessary scripts, export the model, and push the changes.
