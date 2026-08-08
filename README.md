---
title: Mog1 AI (VSLM) - Small Language Model
emoji: 🚀
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.22.0
app_file: app.py
pinned: false
license: mit
language:
- en
tags:
- pytorch
- language-model
- transformer
- small-language-model
- vslm
- mog1-ai
- text-generation
- built-from-scratch
- auto-training
pipeline_tag: text-generation
---

# 🚀 Mog1 AI (VSLM) - 3.3M Parameter PyTorch Small Language Model

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Aquaholograph2014%2Fmog1--ai--vslm-blue)](https://huggingface.co/spaces/Aquaholograph2014/mog1-ai-vslm)
[![GitHub](https://img.shields.io/badge/GitHub-Aqua--code750%2Fvslm--project-181717?logo=github)](https://github.com/Aqua-code750/vslm-project)

**Mog1 AI (VSLM)** is an advanced, lightweight **3.3 Million Parameter** Small Language Model built completely **from scratch in PyTorch** with zero black-box dependencies. Created by **Aqua-code750** / **Aquaholograph2014**. Designed for high-speed local inference, multi-domain reasoning, free-form interactive chat, and automatic internal pretraining.

---

## 🌟 Key Features

- 🧠 **Built From Scratch in PyTorch**: Pure PyTorch implementation of Decoder-Only Multi-Head Self-Attention Transformer blocks (~3.3M parameters).
- 📚 **Multi-Domain Internet Knowledge Base**: Pretrained on Computer Science, AI/ML, Python, Science, Math, History, General Knowledge, and Natural Dialogue.
- 🔄 **Auto-Training Engine**: Internal automatic pretraining triggers and 1-click Web UI fine-tuning button to update knowledge anytime.
- 💬 **Free-Form Interactive Chat**: Real-time CLI (`chat.py`) and Gradio Web UI (`app.py`) for natural conversational Q&A.
- 🎯 **Dual Sampling Modes**:
  - **Smart Mode**: Top-P (Nucleus) & Top-K sampling with temperature scaling for creative, fluent dialogue.
  - **Exact Factual Mode**: Low-temperature greedy decoding for precise technical facts.
- ⚡ **Lightning Fast CPU & GPU Execution**: Runs instantly on standard laptops or CPU servers without requiring high-end GPUs.

---

## 🏗️ Architecture Specifications

| Parameter | Specification |
| :--- | :--- |
| **Model Parameters** | **3,354,624 Parameters (~3.3 Million)** |
| **Architecture** | Decoder-Only Causal Transformer |
| **Attention Mechanism** | 8 Multi-Head Self-Attention Heads |
| **Embedding Dimension** | 256 |
| **Transformer Layers** | 4 Blocks |
| **Context Window** | 64 Tokens |
| **Tokenization** | Subword BPE / Tiktoken GPT-2 Fallback |
| **Optimization** | AdamW with Cosine Annealing Learning Rate Scheduler |

---

## 🛠️ Installation & Setup

1. **Clone the Repository**:
```bash
git clone https://github.com/Aqua-code750/vslm-project.git
cd vslm-project
```

2. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

---

## 💬 Usage Guide

### 1. Gradio Web Interface (Hugging Face / Local)
Launch the interactive web interface locally or deploy to Hugging Face Spaces:
```bash
python app.py
```

### 2. Interactive Terminal CLI Chat
Run the real-time CLI chat in your terminal:
```bash
python chat.py
```
- Type `/smart` to activate Smart Reasoning Mode.
- Type `/exact` to activate Factual Precision Mode.
- Type `/auto-train` to trigger background model training.

### 3. Standalone Generation & Testing
Generate answers programmatically for any prompt:
```bash
python generate.py
```

---

## 📜 License
This project is licensed under the [MIT License](LICENSE).
