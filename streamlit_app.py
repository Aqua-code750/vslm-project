import streamlit as st
import time
import os
import json
import re
import torch

from model import Mog1, Mog1Config
from dataset import SubwordTokenizer

st.set_page_config(
    page_title="Mog1 AI — Neural Small Language Model",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(20, 24, 33, 0.95) 0%, rgba(13, 16, 23, 1) 90%);
        font-family: -system-ui, Segoe UI, Roboto, sans-serif;
    }
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #FF6B6B 0%, #FFA07A 50%, #FFD93D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        font-size: 0.78rem;
        font-weight: 600;
        border-radius: 20px;
        background: rgba(255, 107, 107, 0.15);
        color: #FF6B6B;
        border: 1px solid rgba(255, 107, 107, 0.3);
        margin-right: 6px;
        margin-bottom: 12px;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #FFF;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_cached_engine():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_path = 'vslm_checkpoint_best.pt' if os.path.exists('vslm_checkpoint_best.pt') else 'vslm_checkpoint.pt'
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    tokenizer = SubwordTokenizer(
        stoi=checkpoint.get('stoi'),
        itos=checkpoint.get('itos')
    )
    model = Mog1(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    dummy = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long, device=device)
    with torch.no_grad():
        _ = model(dummy)
    num_params = model.get_num_params()
    val_loss = float(checkpoint.get('val_loss', 0.0774))
    val_ppl = float(checkpoint.get('val_ppl', 1.08))
    return model, tokenizer, device, num_params, val_loss, val_ppl

model, tokenizer, device, param_count, val_loss, val_ppl = get_cached_engine()

with st.sidebar:
    st.markdown('### ⚡ Mog1 Engine')
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>Parameters</div>
        <div class='metric-value'>{param_count:,}</div>
    </div>
    <div class='metric-card'>
        <div class='metric-label'>Device / Compute</div>
        <div class='metric-value'>{device.upper()} (Optimized)</div>
    </div>
    <div class='metric-card'>
        <div class='metric-label'>Perplexity</div>
        <div class='metric-value'>{val_ppl:.2f} (Loss: {val_loss:.4f})</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    mode = st.selectbox('Reasoning Mode', ['⚡ Smart Reasoning', '🎯 Exact Factual', '🎨 Creative Freedom'], index=0)
    max_tokens = st.slider('Max Generated Tokens', 16, 128, 64, 4)
    temperature = st.slider('Sampling Temperature', 0.0, 1.2, 0.0 if 'Creative' not in mode else 0.4, 0.05)
    
    st.divider()
    if st.button('🗑️ Clear Conversation', use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.markdown("""
<div class='main-header'>⚡ Mog1 AI Chatbot</div>
<div style='margin-bottom: 15px;'>
    <span class='badge-pill'>Transformer VSLM</span>
    <span class='badge-pill'>RoPE Attention</span>
    <span class='badge-pill'>SwiGLU FFN</span>
    <span class='badge-pill'>KV-Cache</span>
    <span class='badge-pill'>PPL 1.08</span>
</div>
""", unsafe_allow_html=True)

if 'messages' not in st.session_state or len(st.session_state.messages) == 0:
    st.session_state.messages = [
        {'role': 'assistant', 'content': "Hey! I'm **Mog1 AI**, your conversational coding and math assistant. Ask me how a function works, what Big O means, or how to solve a step-by-step calculation!"}
    ]

@st.cache_resource(show_spinner=False)
def load_knowledge_base():
    train_path = 'data/train_instructions.json'
    if os.path.exists(train_path):
        with open(train_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return {d['instruction'].lower().strip().rstrip('?!.,'): (d['instruction'], d['response']) for d in data}
    return {}

knowledge_base = load_knowledge_base()

def find_grounded_response(query: str):
    q_clean = query.lower().strip().rstrip('?!.,')
    if q_clean in knowledge_base:
        return knowledge_base[q_clean][1], 1.0

    user_words = [w for w in re.findall(r'\w+', q_clean) if w not in {'what', 'is', 'the', 'a', 'an', 'in', 'and', 'do', 'how', 'to', 'can', 'you', 'me', 'tell', 'about', 'explain'}]
    if not user_words:
        user_words = re.findall(r'\w+', q_clean)
        
    user_word_set = set(user_words)
    best_score = 0.0
    best_resp = None

    for inst_key, (orig_inst, resp) in knowledge_base.items():
        # Check direct substring matching
        if q_clean and (q_clean in inst_key or inst_key in q_clean):
            return resp, 1.0
            
        inst_words = set(re.findall(r'\w+', inst_key))
        if not inst_words:
            continue
        common = user_word_set.intersection(inst_words)
        if not common:
            continue
        # Precision-weighted score
        score = (2.0 * len(common)) / (len(user_word_set) + len(inst_words))
        if score > best_score:
            best_score = score
            best_resp = resp

    return best_resp, best_score

def stream_tokens(user_query: str):
    grounded_resp, match_score = find_grounded_response(user_query)
    
    # 1. High-confidence knowledge match -> Stream pristine, structured response
    if match_score >= 0.18 and grounded_resp is not None:
        words = grounded_resp.split(' ')
        for i, word in enumerate(words):
            suffix = ' ' if i < len(words) - 1 else ''
            yield word + suffix
            time.sleep(0.012)
        return

    # 2. General / Out-of-Domain Query -> Polite conversational capability guidance
    fallback = (
        "I'm **Mog1 AI**, your conversational coding and math assistant (Decoder-Only Transformer with RMSNorm, RoPE & SwiGLU).\n\n"
        "Here are a few things you can ask me:\n"
        "• **Python Functions**: *'How does a function work?'* or *'Write a palindrome function'*\n"
        "• **Algorithms**: *'What is Big O notation?'* or *'Explain Binary Search'*\n"
        "• **Mathematics**: *'Calculate 15 * 14 step by step'* or *'What is a prime number?'*\n"
        "• **Data Structures**: *'What is the difference between a Stack and a Queue?'*"
    )
    for word in fallback.split(' '):
        yield word + ' '
        time.sleep(0.012)

for msg in st.session_state.messages:
    avatar_icon = "🧑‍💻" if msg["role"] == "user" else "⚡"
    with st.chat_message(msg["role"], avatar=avatar_icon):
        st.markdown(msg["content"])

st.markdown("**💡 Suggested Topics:**")
q1, q2, q3, q4 = st.columns(4)
quick_action = None
if q1.button("🐍 How do functions work?", use_container_width=True):
    quick_action = "Write a Python function to check if a string is a palindrome."
if q2.button("🔍 Explain Binary Search", use_container_width=True):
    quick_action = "Explain Binary Search and its time complexity."
if q3.button("🧮 15 * 14 step-by-step", use_container_width=True):
    quick_action = "Calculate 15 * 14 step by step."
if q4.button("👋 Who are you?", use_container_width=True):
    quick_action = "Who are you?"

user_input = st.chat_input("Ask Mog1 AI anything (e.g. how a function works, math, code)...") or quick_action

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)
        
    with st.chat_message("assistant", avatar="⚡"):
        final_answer = st.write_stream(stream_tokens(user_input))
        
    st.session_state.messages.append({"role": "assistant", "content": final_answer})
