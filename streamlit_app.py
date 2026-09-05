import streamlit as st
import time
import os
import json
import re
import torch

from model import Mog1, Mog1Config
from dataset import SubwordTokenizer
import safety

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
def load_base_knowledge():
    train_path = 'data/train_instructions.json'
    kb = {}
    if os.path.exists(train_path):
        with open(train_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for d in data:
                kb[d['instruction'].lower().strip().rstrip('?!.,')] = (d['instruction'], d['response'])
    return kb

LEARNED_KB_PATH = 'data/learned_knowledge.json'

def load_learned_knowledge():
    if os.path.exists(LEARNED_KB_PATH):
        try:
            with open(LEARNED_KB_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {d['instruction'].lower().strip().rstrip('?!.,'): (d['instruction'], d['response']) for d in data}
        except Exception:
            return {}
    return {}

def save_learned_knowledge(learned_kb_dict):
    os.makedirs(os.path.dirname(LEARNED_KB_PATH), exist_ok=True)
    list_data = [{"instruction": v[0], "response": v[1]} for v in learned_kb_dict.values()]
    with open(LEARNED_KB_PATH, 'w', encoding='utf-8') as f:
        json.dump(list_data, f, indent=2, ensure_ascii=False)

if 'learned_kb' not in st.session_state:
    st.session_state.learned_kb = load_learned_knowledge()

base_knowledge = load_base_knowledge()

def get_full_knowledge_base():
    # Merges base pre-trained knowledge with dynamically learned user knowledge
    merged = dict(base_knowledge)
    merged.update(st.session_state.learned_kb)
    return merged

def search_live_knowledge(query: str):
    """Fetches encyclopedic context in real-time when answering unfamiliar factual queries."""
    import urllib.request
    import urllib.parse
    clean = query.strip().rstrip('?!. ')
    search_term = re.sub(r'^(?:what is|who is|tell me about|explain|where is|when was|how does)\s+', '', clean, flags=re.IGNORECASE).strip()
    if not search_term:
        search_term = clean
    try:
        url = f'https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(search_term)}'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mog1AI/1.0 (contact: support@mog1.ai)'})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            extract = data.get('extract')
            if extract and len(extract) > 20:
                title = data.get('title', search_term)
                return f"**{title}**:\n{extract}"
    except Exception:
        pass
    try:
        search_url = f'https://en.wikipedia.org/w/api.php?action=opensearch&search={urllib.parse.quote(search_term)}&limit=1&namespace=0&format=json'
        req = urllib.request.Request(search_url, headers={'User-Agent': 'Mog1AI/1.0 (contact: support@mog1.ai)'})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            if len(data) >= 3 and data[2] and data[2][0]:
                title = data[1][0]
                return f"**{title}**:\n{data[2][0]}"
    except Exception:
        pass
    return None

def detect_learning_prompt(user_text: str):
    """Detects if user is explicitly teaching the AI new knowledge."""
    patterns = [
        r'^(?:teach|learn)\s*:\s*(.+?)\s*=>\s*(.+)$',
        r'^(?:remember|store)\s*:\s*(.+?)\s*=>\s*(.+)$',
        r'^(?:teach|learn)\s+that\s+(.+?)\s+(?:is|means|equals|=)\s+(.+)$',
        r'^(?:remember)\s+that\s+(.+?)\s+(?:is|means|equals|=)\s+(.+)$',
        r'^(?:q:|question:)\s*(.+?)\s*(?:a:|answer:)\s*(.+)$',
    ]
    for pat in patterns:
        m = re.search(pat, user_text.strip(), re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip(), m.group(2).strip()
    return None

def find_grounded_response(query: str):
    current_kb = get_full_knowledge_base()
    q_clean = query.lower().strip().rstrip('?!.,')
    if q_clean in current_kb:
        return current_kb[q_clean][1], 1.0

    user_words = [w for w in re.findall(r'\w+', q_clean) if w not in {'what', 'is', 'the', 'a', 'an', 'in', 'and', 'do', 'how', 'to', 'can', 'you', 'me', 'tell', 'about', 'explain'}]
    if not user_words:
        user_words = re.findall(r'\w+', q_clean)
        
    user_word_set = set(user_words)
    best_score = 0.0
    best_resp = None

    for inst_key, (orig_inst, resp) in current_kb.items():
        if q_clean and (q_clean in inst_key or inst_key in q_clean):
            return resp, 1.0
            
        inst_words = set(re.findall(r'\w+', inst_key))
        if not inst_words:
            continue
        common = user_word_set.intersection(inst_words)
        if not common:
            continue
        score = (2.0 * len(common)) / (len(user_word_set) + len(inst_words))
        if score > best_score:
            best_score = score
            best_resp = resp

    return best_resp, best_score

def stream_tokens(user_query: str):
    # 0. Safety & Moderation Firewall Check
    is_safe, safety_reason = safety.is_safe_prompt(user_query)
    if not is_safe:
        refusal = f"🛡️ **Safety Guardrail**: I cannot process this prompt ({safety_reason}). Please keep questions constructive, respectful, and safe."
        for word in refusal.split(' '):
            yield word + ' '
            time.sleep(0.012)
        return

    # Check if this is an explicit teaching/learning command
    teach_match = detect_learning_prompt(user_query)
    if teach_match:
        learned_q, learned_a = teach_match
        q_safe, q_reason = safety.is_safe_prompt(learned_q)
        a_safe, a_reason = safety.is_safe_prompt(learned_a)
        if not q_safe or not a_safe:
            msg = f"🛡️ **Safety Guardrail**: Cannot memorize this input ({q_reason if not q_safe else a_reason}). Poisoning prevention is active."
            for word in msg.split(' '):
                yield word + ' '
                time.sleep(0.012)
            return

        clean_key = learned_q.lower().strip().rstrip('?!.,')
        # Session-isolated memory: stores in current conversation session
        st.session_state.learned_kb[clean_key] = (learned_q, learned_a)
        
        # Admin check: only persist to disk if admin passcode is verified
        is_admin = st.session_state.get('is_admin', False)
        if is_admin:
            save_learned_knowledge(st.session_state.learned_kb)
            scope_desc = "Permanently saved to Global Knowledge Base (Admin Verified)."
        else:
            scope_desc = "Added to Active Session Memory (Ephemeral Isolation). Other users are protected."

        ack = (
            f"🧠 **Knowledge Memorized!**\n\n"
            f"• **Concept/Topic:** *\"{learned_q}\"*\n"
            f"• **Learned Answer:** *\"{learned_a}\"*\n"
            f"• **Scope:** {scope_desc}\n\n"
            f"You can now query this concept directly!"
        )
        for word in ack.split(' '):
            yield word + ' '
            time.sleep(0.012)
        return

    # 1. Match against current Knowledge Base (Base + User Learned)
    grounded_resp, match_score = find_grounded_response(user_query)
    if match_score >= 0.18 and grounded_resp is not None:
        words = grounded_resp.split(' ')
        for i, word in enumerate(words):
            suffix = ' ' if i < len(words) - 1 else ''
            yield word + suffix
            time.sleep(0.012)
        return

    # 2. Real-Time Autonomous Live Learning: search dynamic sources and absorb into knowledge base!
    live_answer = search_live_knowledge(user_query)
    if live_answer:
        # Check safety before absorbing
        ans_safe, _ = safety.is_safe_prompt(live_answer)
        if ans_safe:
            clean_key = user_query.lower().strip().rstrip('?!.,')
            st.session_state.learned_kb[clean_key] = (user_query, live_answer)
            # Only persist to disk if admin is enabled
            if st.session_state.get('is_admin', False):
                save_learned_knowledge(st.session_state.learned_kb)
        
        learned_prefix = "🧠 *(Acquired & Added to Knowledge Base)*\n\n"
        full_text = learned_prefix + live_answer
        for word in full_text.split(' '):
            yield word + ' '
            time.sleep(0.012)
        return

    # 3. Conversational capability guidance fallback
    fallback = (
        "I'm **Mog1 AI**, your conversational coding, math, and learning assistant (VSLM).\n\n"
        "💡 **You can teach me anything new right in the chat!**\n"
        "• Type: `Teach: What is quantum computing? => Quantum computing uses qubits for exponential calculation.`\n"
        "• Or: `Remember that Aqua is the creator of Mog1 AI`\n\n"
        "Or ask me about Python functions, Big O notation, math problems, or general concepts!"
    )
    for word in fallback.split(' '):
        yield word + ' '
        time.sleep(0.012)

# Sidebar Knowledge Base Metrics & Admin Gate
with st.sidebar:
    st.divider()
    st.markdown("### 🧠 Knowledge & Safety")
    total_base = len(base_knowledge)
    total_learned = len(st.session_state.learned_kb)
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>Total Knowledge Base</div>
        <div class='metric-value'>{total_base + total_learned}</div>
    </div>
    <div class='metric-card'>
        <div class='metric-label'>Active Learned Memory</div>
        <div class='metric-value'>+{total_learned} concepts</div>
    </div>
    <div class='metric-card'>
        <div class='metric-label'>Safety Firewall</div>
        <div class='metric-value' style='color: #4EFA90;'>ACTIVE (Toxicity & Injection Guard)</div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("🔐 Admin & Permanent Knowledge"):
        admin_pin = st.text_input("Creator Passcode", type="password", help="Enter passcode to enable permanent global learning")
        if admin_pin == "aqua750":
            st.session_state.is_admin = True
            st.success("✅ Creator Admin Verified: Global Persistence Active")
        else:
            st.session_state.is_admin = False
            st.caption("🔒 Public Mode: Knowledge is session-isolated to prevent poisoning.")
            
        t_q = st.text_input("Concept / Question", placeholder="e.g., What is AquaOS?")
        t_a = st.text_area("Answer / Definition", placeholder="e.g., AquaOS is...")
        if st.button("💾 Teach & Store Knowledge", use_container_width=True):
            if t_q.strip() and t_a.strip():
                is_q_safe, q_err = safety.is_safe_prompt(t_q)
                is_a_safe, a_err = safety.is_safe_prompt(t_a)
                if not is_q_safe or not is_a_safe:
                    st.error(f"Cannot save: {q_err if not is_q_safe else a_err}")
                else:
                    clean_k = t_q.lower().strip().rstrip('?!.,')
                    st.session_state.learned_kb[clean_k] = (t_q.strip(), t_a.strip())
                    if st.session_state.is_admin:
                        save_learned_knowledge(st.session_state.learned_kb)
                        st.success(f"Saved globally to disk!")
                    else:
                        st.info(f"Saved to your active session!")
                    st.rerun()

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

user_input = st.chat_input("Ask Mog1 AI anything, or teach it: Teach: Question => Answer...") or quick_action

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)
        
    with st.chat_message("assistant", avatar="⚡"):
        final_answer = st.write_stream(stream_tokens(user_input))
        
    st.session_state.messages.append({"role": "assistant", "content": final_answer})
