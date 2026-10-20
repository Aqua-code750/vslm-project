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
    page_title="Mog1 AI — Neural Conversational Assistant",
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
        font-size: 1.15rem;
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
    """
    Safely loads or initializes the neural engine without crashing on Streamlit Cloud
    or when encountering Python version unpickling differences.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_path = 'vslm_checkpoint_best.pt' if os.path.exists('vslm_checkpoint_best.pt') else 'vslm_checkpoint.pt'
    
    num_params = 464_000_000  # Default ~0.5B parameters for Qwen
    val_loss = 0.0774
    val_ppl = 1.08
    model = None
    tokenizer = None

    if os.path.exists(ckpt_path):
        try:
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            config = checkpoint.get('config', Mog1Config())
            tokenizer = SubwordTokenizer(
                stoi=checkpoint.get('stoi'),
                itos=checkpoint.get('itos')
            )
            model = Mog1(config).to(device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            num_params = model.get_num_params()
            val_loss = float(checkpoint.get('val_loss', 0.0774))
            val_ppl = float(checkpoint.get('val_ppl', 1.08))
        except Exception:
            config = Mog1Config()
            tokenizer = SubwordTokenizer()
            model = Mog1(config).to(device)
            model.eval()
            num_params = model.get_num_params()
    else:
        config = Mog1Config()
        tokenizer = SubwordTokenizer()
        model = Mog1(config).to(device)
        model.eval()
        num_params = model.get_num_params()

    return model, tokenizer, device, num_params, val_loss, val_ppl

model, tokenizer, device, param_count, val_loss, val_ppl = get_cached_engine()

with st.sidebar:
    st.markdown('### ⚡ Neural Engine')
    
    engine_choice = st.selectbox(
        'Model Weights',
        [
            'Qwen 1.5 0.5B Chat (Fast & Fluid)',
            'Qwen 1.5 1.8B Chat (Deeper Reasoning)',
            'Mog1 VSLM (3.3M Scratch Checkpoint)'
        ],
        index=0
    )
    
    # Calculate displayed params based on selected weights
    if '1.8B' in engine_choice:
        disp_params = "1,836,000,000 (~1.8B)"
    elif '0.5B' in engine_choice:
        disp_params = "464,000,000 (~0.5B)"
    else:
        disp_params = f"{param_count:,} (~3.3M)"

    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>Active Weights</div>
        <div class='metric-value'>{disp_params}</div>
    </div>
    <div class='metric-card'>
        <div class='metric-label'>Compute Target</div>
        <div class='metric-value'>{device.upper()} (Optimized)</div>
    </div>
    <div class='metric-card'>
        <div class='metric-label'>Engine Status</div>
        <div class='metric-value' style='color: #4EFA90;'>Ready & Streaming</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    mode = st.selectbox(
        'Reasoning & Interaction Mode',
        [
            '💬 Chat Mode (Qwen 1.5 Conversational)',
            '⚡ Smart Reasoning Mode',
            '🎯 Exact Factual Mode',
            '🎨 Creative Freedom Mode'
        ],
        index=0
    )
    
    max_tokens = st.slider('Max Generated Tokens', 32, 512, 160 if 'Chat' in mode else 64, 16)
    default_temp = 0.7 if 'Chat' in mode else (0.4 if 'Creative' in mode else 0.0)
    temperature = st.slider('Sampling Temperature', 0.0, 1.2, default_temp, 0.05)
    
    st.divider()
    if st.button('🗑️ Clear Conversation', use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Header Display
st.markdown("""
<div class='main-header'>⚡ Mog1 SI — Super Intelligence</div>
<div style='margin-bottom: 15px;'>
    <span class='badge-pill'>🧠 Super Intelligence (SI)</span>
    <span class='badge-pill'>💬 Chat Mode Active</span>
    <span class='badge-pill'>Multi-Turn Context</span>
    <span class='badge-pill'>Live Token Streaming</span>
</div>
""", unsafe_allow_html=True)

if 'messages' not in st.session_state or len(st.session_state.messages) == 0:
    st.session_state.messages = [
        {'role': 'assistant', 'content': "Hey! I'm **Mog1 SI** (Super Intelligence), created by **Holograph Inc.**! Ask me anything, have a deep conversation, request code, or explore complex reasoning."}
    ]

@st.cache_resource(show_spinner=False)
def load_base_knowledge():
    train_path = 'data/train_instructions.json'
    kb = {}
    if os.path.exists(train_path):
        try:
            with open(train_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for d in data:
                    kb[d['instruction'].lower().strip().rstrip('?!.,')] = (d['instruction'], d['response'])
        except Exception:
            pass
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
    try:
        os.makedirs(os.path.dirname(LEARNED_KB_PATH), exist_ok=True)
        list_data = [{"instruction": v[0], "response": v[1]} for v in learned_kb_dict.values()]
        with open(LEARNED_KB_PATH, 'w', encoding='utf-8') as f:
            json.dump(list_data, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

if 'learned_kb' not in st.session_state:
    st.session_state.learned_kb = load_learned_knowledge()

base_knowledge = load_base_knowledge()

def get_full_knowledge_base():
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

def stream_tokens(user_query: str, current_mode: str = "Chat", engine_type: str = "Qwen"):
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
        st.session_state.learned_kb[clean_key] = (learned_q, learned_a)
        
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

    # Determine model target
    target_model = "Qwen/Qwen1.5-1.8B-Chat" if "1.8B" in engine_type else "Qwen/Qwen1.5-0.5B-Chat"

    # 1. Chat Mode: Direct Conversational AI powered by Qwen 1.5
    if "Chat" in current_mode or "Qwen" in engine_type:
        from generate import generate_stream
        last_len = 0
        try:
            hist = st.session_state.get('messages', [])
            for partial in generate_stream(
                user_query,
                history=hist,
                mode=current_mode,
                max_tokens=max_tokens,
                temperature=temperature,
                model_name=target_model
            ):
                chunk = partial[last_len:]
                last_len = len(partial)
                yield chunk
            return
        except Exception as e:
            yield f"*(Chat Engine Notice: {e})*\n\n"

    # 2. Match against Knowledge Base (for reasoning modes with Mog1)
    grounded_resp, match_score = find_grounded_response(user_query)
    if match_score >= 0.18 and grounded_resp is not None:
        words = grounded_resp.split(' ')
        for i, word in enumerate(words):
            suffix = ' ' if i < len(words) - 1 else ''
            yield word + suffix
            time.sleep(0.012)
        return

    # 3. Real-Time Autonomous Live Learning: search dynamic sources and absorb
    live_answer = search_live_knowledge(user_query)
    if live_answer:
        ans_safe, _ = safety.is_safe_prompt(live_answer)
        if ans_safe:
            clean_key = user_query.lower().strip().rstrip('?!.,')
            st.session_state.learned_kb[clean_key] = (user_query, live_answer)
            if st.session_state.get('is_admin', False):
                save_learned_knowledge(st.session_state.learned_kb)
        
        learned_prefix = "🧠 *(Acquired & Added to Knowledge Base)*\n\n"
        full_text = learned_prefix + live_answer
        for word in full_text.split(' '):
            yield word + ' '
            time.sleep(0.012)
        return

    # 4. Neural Generation fallback via Qwen
    from generate import generate_stream
    last_len = 0
    try:
        hist = st.session_state.get('messages', [])
        for partial in generate_stream(
            user_query,
            history=hist,
            mode=current_mode,
            max_tokens=max_tokens,
            temperature=temperature,
            model_name=target_model
        ):
            chunk = partial[last_len:]
            last_len = len(partial)
            yield chunk
        return
    except Exception as e:
        pass

    fallback = (
        "I'm **Mog1 AI**, your conversational coding, math, and learning assistant (VSLM).\n\n"
        "💡 **You can teach me anything new right in the chat!**\n"
        "• Type: `Teach: What is quantum computing? => Quantum computing uses qubits for exponential calculation.`\n"
        "• Or: `Remember that Aqua is the creator of Mog1 AI`\n\n"
        "Or switch to **💬 Chat Mode** in the sidebar to talk with the full Qwen neural engine!"
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
        final_answer = st.write_stream(stream_tokens(user_input, current_mode=mode, engine_type=engine_choice))
        
    st.session_state.messages.append({"role": "assistant", "content": final_answer})
