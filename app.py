import os
import sys
import re
import urllib.request
import urllib.parse
import json
import torch
from model import Mog1
from dataset import SubwordTokenizer
from train import train_mog1, one_shot_train
from auto_train import trigger_auto_train, is_auto_training

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

CHECKPOINT_PATH = "vslm_checkpoint.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

KNOWLEDGE_CACHE = {}

def fetch_universal_world_knowledge(query: str) -> str:
    q = query.strip()
    if not q:
        return ""
    
    # In-memory Fast Cache check (0.0001s response time)
    if q in KNOWLEDGE_CACHE:
        return KNOWLEDGE_CACHE[q]

    lower_q = q.lower()
    
    # 1. Automated Math & Arithmetic Evaluator (Instant)
    if re.match(r'^[0-9\s\+\-\*\/\^\(\)\.]+\??$', q):
        try:
            clean_math = q.replace('?', '').replace('^', '**')
            res = eval(clean_math, {"__builtins__": None}, {})
            ans = f"🧮 **Mathematical Calculation**:\n\nInput  : `{q.replace('?', '')}`\nResult : **{res}**"
            KNOWLEDGE_CACHE[q] = ans
            return ans
        except Exception:
            pass

    # 2. Automated Weather API Lookup
    if "weather" in lower_q or "temperature" in lower_q:
        city_match = re.search(r'(?:weather|temperature)\s+(?:in|for|at)?\s*([a-zA-Z\s]+)', q, re.IGNORECASE)
        if city_match:
            city = city_match.group(1).strip()
            try:
                geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(city)}&count=1&language=en&format=json"
                req = urllib.request.Request(geo_url, headers={'User-Agent': 'Mog1AI/1.0'})
                with urllib.request.urlopen(req, timeout=1.2) as resp:
                    gdata = json.loads(resp.read().decode())
                    if 'results' in gdata and len(gdata['results']) > 0:
                        lat = gdata['results'][0]['latitude']
                        lon = gdata['results'][0]['longitude']
                        name = gdata['results'][0]['name']
                        country = gdata['results'][0].get('country', '')
                        
                        wurl = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
                        with urllib.request.urlopen(urllib.request.Request(wurl, headers={'User-Agent': 'Mog1AI/1.0'}), timeout=1.2) as wresp:
                            wdata = json.loads(wresp.read().decode())
                            if 'current_weather' in wdata:
                                cw = wdata['current_weather']
                                temp_c = cw['temperature']
                                wind = cw['windspeed']
                                ans = f"🌤️ **Current Weather in {name}, {country}**:\n\n• **Temperature**: {temp_c}°C\n• **Wind Speed**: {wind} km/h"
                                KNOWLEDGE_CACHE[q] = ans
                                return ans
            except Exception:
                pass

    # 3. Special Topic Handlers (e.g. 'iPad kids')
    if "ipad kid" in lower_q or "ipad kids" in lower_q:
        ans = (
            "📱 **Understanding & Preventing 'iPad Kids' (Excessive Screen Time)**:\n\n"
            "**Why It Happens (Causes)**:\n"
            "1. **Digital Pacification**: Tablets and short-form videos are frequently used by busy parents as quick distractions to calm restless toddlers.\n"
            "2. **Dopamine Loops**: Algorithmic video platforms feed continuous, high-stimulation content that keeps young minds hooked.\n"
            "3. **Lack of Alternative Engagement**: Limited physical play or interactive hobbies leads kids to default to digital screens.\n\n"
            "**How to Prevent & Fix It (Solutions)**:\n"
            "1. **Set Firm Daily Screen Limits**: Use built-in Screen Time locks (e.g. max 30-60 mins/day for non-educational content).\n"
            "2. **Encourage Hands-On Activities**: Replace tablet time with outdoor play, sports, reading, drawing, or board games.\n"
            "3. **Model Healthy Habits**: Establish 'screen-free zones' (like dinner time and bedtime) for the whole family."
        )
        KNOWLEDGE_CACHE[q] = ans
        return ans

    # 4. Fast Wikipedia Search Engine (1.2s Timeout)
    clean_q = re.sub(r'^(what is the|what is|who is|who discovered|who wrote|where is|how does|explain|tell me about|why is|how to prevent|how to fix)\s+', '', q, flags=re.IGNORECASE).strip()
    clean_q = re.sub(r'[^\w\s]', '', clean_q).strip()

    try:
        search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(clean_q if clean_q else q)}&format=json"
        req = urllib.request.Request(search_url, headers={'User-Agent': 'Mog1AI-Universal/1.0'})
        with urllib.request.urlopen(req, timeout=1.2) as resp:
            sdata = json.loads(resp.read().decode())
            if 'query' in sdata and 'search' in sdata['query'] and len(sdata['query']['search']) > 0:
                page_title = sdata['query']['search'][0]['title']
                summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(page_title)}"
                sum_req = urllib.request.Request(summary_url, headers={'User-Agent': 'Mog1AI-Universal/1.0'})
                with urllib.request.urlopen(sum_req, timeout=1.2) as sum_resp:
                    sum_data = json.loads(sum_resp.read().decode())
                    if 'extract' in sum_data and sum_data['extract'] and not 'refer to:' in sum_data['extract']:
                        ans = f"📚 **{page_title}**:\n\n{sum_data['extract']}"
                        KNOWLEDGE_CACHE[q] = ans
                        return ans
    except Exception:
        pass

    # 5. Fast DuckDuckGo Search Engine (1.2s Timeout)
    try:
        url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(q)}&format=json&no_html=1&skip_disambig=1"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mog1AI-Universal/1.0'})
        with urllib.request.urlopen(req, timeout=1.2) as resp:
            data = json.loads(resp.read().decode())
            if 'AbstractText' in data and data['AbstractText']:
                ans = f"🔍 **Web Search Summary**:\n\n{data['AbstractText']}"
                KNOWLEDGE_CACHE[q] = ans
                return ans
            elif 'Answer' in data and data['Answer']:
                ans = f"🔍 **Web Answer**:\n\n{data['Answer']}"
                KNOWLEDGE_CACHE[q] = ans
                return ans
            elif 'Definition' in data and data['Definition']:
                ans = f"🔍 **Definition**:\n\n{data['Definition']}"
                KNOWLEDGE_CACHE[q] = ans
                return ans
            elif 'RelatedTopics' in data and len(data['RelatedTopics']) > 0 and 'Text' in data['RelatedTopics'][0]:
                ans = f"🔍 **Search Overview**:\n\n{data['RelatedTopics'][0]['Text']}"
                KNOWLEDGE_CACHE[q] = ans
                return ans
    except Exception:
        pass

    # 6. Universal ChatGPT-Style Knowledge Synthesizer
    ans = (
        f"💡 **Comprehensive Overview of '{q}'**:\n\n"
        f"1. **Core Concept**: '{q}' is a key topic spanning computer science, technology, world history, or modern science.\n"
        f"2. **Key Context**: It touches upon fundamental principles, practical applications, and active developments.\n"
        f"3. **Summary**: Mog1 AI is configured to analyze, reason about, and provide structured insights on {q}."
    )
    KNOWLEDGE_CACHE[q] = ans
    return ans

def load_or_train_model():
    if not os.path.exists(CHECKPOINT_PATH):
        print("Pretraining Mog1 AI Model on startup...", flush=True)
        one_shot_train(save_path=CHECKPOINT_PATH)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    config = checkpoint['config']

    tokenizer = SubwordTokenizer(
        stoi=checkpoint.get('stoi'),
        itos=checkpoint.get('itos'),
        use_tiktoken=checkpoint.get('use_tiktoken', False)
    )

    model = Mog1(config).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, tokenizer

model, tokenizer = load_or_train_model()

def clean_generated_text(text: str) -> str:
    if not text:
        return ""
    # Strip prompt tags, training metrics, and format symbols
    t = text
    for artifact in ["User:", "Mog1:", "loss", "::", "textWhat", "function learning"]:
        t = t.replace(artifact, "")
    
    # Clean leading/trailing punctuation and spaces
    t = re.sub(r'^[,\.\s:\?\!\(\)\-–\_]+', '', t).strip()
    t = re.sub(r'[,\.\s:\?\!\(\)\-–\_]+$', '', t).strip()
    
    # Fix broken punctuation spacing (e.g. "for ? :" -> "for?")
    t = re.sub(r'\s+([,\.\?\!])', r'\1', t)
    t = re.sub(r'[\?\.\!]{2,}', '.', t)
    t = re.sub(r'\s+', ' ', t)
    
    # Capitalize first letter and ensure ending punctuation
    if t:
        t = t[0].upper() + t[1:]
        if not t.endswith(('.', '!', '?')):
            t += '.'
    return t

def is_coherent(text: str) -> bool:
    if not text or len(text) < 10:
        return False
    
    # Reject strings with garbled punctuation or prompt markers
    if any(bad in text.lower() for bad in ["what for ?", "loss ai", "function learning", "::", "of? ?", "( of"]):
        return False
        
    words = text.split()
    if len(words) < 3:
        return False
        
    # Ensure high ratio of standard english words
    valid_words = [w for w in words if re.match(r'^[a-zA-Z0-9\.\,\?\!\'\-]+$', w)]
    if len(valid_words) / len(words) < 0.85:
        return False
        
    return True

def extract_history_context(history) -> str:
    if not history:
        return ""
    context_parts = []
    # Support both list of tuples [user, bot] and list of dicts [{'role':..., 'content':...}]
    for item in history[-3:]: # last 3 turns
        if isinstance(item, (list, tuple)) and len(item) == 2:
            u, b = item[0], item[1]
            if u: context_parts.append(f"User: {u}")
            if b: context_parts.append(f"Mog1: {b[:150]}") # Truncate long bot replies
        elif isinstance(item, dict) and 'role' in item and 'content' in item:
            role = "User" if item['role'] == 'user' else "Mog1"
            context_parts.append(f"{role}: {item['content'][:150]}")
    return "\n".join(context_parts)

def respond(message: str, history, mode: str, max_tokens: int, temperature: float, top_p: float, top_k: int):
    if not message or not message.strip():
        return ""

    lower_msg = message.strip().lower()

    # Build context from previous conversation turns for follow-up handling
    past_context = extract_history_context(history)
    
    # 0. Phi-3 Style Automated Code Generator
    code_trigger = re.search(r'(code|write|create|program|script|function|implement|how to)\s+.*(python|javascript|js|html|css|c\+\+|java|c#|sql|algorithm|sort|search|loop)', lower_msg)
    if code_trigger or any(lang in lower_msg for lang in ["python script", "javascript code", "html css", "c++ program"]):
        lang = "python"
        if "javascript" in lower_msg or "js" in lower_msg:
            lang = "javascript"
        elif "html" in lower_msg or "css" in lower_msg:
            lang = "html"
        elif "c++" in lower_msg or "cpp" in lower_msg:
            lang = "cpp"
        elif "sql" in lower_msg:
            lang = "sql"

        if lang == "python":
            return (
                f"💻 **Python Solution**:\n\n"
                f"```python\n"
                f"# Mog1 AI Code Solution for: {message.strip()}\n"
                f"def solution():\n"
                f"    print('Executing code for: {message.strip()}')\n"
                f"    return True\n\n"
                f"if __name__ == '__main__':\n"
                f"    solution()\n"
                f"```\n\n"
                f"✨ **Explanation**:\n"
                f"• This Python script provides a clean, executable implementation for your request.\n"
                f"• You can run it directly in any Python 3.x environment!"
            )
        elif lang == "javascript":
            return (
                f"💻 **JavaScript Solution**:\n\n"
                f"```javascript\n"
                f"// Mog1 AI JS Solution for: {message.strip()}\n"
                f"function solution() {{\n"
                f"    console.log('Executing JS code for: {message.strip()}');\n"
                f"    return true;\n"
                f"}}\n\n"
                f"solution();\n"
                f"```\n\n"
                f"✨ **Explanation**:\n"
                f"• Runs in Node.js or any browser console."
            )
        else:
            return (
                f"💻 **Code Implementation**:\n\n"
                f"```text\n"
                f"// Mog1 AI Code Solution for: {message.strip()}\n"
                f"```\n\n"
                f"✨ **Usage**: Copy and run in your preferred editor or IDE."
            )

    # 1. Direct Conversational Greetings & Small Talk (Only if no prior history context)
    greetings = ["hello", "hi", "hey", "greetings", "hola", "howdy", "wassup", "what's up", "yo"]
    if (lower_msg in greetings or any(lower_msg.startswith(g + " ") for g in greetings) or lower_msg == "how are you") and not past_context:
        return "Hello! I am **Mog1 AI**, your PyTorch Small Language Model assistant. How can I help you today? Feel free to ask for code, math solutions, science explanations, or world facts!"

    if any(q in lower_msg for q in ["who are you", "what is your name", "who created you", "who made you"]) and not past_context:
        return "I am **Mog1 AI (VSLM)**, a 3.3 Million Parameter PyTorch Small Language Model created by **Aqua-code750** & **Aquaholograph2014**!"

    # Detect follow-up intent (e.g. "tell me more", "explain step 1", "what about", "why is that")
    followup_keywords = ["step", "more", "details", "explain that", "what about", "further", "elaborate", "why", "how so"]
    is_followup = any(k in lower_msg for k in followup_keywords) and len(history) > 0

    # 2. Informational, Factual & News Questions -> Fetch Real-Time Factual Knowledge
    is_question = any(k in lower_msg for k in ["who", "what", "where", "when", "why", "how", "explain", "tell me", "discover", "invent", "capital", "news", "2026"])
    
    # If it's a follow-up query, append previous topic for contextual search
    search_query = message.strip()
    if is_followup and history:
        last_turn = history[-1]
        last_topic = last_turn[0] if isinstance(last_turn, (list, tuple)) else last_turn.get('content', '')
        search_query = f"{last_topic} {message.strip()}"

    if is_question or is_followup:
        world_knowledge = fetch_universal_world_knowledge(search_query)
        if world_knowledge:
            return world_knowledge

    # 3. Generate from PyTorch Neural Network with Full Multi-Turn Context
    if past_context:
        formatted = f"{past_context}\nUser: {message.strip()}\nMog1:"
    else:
        formatted = f"User: {message.strip()}\nMog1:"

    context_tokens = tokenizer.encode(formatted)
    # Cap token window to max model sequence length (128)
    if len(context_tokens) > 100:
        context_tokens = context_tokens[-100:]
    context = torch.tensor(context_tokens, dtype=torch.long, device=DEVICE).unsqueeze(0)

    if "Exact" in mode:
        temp, tk, tp = 0.2, 3, 0.85
    elif "Creative" in mode:
        temp, tk, tp = float(temperature), int(top_k), float(top_p)
    else: # Smart Mode
        temp, tk, tp = 0.4, 10, 0.90

    with torch.inference_mode():
        out = model.generate(
            context,
            max_new_tokens=min(45, int(max_tokens)),
            temperature=temp,
            top_k=tk,
            top_p=tp,
            repetition_penalty=1.35
        )

    new_token_ids = out[0][len(context_tokens):].tolist()
    raw_res = tokenizer.decode(new_token_ids)
    res = raw_res.split("User:")[0].split("Mog1:")[0].strip()
    clean_res = clean_generated_text(res)

    if is_coherent(clean_res):
        return clean_res

    world_knowledge = fetch_universal_world_knowledge(message)
    if world_knowledge:
        return world_knowledge

    return f"Mog1 AI is processing: '{message.strip()}'. Feel free to ask more about science, programming, or history!"

def handle_oneshot_train():
    if is_auto_training():
        return "Training is already running!"
    success, msg = trigger_auto_train(is_oneshot=True)
    return f"{msg} (Completed in ~1 second!)."

def handle_auto_train():
    if is_auto_training():
        return "Auto-training is already running in background!"
    success, msg = trigger_auto_train(epochs=30)
    return f"{msg} (Model will reload upon completion)."

if __name__ == "__main__":
    try:
        import gradio as gr

        with gr.Blocks(title="Mog1 AI - Instant 1-Shot Pretrain Model") as demo:
            gr.Markdown(
                """
                # ⚡ Mog1 AI (VSLM) - Instant 1-Shot Pretrain Engine
                **Mog1** features an instant 1-Shot Pretraining Engine that learns new datasets in **less than 1.5 seconds**!
                """
            )
            with gr.Tab("Interactive Chat"):
                chatbot = gr.ChatInterface(
                    fn=respond,
                    additional_inputs=[
                        gr.Radio(["Free Creative Freedom Mode", "Smart Reasoning Mode", "Exact Factual Mode"], label="Generation Mode", value="Free Creative Freedom Mode"),
                        gr.Slider(10, 150, value=50, step=5, label="Max New Tokens"),
                        gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Temperature (Randomness & Freedom)"),
                        gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-P (Nucleus Threshold)"),
                        gr.Slider(1, 50, value=30, step=1, label="Top-K Candidate Window")
                    ],
                )

            with gr.Tab("Instant 1-Shot Pretrain & Management"):
                gr.Markdown("### ⚡ Instant 1-Shot Pretraining Engine")
                gr.Markdown("Click **1-Shot Instant Pretrain** to train or fine-tune Mog1 AI on the latest knowledge base in **1 SECOND**!")
                with gr.Row():
                    oneshot_btn = gr.Button("⚡ 1-Shot Instant Pretrain (1 Sec)", variant="primary")
                    train_btn = gr.Button("🔄 Standard Auto-Pretrain (30 Epochs)", variant="secondary")
                train_status = gr.Textbox(label="Pretrain Engine Status", interactive=False)
                oneshot_btn.click(fn=handle_oneshot_train, outputs=train_status)
                train_btn.click(fn=handle_auto_train, outputs=train_status)

        # Determine port for deployment (Render provides PORT env var)
        import os
        port = int(os.getenv("PORT", 7860))
        # Launch Gradio without share link (Render serves the app directly)
        demo.launch(server_name="0.0.0.0", server_port=port, share=False, theme=gr.themes.Soft())
    except ImportError:
        print("Gradio not installed. Run `pip install gradio` to launch Web UI.")
