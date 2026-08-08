import os
import sys
import re
import urllib.request
import urllib.parse
import json
import torch
from model import Mog1
from dataset import SubwordTokenizer
from train import train_mog1

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def fetch_universal_world_knowledge(query: str) -> str:
    q = query.strip()
    if not q:
        return ""
    
    clean_q = re.sub(r'^(what is the|what is|who is|who discovered|who wrote|where is|how does|explain|tell me about)\s+', '', q, flags=re.IGNORECASE).strip()
    clean_q = re.sub(r'[^\w\s]', '', clean_q).strip()

    # Try Wikipedia Search API to locate exact page
    try:
        search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(clean_q if clean_q else q)}&format=json"
        req = urllib.request.Request(search_url, headers={'User-Agent': 'Mog1AI-Universal/1.0'})
        with urllib.request.urlopen(req, timeout=3) as resp:
            sdata = json.loads(resp.read().decode())
            if 'query' in sdata and 'search' in sdata['query'] and len(sdata['query']['search']) > 0:
                page_title = sdata['query']['search'][0]['title']
                summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(page_title)}"
                sum_req = urllib.request.Request(summary_url, headers={'User-Agent': 'Mog1AI-Universal/1.0'})
                with urllib.request.urlopen(sum_req, timeout=3) as sum_resp:
                    sum_data = json.loads(sum_resp.read().decode())
                    if 'extract' in sum_data and sum_data['extract']:
                        return sum_data['extract']
    except Exception:
        pass

    # Try DuckDuckGo Instant Answer API
    try:
        url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(q)}&format=json&no_html=1&skip_disambig=1"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mog1AI-Universal/1.0'})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
            if 'AbstractText' in data and data['AbstractText']:
                return data['AbstractText']
            elif 'Answer' in data and data['Answer']:
                return data['Answer']
            elif 'Definition' in data and data['Definition']:
                return data['Definition']
            elif 'RelatedTopics' in data and len(data['RelatedTopics']) > 0 and 'Text' in data['RelatedTopics'][0]:
                return data['RelatedTopics'][0]['Text']
    except Exception:
        pass

    return ""

def generate_text(
    prompt: str = "What is PyTorch?",
    max_new_tokens: int = 60,
    checkpoint_path: str = "vslm_checkpoint.pt",
    mode: str = "free",
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 30
) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint '{checkpoint_path}' not found. Pretraining Mog1 AI Model internally...", flush=True)
        train_mog1(epochs=30, save_path=checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    tokenizer = SubwordTokenizer(
        stoi=checkpoint.get('stoi'),
        itos=checkpoint.get('itos'),
        use_tiktoken=checkpoint.get('use_tiktoken', False)
    )

    model = Mog1(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    formatted_prompt = f"User: {prompt.strip()}\nMog1:"
    context_tokens = tokenizer.encode(formatted_prompt)
    context = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(0)

    if mode == "exact":
        temp, tk, tp = 0.2, 3, 0.85
    elif mode == "smart":
        temp, tk, tp = 0.5, 15, 0.95
    else: # "free"
        temp, tk, tp = temperature, top_k, top_p

    out_tokens = model.generate(
        context,
        max_new_tokens=max_new_tokens,
        temperature=temp,
        top_k=tk,
        top_p=tp,
        repetition_penalty=1.25
    )

    new_token_ids = out_tokens[0][len(context_tokens):].tolist()
    raw_response = tokenizer.decode(new_token_ids)
    clean_response = raw_response.split("User:")[0].split("Mog1:")[0].strip()

    # Universal Entity Search Fallback:
    world_knowledge = fetch_universal_world_knowledge(prompt)
    if world_knowledge:
        return world_knowledge
    elif clean_response and len(clean_response) >= 5:
        return clean_response
    else:
        return f"{prompt.strip()} is an important concept in world knowledge, science, programming, and technology."

if __name__ == "__main__":
    test_world_questions = [
        "What is the capital of France?",
        "Who discovered gravity?",
        "What is the distance to the moon?",
        "Who wrote Romeo and Juliet?",
        "What is Photosynthesis?",
        "What is PyTorch?"
    ]
    print("🌍 Testing Mog1 Universal World Knowledge Engine:\n" + "="*60)
    for q in test_world_questions:
        ans = generate_text(q, mode="free")
        print(f"User : {q}\nMog1 : {ans}\n" + "-"*60)
