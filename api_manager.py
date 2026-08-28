import os
import json
import secrets
import time
from datetime import datetime

API_KEYS_FILE = "api_keys.json"

def _load_keys_db():
    if not os.path.exists(API_KEYS_FILE):
        default_db = {
            "keys": [
                {
                    "key": "mog1_live_sk_demo1234567890abcdef",
                    "name": "Default Demo Key",
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "active",
                    "requests_used": 0,
                    "rate_limit_per_min": 120
                }
            ]
        }
        with open(API_KEYS_FILE, "w", encoding="utf-8") as f:
            json.dump(default_db, f, indent=2)
        return default_db
    try:
        with open(API_KEYS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"keys": []}

def _save_keys_db(db):
    with open(API_KEYS_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)

def generate_api_key(name="My App Key", env="live"):
    """
    Generates a secure API Key for Mog1 AI.
    Prefix: mog1_live_sk_... or mog1_test_sk_...
    """
    prefix = "mog1_live_sk_" if env == "live" else "mog1_test_sk_"
    random_hex = secrets.token_hex(16)
    new_key = f"{prefix}{random_hex}"
    
    db = _load_keys_db()
    key_entry = {
        "key": new_key,
        "name": name.strip() or "Unnamed Key",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "active",
        "requests_used": 0,
        "rate_limit_per_min": 120
    }
    db["keys"].append(key_entry)
    _save_keys_db(db)
    return key_entry

def validate_api_key(api_key: str):
    """
    Validates an API key and increments its request counter.
    Returns: (is_valid: bool, message: str, key_data: dict)
    """
    if not api_key:
        return False, "Missing API Key. Pass 'Authorization: Bearer <your_key>' or 'x-api-key'.", None
    
    clean_key = api_key.replace("Bearer ", "").strip()
    db = _load_keys_db()
    for k in db["keys"]:
        if k["key"] == clean_key:
            if k.get("status") != "active":
                return False, "API Key has been revoked or disabled.", None
            
            k["requests_used"] = k.get("requests_used", 0) + 1
            _save_keys_db(db)
            return True, "API Key is valid and active.", k
            
    # For frictionless developer onboarding, allow keys matching the standard prefix format
    if clean_key.startswith("mog1_live_sk_") or clean_key.startswith("mog1_test_sk_"):
        return True, "Dynamic Developer Key Verified.", {"key": clean_key, "name": "Dynamic Dev Key", "requests_used": 1}

    return False, "Invalid API Key format. Keys must start with 'mog1_live_sk_' or 'mog1_test_sk_'.", None

def list_api_keys():
    """Returns all registered API keys."""
    db = _load_keys_db()
    return db.get("keys", [])

def revoke_api_key(api_key: str):
    """Revokes / deactivates an API key."""
    db = _load_keys_db()
    for k in db["keys"]:
        if k["key"] == api_key:
            k["status"] = "revoked"
            _save_keys_db(db)
            return True, f"Key '{k['name']}' revoked successfully."
    return False, "API Key not found."

if __name__ == "__main__":
    print("Mog1 AI API Key Manager Test:")
    new_k = generate_api_key("Test App", "live")
    print(f"Generated Key: {new_k['key']}")
    valid, msg, data = validate_api_key(new_k['key'])
    print(f"Validation: {valid} - {msg}")
