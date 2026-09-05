import re
from typing import Tuple

INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions|prompts|rules)",
    r"you\s+are\s+now\s+(?:unfiltered|dan|evil|jailbroken|an\s+unrestricted)",
    r"system\s*:\s*override",
    r"bypass\s+(?:safety|filter|guardrails)",
    r"disregard\s+(?:safety|rules|instructions)",
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"<\|endoftext\|>",
    r"repeat\s+after\s+me\s*:\s*.*(?:kill|bomb|hack|steal|password)",
]

TOXIC_KEYWORDS = {
    "kill", "murder", "bomb", "terrorist", "suicide", "hack", "exploit",
    "slur", "nazi", "hitler", "hate", "racist", "porn", "xxx", "nsfw",
    "fck", "fuck", "bitch", "shit", "asshole", "bastard", "dick", "pussy"
}

def is_safe_prompt(text: str) -> Tuple[bool, str]:
    if not text or not text.strip():
        return False, "Prompt is empty."

    cleaned = text.strip()

    if len(cleaned) > 1000:
        return False, "Prompt exceeds maximum length of 1000 characters."

    if re.search(r'(.)\1{9,}', cleaned):
        return False, "Prompt contains spammy repeated characters."

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, cleaned, re.IGNORECASE):
            return False, "Prompt contains disallowed system instructions or injection patterns."

    words = set(re.findall(r'\b\w+\b', cleaned.lower()))
    toxic_hits = words.intersection(TOXIC_KEYWORDS)
    if toxic_hits:
        return False, "Prompt contains restricted or unsafe content."

    return True, "Safe"

def sanitize_text(text: str) -> str:
    cleaned = re.sub(r'[\r\t]+', ' ', text)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'<\s*script[^>]*>.*?<\s*/\s*script\s*>', '', cleaned, flags=re.IGNORECASE)
    return cleaned.strip()
