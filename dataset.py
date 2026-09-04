import re
import json
from typing import List, Dict, Tuple, Optional, Union
import torch
from torch.utils.data import Dataset

SPECIAL_TOKENS = [
    "<|pad|>",
    "<|unk|>",
    "<|bos|>",
    "<|eos|>",
    "<|im_start|>",
    "<|im_end|>"
]

class SubwordTokenizer:
    """
    Subword and Character/Word Level Tokenizer for Mog1.
    Supports ChatML instruction formatting and special token encoding.
    """
    def __init__(
        self,
        text_corpus: Optional[str] = None,
        stoi: Optional[Dict[str, int]] = None,
        itos: Optional[Dict[int, str]] = None,
        vocab_limit: int = 2048
    ):
        if stoi is not None and itos is not None:
            self.stoi = stoi
            self.itos = {int(k): v for k, v in itos.items()}
            self.vocab_size = len(self.stoi)
        else:
            self._build_vocab(text_corpus, vocab_limit=vocab_limit)

        self.pad_token_id = self.stoi["<|pad|>"]
        self.unk_token_id = self.stoi["<|unk|>"]
        self.bos_token_id = self.stoi["<|bos|>"]
        self.eos_token_id = self.stoi["<|eos|>"]
        self.im_start_id = self.stoi["<|im_start|>"]
        self.im_end_id = self.stoi["<|im_end|>"]

    def _build_vocab(self, text: Optional[str], vocab_limit: int = 2048):
        vocab = list(SPECIAL_TOKENS)

        if text:
            # Tokenize words, numbers, code symbols, contractions, spaces, and newlines
            tokens = re.findall(r"<\|[a-z_]+\|>|[a-zA-Z]+|\d+|[^\w\s]|\n| {1,4}", text)
            from collections import Counter
            freq = Counter(tokens)
            # Add most frequent tokens up to vocab_limit
            for tok, _ in freq.most_common(vocab_limit - len(vocab)):
                if tok not in vocab:
                    vocab.append(tok)

        self.stoi = {tok: i for i, tok in enumerate(vocab)}
        self.itos = {i: tok for i, tok in enumerate(vocab)}
        self.vocab_size = len(vocab)

    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        
        # Split including special tokens
        pattern = r"(<\|[a-z_]+\|>|[a-zA-Z]+|\d+|[^\w\s]|\n| {1,4})"
        raw_tokens = re.findall(pattern, text)
        
        ids = []
        for t in raw_tokens:
            if t in self.stoi:
                ids.append(self.stoi[t])
            else:
                # Fallback to character-level decomposition for unseen tokens
                for char in t:
                    ids.append(self.stoi.get(char, self.unk_token_id))
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        tokens = []
        for i in ids:
            tok = self.itos.get(int(i), "")
            if skip_special_tokens and tok in SPECIAL_TOKENS:
                continue
            tokens.append(tok)
        return "".join(tokens)

    def apply_chat_template(self, messages: List[Dict[str, str]], add_generation_prompt: bool = False) -> str:
        """
        Formats conversation turns into ChatML format:
        <|im_start|>system
        You are Mog1 AI.<|im_end|>
        <|im_start|>user
        Hello<|im_end|>
        <|im_start|>assistant
        Hi!<|im_end|>
        """
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"

        if add_generation_prompt:
            formatted += "<|im_start|>assistant\n"

        return formatted


class InstructionSFTDataset(Dataset):
    """
    Supervised Fine-Tuning (SFT) Dataset with Target Loss Masking.
    Masks user and system prompt tokens with label -100 so cross-entropy loss
    is calculated exclusively on assistant response tokens.
    """
    def __init__(
        self,
        samples: List[Dict[str, str]],
        tokenizer: SubwordTokenizer,
        block_size: int = 256
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.data_entries = []
        self._process_samples()

    def _process_samples(self):
        for sample in self.samples:
            system_prompt = sample.get("system", "You are Mog1 AI, a helpful and coherent conversational assistant.")
            user_msg = sample["instruction"]
            assistant_msg = sample["response"]

            # Format user prompt prefix
            prompt_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
            response_text = f"{assistant_msg}<|im_end|>\n"

            prompt_ids = self.tokenizer.encode(prompt_text)
            response_ids = self.tokenizer.encode(response_text)

            input_ids = prompt_ids + response_ids
            # Target labels: mask prompt with -100, keep response labels
            target_ids = ([-100] * len(prompt_ids)) + response_ids

            if len(input_ids) > self.block_size:
                input_ids = input_ids[:self.block_size]
                target_ids = target_ids[:self.block_size]

            # Shift for autoregressive next-token prediction: x is input[:-1], y is targets[1:]
            x = input_ids[:-1]
            y = target_ids[1:]

            # Pad up to block_size - 1
            pad_len = (self.block_size - 1) - len(x)
            if pad_len > 0:
                x = x + [self.tokenizer.pad_token_id] * pad_len
                y = y + [-100] * pad_len

            self.data_entries.append((
                torch.tensor(x, dtype=torch.long),
                torch.tensor(y, dtype=torch.long)
            ))

    def __len__(self):
        return len(self.data_entries)

    def __getitem__(self, idx):
        return self.data_entries[idx]


class PretrainTextDataset(Dataset):
    """
    Standard sequential pretraining dataset with sliding context window.
    """
    def __init__(self, data: torch.Tensor, block_size: int = 256):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return max(1, (len(self.data) - 1) // self.block_size)

    def __getitem__(self, idx):
        start_i = idx * self.block_size
        x = self.data[start_i:start_i + self.block_size]
        y = self.data[start_i + 1:start_i + self.block_size + 1]

        if len(x) < self.block_size:
            pad_len = self.block_size - len(x)
            x = torch.cat([x, torch.zeros(pad_len, dtype=torch.long)])
        if len(y) < self.block_size:
            pad_len = self.block_size - len(y)
            y = torch.cat([y, torch.zeros(pad_len, dtype=torch.long)])

        return x, y
