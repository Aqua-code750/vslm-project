import re
import torch
from torch.utils.data import Dataset

class SubwordTokenizer:
    def __init__(self, text_corpus: str = None, stoi: dict = None, itos: dict = None, use_tiktoken: bool = False):
        self.use_tiktoken = False
        if use_tiktoken:
            try:
                import tiktoken
                self.enc = tiktoken.get_encoding("gpt2")
                self.vocab_size = self.enc.n_vocab
                self.use_tiktoken = True
            except ImportError:
                pass

        if not self.use_tiktoken:
            if stoi is not None and itos is not None:
                self.stoi = stoi
                self.itos = {int(k): v for k, v in itos.items()} if isinstance(list(itos.keys())[0], str) else itos
                self.vocab_size = len(self.stoi)
            else:
                self._build_fallback(text_corpus)

    def _build_fallback(self, text: str):
        if not text:
            text = "User: Hello\nMog1: I am Mog1 AI model."
        tokens = re.findall(r"\w+|[^\w\s]|\n| ", text)
        vocab = sorted(list(set(tokens)))
        if "<unk>" not in vocab:
            vocab.insert(0, "<unk>")
        self.stoi = {tok: i for i, tok in enumerate(vocab)}
        self.itos = {i: tok for i, tok in enumerate(vocab)}
        self.vocab_size = len(vocab)

    def encode(self, text: str) -> list[int]:
        if self.use_tiktoken:
            return self.enc.encode(text, allowed_special="all")
        else:
            tokens = re.findall(r"\w+|[^\w\s]|\n| ", text)
            unk_id = self.stoi.get("<unk>", 0)
            return [self.stoi.get(t, unk_id) for t in tokens]

    def decode(self, ids: list[int]) -> str:
        if self.use_tiktoken:
            return self.enc.decode(ids)
        else:
            return "".join([self.itos.get(i, "") for i in ids])


class TextDataset(Dataset):
    def __init__(self, data: torch.Tensor, block_size: int):
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
