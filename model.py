import math
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

class Mog1Config:
    def __init__(
        self,
        vocab_size: int = 2048,
        n_embd: int = 288,
        n_head: int = 6,
        n_layer: int = 6,
        d_ffn: int = 768,
        block_size: int = 256,
        dropout: float = 0.0,
        rope_theta: float = 10000.0,
        tie_weights: bool = True
    ):
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.n_layer = n_layer
        self.d_ffn = d_ffn
        self.block_size = block_size
        self.dropout = dropout
        self.rope_theta = rope_theta
        self.tie_weights = tie_weights

    def to_dict(self):
        return {
            "vocab_size": self.vocab_size,
            "n_embd": self.n_embd,
            "n_head": self.n_head,
            "head_dim": self.head_dim,
            "n_layer": self.n_layer,
            "d_ffn": self.d_ffn,
            "block_size": self.block_size,
            "dropout": self.dropout,
            "rope_theta": self.rope_theta,
            "tie_weights": self.tie_weights
        }


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm) for faster and stable pre-norm."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_normed


def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precomputes complex rotary frequency tensors for RoPE positional encoding."""
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    freq_indices = torch.arange(0, head_dim, 2).float() / head_dim
    freqs = 1.0 / (theta ** freq_indices)  # shape: [head_dim // 2]
    
    positions = torch.arange(max_seq_len).float()  # shape: [max_seq_len]
    angles = torch.outer(positions, freqs)  # shape: [max_seq_len, head_dim // 2]
    
    cos = torch.cos(angles).repeat_interleave(2, dim=-1)  # shape: [max_seq_len, head_dim]
    sin = torch.sin(angles).repeat_interleave(2, dim=-1)  # shape: [max_seq_len, head_dim]
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0) -> torch.Tensor:
    """
    Applies Rotary Position Embedding (RoPE) to tensor x.
    x shape: [batch_size, n_heads, seq_len, head_dim]
    """
    B, H, T, D = x.shape
    cos_slice = cos[offset:offset + T, :].unsqueeze(0).unsqueeze(1).to(x.device, x.dtype)
    sin_slice = sin[offset:offset + T, :].unsqueeze(0).unsqueeze(1).to(x.device, x.dtype)

    x_rotated = torch.empty_like(x)
    x_rotated[..., 0::2] = -x[..., 1::2]
    x_rotated[..., 1::2] = x[..., 0::2]

    return (x * cos_slice) + (x_rotated * sin_slice)


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit (SwiGLU) Feed-Forward Network.
    Formula: SwiGLU(x) = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down
    """
    def __init__(self, n_embd: int, d_ffn: int, dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(n_embd, d_ffn, bias=False)
        self.w_up = nn.Linear(n_embd, d_ffn, bias=False)
        self.w_down = nn.Linear(d_ffn, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.dropout(self.w_down(gate * up))


class Attention(nn.Module):
    """
    Multi-Head Causal Self-Attention with RoPE and dynamic Key-Value (KV) Caching.
    Uses PyTorch's optimized scaled dot-product attention.
    """
    def __init__(self, config: Mog1Config):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_embd = config.n_embd
        self.dropout_p = config.dropout

        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # [B, H, T, D]

        offset = kv_cache[0].shape[2] if kv_cache is not None else 0
        q = apply_rope(q, cos, sin, offset=offset)
        k = apply_rope(k, cos, sin, offset=offset)

        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)

        new_kv_cache = (k, v) if use_cache else None

        is_causal = (T > 1)
        
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.out_proj(attn_out))
        return out, new_kv_cache


class TransformerBlock(nn.Module):
    """Pre-RMSNorm Transformer Decoder Block with RoPE Attention and SwiGLU MLP."""
    def __init__(self, config: Mog1Config):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd)
        self.attn = Attention(config)
        self.mlp_norm = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config.n_embd, config.d_ffn, dropout=config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        normed_x = self.attn_norm(x)
        attn_out, new_kv = self.attn(normed_x, cos, sin, kv_cache=kv_cache, use_cache=use_cache)
        x = x + attn_out

        x = x + self.mlp(self.mlp_norm(x))
        return x, new_kv


class Mog1(nn.Module):
    """
    Mog1 Small Language Model (VSLM).
    - Causal Decoder Transformer
    - Rotary Position Embedding (RoPE)
    - Root Mean Square Normalization (RMSNorm)
    - Swish-Gated Linear Units (SwiGLU)
    - Tied Token Embeddings & Output Projection
    - KV Cache Autoregressive Generation
    """
    def __init__(self, config: Mog1Config):
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share input token embeddings with LM output projection
        if config.tie_weights:
            self.lm_head.weight = self.tok_embeddings.weight

        # Precompute RoPE frequencies
        cos, sin = precompute_rope_freqs(config.head_dim, config.block_size * 2, theta=config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Returns the total number of trainable parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding and not self.config.tie_weights:
            n_params -= self.tok_embeddings.weight.numel()
        return n_params

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        B, T = idx.shape
        x = self.tok_embeddings(idx)

        new_kv_caches = [] if use_cache else None

        for i, block in enumerate(self.blocks):
            layer_kv = kv_caches[i] if kv_caches is not None else None
            x, new_kv = block(x, self.rope_cos, self.rope_sin, kv_cache=layer_kv, use_cache=use_cache)
            if use_cache:
                new_kv_caches.append(new_kv)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100
            )

        return logits, loss, new_kv_caches

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.7,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.15,
        stop_token_ids: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Fast autoregressive text generation using KV-Caching.
        """
        self.eval()
        B, T = idx.shape
        generated = idx.clone()

        logits, _, kv_caches = self(idx[:, -self.config.block_size:], use_cache=True)
        next_token_logits = logits[:, -1, :]

        for _ in range(max_new_tokens):
            logits = next_token_logits

            if repetition_penalty != 1.0:
                for b in range(B):
                    for prev_tok in set(generated[b].tolist()):
                        if logits[b, prev_tok] < 0:
                            logits[b, prev_tok] *= repetition_penalty
                        else:
                            logits[b, prev_tok] /= repetition_penalty

            if temperature > 0:
                logits = logits / temperature
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                if stop_token_ids and next_token.item() in stop_token_ids:
                    break
                generated = torch.cat([generated, next_token], dim=1)
                logits, _, kv_caches = self(next_token, kv_caches=kv_caches, use_cache=True)
                next_token_logits = logits[:, -1, :]
                continue

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                for b in range(B):
                    indices_to_remove = sorted_indices[b, sorted_indices_to_remove[b]]
                    logits[b, indices_to_remove] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if stop_token_ids and next_token.item() in stop_token_ids:
                break

            generated = torch.cat([generated, next_token], dim=1)
            logits, _, kv_caches = self(next_token, kv_caches=kv_caches, use_cache=True)
            next_token_logits = logits[:, -1, :]

        return generated