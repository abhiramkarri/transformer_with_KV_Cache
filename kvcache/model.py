"""A minimal GPT-style decoder-only transformer with an explicit KV cache.

The cache is a list with one (k, v) tuple per layer, each of shape
(batch, n_head, seq_len, head_dim). It grows by concatenation on every step.
(Phase 2 replaces this with a preallocated cache.)
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 512


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)

    def forward(self, x, past_kv=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, -1).transpose(1, 2)
        k = k.view(B, T, self.n_head, -1).transpose(1, 2)
        v = v.view(B, T, self.n_head, -1).transpose(1, 2)

        past_len = 0
        if past_kv is not None:
            past_k, past_v = past_kv
            past_len = past_k.size(2)
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        present = (k, v)

        if past_len == 0:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif T == 1:
            # One new token may attend to everything cached; no mask needed.
            y = F.scaled_dot_product_attention(q, k, v)
        else:
            # Several new tokens after a cached prefix: query i sees keys <= past_len + i.
            S = k.size(2)
            mask = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(diagonal=S - T)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y), present


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
        )

    def forward(self, x, past_kv=None):
        a, present = self.attn(self.ln1(x), past_kv)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, present


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, past_kvs=None, last_only=False):
        """idx: (B, T) token ids. Returns (logits, presents).

        past_kvs: cache from a previous call, or None. With a cache, idx holds
        only the NEW tokens and positions continue from the cached length.
        last_only: compute logits for the final position only (saves work in decoding).
        """
        B, T = idx.shape
        past_len = past_kvs[0][0].size(2) if past_kvs is not None else 0
        assert past_len + T <= self.cfg.block_size, "sequence exceeds block_size"

        pos = torch.arange(past_len, past_len + T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)

        presents = []
        for i, block in enumerate(self.blocks):
            x, present = block(x, None if past_kvs is None else past_kvs[i])
            presents.append(present)

        x = self.ln_f(x)
        if last_only:
            x = x[:, -1:, :]
        return self.lm_head(x), presents
