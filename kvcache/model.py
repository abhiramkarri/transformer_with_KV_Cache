"""A minimal GPT-style decoder-only transformer with an explicit KV cache.

Two cache styles are supported:

* "cat" cache (`past_kvs`): a list with one (k, v) tuple per layer, each of shape
  (batch, n_head, seq_len, head_dim). It grows by `torch.cat` on every step, which
  allocates and copies the whole cache each time.
* static cache (`cache=KVCache(...)`): one preallocated buffer. New keys/values are
  written in place and attention reads a view of the filled part. No per-step copies.
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


class KVCache:
    """Preallocated KV cache: tensors of shape (n_layer, batch, n_head, max_len, head_dim)."""

    def __init__(self, cfg: GPTConfig, batch_size, max_len, device="cpu", dtype=torch.float32):
        head_dim = cfg.n_embd // cfg.n_head
        shape = (cfg.n_layer, batch_size, cfg.n_head, max_len, head_dim)
        self.k = torch.empty(shape, device=device, dtype=dtype)
        self.v = torch.empty(shape, device=device, dtype=dtype)
        self.max_len = max_len
        self.length = 0  # number of positions already filled

    def update(self, layer, k_new, v_new):
        """Write new K/V for `layer` at the current position; return views of the filled part."""
        T = k_new.size(2)
        start = self.length
        assert start + T <= self.max_len, "KVCache is full"
        self.k[layer, :, :, start : start + T] = k_new
        self.v[layer, :, :, start : start + T] = v_new
        return self.k[layer, :, :, : start + T], self.v[layer, :, :, : start + T]

    def advance(self, T):
        """Call once per forward pass, after every layer has written its K/V."""
        self.length += T

    def nbytes(self):
        return self.k.numel() * self.k.element_size() * 2


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)

    def forward(self, x, past_kv=None, cache=None, layer_idx=0):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, -1).transpose(1, 2)
        k = k.view(B, T, self.n_head, -1).transpose(1, 2)
        v = v.view(B, T, self.n_head, -1).transpose(1, 2)

        past_len = 0
        present = None
        if cache is not None:
            past_len = cache.length
            k, v = cache.update(layer_idx, k, v)
        elif past_kv is not None:
            past_k, past_v = past_kv
            past_len = past_k.size(2)
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            present = (k, v)
        else:
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

    def forward(self, x, past_kv=None, cache=None, layer_idx=0):
        a, present = self.attn(self.ln1(x), past_kv, cache, layer_idx)
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

    def forward(self, idx, past_kvs=None, cache=None, last_only=False):
        """idx: (B, T) token ids. Returns (logits, presents).

        past_kvs: "cat" cache from a previous call, or None.
        cache: a preallocated KVCache (updated in place; `presents` is then None).
        With either cache, idx holds only the NEW tokens and positions continue
        from the cached length.
        last_only: compute logits for the final position only (saves work in decoding).
        """
        assert past_kvs is None or cache is None, "use either past_kvs or cache, not both"
        B, T = idx.shape
        if cache is not None:
            past_len = cache.length
        elif past_kvs is not None:
            past_len = past_kvs[0][0].size(2)
        else:
            past_len = 0
        assert past_len + T <= self.cfg.block_size, "sequence exceeds block_size"

        pos = torch.arange(past_len, past_len + T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)

        presents = [] if cache is None else None
        for i, block in enumerate(self.blocks):
            past = None if past_kvs is None else past_kvs[i]
            x, present = block(x, past, cache, i)
            if presents is not None:
                presents.append(present)
        if cache is not None:
            cache.advance(T)

        x = self.ln_f(x)
        if last_only:
            x = x[:, -1:, :]
        return self.lm_head(x), presents
