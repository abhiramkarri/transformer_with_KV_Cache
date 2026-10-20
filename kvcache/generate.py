import torch

from .model import KVCache


@torch.no_grad()
def generate(model, idx, max_new_tokens, use_cache=True, static_cache=False):
    """Greedy decoding. idx: (B, T) prompt token ids. Returns (B, T + max_new_tokens).

    use_cache=False:                  recompute the whole sequence every step (baseline)
    use_cache=True:                   KV cache grown with torch.cat each step
    use_cache=True, static_cache=True: preallocated KV cache, written in place
    """
    model.eval()

    if not use_cache:
        # Baseline: re-run the whole sequence through the model for every new token.
        for _ in range(max_new_tokens):
            logits, _ = model(idx, last_only=True)
            idx = torch.cat([idx, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
        return idx

    cache = None
    past = None
    if static_cache:
        dtype = next(model.parameters()).dtype
        cache = KVCache(model.cfg, idx.size(0), idx.size(1) + max_new_tokens, idx.device, dtype)

    # Prefill: process the whole prompt once and keep every layer's K/V.
    logits, past = model(idx, cache=cache, last_only=True)
    next_tok = logits[:, -1].argmax(-1, keepdim=True)
    out = [idx, next_tok]

    # Decode: feed only the newest token; attention reads K/V from the cache.
    for _ in range(max_new_tokens - 1):
        logits, past = model(next_tok, past_kvs=past, cache=cache, last_only=True)
        next_tok = logits[:, -1].argmax(-1, keepdim=True)
        out.append(next_tok)
    return torch.cat(out, dim=1)
