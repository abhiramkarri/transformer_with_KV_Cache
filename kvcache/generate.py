import torch


@torch.no_grad()
def generate(model, idx, max_new_tokens, use_cache=True):
    """Greedy decoding. idx: (B, T) prompt token ids. Returns (B, T + max_new_tokens)."""
    model.eval()

    if not use_cache:
        # Baseline: re-run the whole sequence through the model for every new token.
        for _ in range(max_new_tokens):
            logits, _ = model(idx, last_only=True)
            idx = torch.cat([idx, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
        return idx

    # Prefill: process the whole prompt once and keep every layer's K/V.
    logits, past = model(idx, last_only=True)
    next_tok = logits[:, -1].argmax(-1, keepdim=True)
    out = [idx, next_tok]

    # Decode: feed only the newest token; attention reads K/V from the cache.
    for _ in range(max_new_tokens - 1):
        logits, past = model(next_tok, past_kvs=past, last_only=True)
        next_tok = logits[:, -1].argmax(-1, keepdim=True)
        out.append(next_tok)
    return torch.cat(out, dim=1)
