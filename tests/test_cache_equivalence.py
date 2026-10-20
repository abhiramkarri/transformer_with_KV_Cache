import torch

from kvcache import GPT, GPTConfig, generate


def make_model():
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=100, block_size=128, n_layer=2, n_head=4, n_embd=64)
    return GPT(cfg).eval()


def test_cached_logits_match_full_forward():
    model = make_model()
    idx = torch.randint(0, 100, (2, 20))
    with torch.no_grad():
        full, _ = model(idx)
        # Prefill the first 12 tokens, then feed the remaining 8 one at a time.
        _, past = model(idx[:, :12])
        steps = []
        for t in range(12, 20):
            logits, past = model(idx[:, t : t + 1], past_kvs=past)
            steps.append(logits)
        cached = torch.cat(steps, dim=1)
    assert torch.allclose(full[:, 12:], cached, atol=1e-4)


def test_multi_token_chunk_with_cache():
    model = make_model()
    idx = torch.randint(0, 100, (1, 20))
    with torch.no_grad():
        full, _ = model(idx)
        _, past = model(idx[:, :10])
        chunk, _ = model(idx[:, 10:], past_kvs=past)  # 10 new tokens after a cached prefix
    assert torch.allclose(full[:, 10:], chunk, atol=1e-4)


def test_generate_same_with_and_without_cache():
    model = make_model()
    prompt = torch.randint(0, 100, (2, 8))
    a = generate(model, prompt, max_new_tokens=16, use_cache=True)
    b = generate(model, prompt, max_new_tokens=16, use_cache=False)
    assert torch.equal(a, b)
