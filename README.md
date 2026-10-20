# transformer_with_KV_Cache

A small GPT-style transformer written from scratch in PyTorch, with an explicit KV cache, built to understand how LLM inference works and where the time goes.

## Why

Autoregressive decoding generates one token at a time. Without a cache, every step recomputes keys and values for the whole sequence, so cost per token grows with length. With a KV cache, each layer's K/V are stored once and each step only processes the newest token.

## Layout

- `kvcache/model.py`: transformer with `past_kvs` support
- `kvcache/generate.py`: greedy decoding with and without the cache
- `tests/`: cached and uncached outputs must match
- `benchmark.py`: tokens/sec with vs. without cache
- `notebooks/`: Colab quickstart

## Run

```bash
pip install -r requirements.txt
pytest -q
python benchmark.py
```

## Roadmap

- [x] Phase 1: model, KV cache, equivalence tests, benchmark
- [ ] Phase 2: preallocated static cache (avoid `torch.cat` copies), measure memory
- [ ] Phase 3: batched decoding, then continuous batching
- [ ] Phase 4: sampling (temperature, top-k/top-p), fp16, quantization
- [ ] Phase 5: compare against Hugging Face and vLLM on the same model

## Results

_Fill in after running `benchmark.py` on a Colab T4: table, graph, and what surprised you._

## Findings and tradeoffs

_Write these as you go: memory growth of the cache vs. sequence length, why decode is memory-bound, where speedup flattens._
