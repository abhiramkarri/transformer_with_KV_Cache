# transformer_with_KV_Cache

A small GPT-style transformer written from scratch in PyTorch, with an explicit KV cache, built to understand how LLM inference works and where the time goes.

## Why

Autoregressive decoding generates one token at a time. Without a cache, every step recomputes keys and values for the whole sequence, so cost per token grows with length. With a KV cache, each layer's K/V are stored once and each step only processes the newest token.

## Layout

- `kvcache/model.py`: transformer, `past_kvs` (cat) cache, and preallocated `KVCache`
- `kvcache/generate.py`: greedy decoding: no cache, cat cache, or static cache
- `tests/`: cached and uncached outputs must match
- `benchmark.py`: tokens/sec and peak GPU memory for no cache vs. cat cache vs. static cache
- `notebooks/`: Colab quickstart

## Run

```bash
pip install -r requirements.txt
pytest -q
python benchmark.py
```

## Roadmap

- [x] Phase 1: model, KV cache, equivalence tests, benchmark
- [ ] Phase 2: preallocated static cache (avoid `torch.cat` copies), measure memory. Implemented and tested; benchmark results pending (see Phase 2 section)
- [ ] Phase 3: batched decoding, then continuous batching
- [ ] Phase 4: sampling (temperature, top-k/top-p), fp16, quantization
- [ ] Phase 5: compare against Hugging Face and vLLM on the same model

## Results

Setup: Colab T4, fp32, 6 layers, 8 heads, d_model 512, prompt length 32, greedy decoding. Throughput in tokens/sec (batch × new tokens / time).

**With KV cache**

| new tokens | batch 1 | batch 8 | batch 32 |
|---|---|---|---|
| 32 | 414 | 2,828 | 7,531 |
| 64 | 411 | 2,940 | 10,105 |
| 128 | 401 | 3,057 | 9,011 |
| 256 | 398 | 2,684 | 6,899 |
| 512 | 408 | 2,661 | 4,491 |

**Speedup over no cache (same batch)**

| new tokens | batch 1 | batch 8 | batch 32 |
|---|---|---|---|
| 32 | 1.1x | 2.2x | 4.5x |
| 128 | 1.1x | 3.8x | 10.8x |
| 512 | 1.9x | 9.8x | 18.5x |

## Findings and tradeoffs

1. **Batch 1 is overhead-bound.** Cached throughput is flat at about 400 tok/s regardless of length (about 2.5 ms per step). The model is too small to keep the GPU busy, so Python and kernel-launch overhead dominate, and the cache helps little (1.1x) until sequences get long.
2. **Batching amortizes that overhead.** Batch 8 gives about 7x the throughput of batch 1 and batch 32 peaks at about 25x (10.1k tok/s), for nearly the same per-step latency at short lengths.
3. **The cache speedup grows with batch size and sequence length** (up to 18.5x at batch 32, 512 tokens), because recomputing the full sequence costs more as both grow.
4. **At large batch, cached throughput falls as sequences grow** (10.1k to 4.5k tok/s at batch 32). Working hypothesis: the cache costs 24 KB per token (2 x 6 layers x 512 x 4 bytes), so batch 32 at ~550 tokens holds about 430 MB, and every step both copies it (`torch.cat`) and reads it for attention. Decode becomes memory-bandwidth-bound. Phase 2 tests this by removing the copies with a preallocated cache.


## Phase 2: static (preallocated) cache

Hypothesis from Phase 1: at large batch and long sequences, `torch.cat` copies the whole cache every step, so throughput falls as sequences grow. The static cache allocates `(n_layer, batch, n_head, max_len, head_dim)` once, writes each new K/V in place, and attention reads a view of the filled part.

_Results pending: run `python benchmark.py --batch 32` on a T4 and fill in the table (tok/s and peak MB for no cache, cat, static). If throughput at 512 tokens recovers, the copies were a large part of the cost; whatever gap remains is attention reading the cache, which is memory-bound._
