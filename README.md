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
- [x] Phase 2: preallocated static cache (avoid `torch.cat` copies), measure memory
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

Same setup as above (Colab T4, fp32, prompt length 32). Tokens/sec, and extra peak GPU memory (MiB) beyond the model weights.

**Batch 32**

| new tokens | no cache | cat cache | static cache | static / cat | cat MiB | static MiB |
|---|---|---|---|---|---|---|
| 32 | 1,811 | 11,023 | 10,782 | 0.98x | 104 | 73 |
| 128 | 927 | 9,517 | 11,125 | 1.17x | 247 | 145 |
| 256 | 550 | 7,221 | 9,885 | 1.37x | 438 | 241 |
| 512 | 276 | 4,430 | 6,733 | 1.52x | 822 | 433 |

**Batch 8**

| new tokens | no cache | cat cache | static cache | static / cat | cat MiB | static MiB |
|---|---|---|---|---|---|---|
| 32 | 1,174 | 2,875 | 2,426 | 0.84x | 25 | 18 |
| 128 | 699 | 2,913 | 2,670 | 0.92x | 63 | 36 |
| 256 | 415 | 2,810 | 2,555 | 0.91x | 111 | 62 |
| 512 | 227 | 2,432 | 2,698 | 1.11x | 207 | 110 |

### Findings

1. **The cache size formula checks out.** Batch 32 at 544 total tokens is 24,576 B/token x 17,408 tokens = 408 MiB. The static cache's measured peak is 433 MiB (cache plus about 25 MiB of activations).
2. **`torch.cat` costs about 2x the cache memory.** The cat cache peaks at 822 MiB, about twice the cache, because old and new copies coexist while growing.
3. **The Phase 1 hypothesis is partly confirmed.** At batch 32 and 512 tokens the static cache is 1.52x faster than cat, and throughput falls 41% from its peak as sequences grow, versus 60% for cat. Removing the copies recovers a good part of the loss, but the remaining decline is attention reading the whole cache each step, which is inherent to decoding (memory-bandwidth-bound).
4. **At small batch and short sequences the static cache is not faster, and is often slightly slower** (0.84x to 0.95x at batch 8). There the step is overhead-bound, and in-place writes plus strided views add a little work without saving much traffic. Hypothesis only; not isolated yet.

### Caveats

- Timings come from single runs on a shared Colab T4. Run-to-run variance is real: the cat cache at batch 32 and 32 tokens measured 7.5k tok/s in the Phase 1 run and 11.0k here. Treat differences under about 10-15% as noise; the batch 8 gaps are in that range. Next step: repeat each measurement and report the median.
- The no-cache baseline still builds and holds the per-layer K/V tensors during the forward pass, so its peak-memory numbers are inflated. Only compare the cat and static memory columns.
