"""Benchmark decoding speed and memory: no cache vs. "cat" cache vs. static (preallocated) cache.

Usage: python benchmark.py [--batch 8] [--new-tokens 32 64 128 256 512] [--out results.csv]
"""
import argparse
import csv
import time

import torch

from kvcache import GPT, GPTConfig, generate


def run(fn, device):
    """Return (seconds, extra peak GPU memory in MB beyond what was allocated before the run)."""
    peak_mb = 0.0
    if device == "cuda":
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    fn()
    if device == "cuda":
        torch.cuda.synchronize()
        peak_mb = (torch.cuda.max_memory_allocated() - base) / 2**20
    return time.perf_counter() - t0, peak_mb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--new-tokens", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    p.add_argument("--prompt-len", type=int, default=32)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--out", default="results.csv")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = GPTConfig(vocab_size=32000, block_size=1024, n_layer=6, n_head=8, n_embd=512)
    model = GPT(cfg).to(device).eval()
    prompt = torch.randint(0, cfg.vocab_size, (args.batch, args.prompt_len), device=device)

    variants = {
        "nocache": dict(use_cache=False),
        "cat": dict(use_cache=True),
        "static": dict(use_cache=True, static_cache=True),
    }
    for kw in variants.values():  # warmup
        generate(model, prompt, 8, **kw)

    print(f"device={device} batch={args.batch} prompt_len={args.prompt_len}")
    print(f"{'new':>5} | {'nocache':>9} {'cat':>9} {'static':>9} tok/s | {'static/cat':>10} | "
          f"{'nocache':>8} {'cat':>8} {'static':>8} peak MB")
    rows = []
    for n in args.new_tokens:
        res = {}
        for name, kw in variants.items():
            sec, mb = run(lambda: generate(model, prompt, n, **kw), device)
            res[name] = (args.batch * n / sec, mb)
        row = {
            "new_tokens": n,
            "nocache_tps": res["nocache"][0],
            "cache_tps": res["cat"][0],  # "cat" cache; name kept from Phase 1
            "static_tps": res["static"][0],
            "static_vs_cat": res["static"][0] / res["cat"][0],
            "nocache_mb": res["nocache"][1],
            "cat_mb": res["cat"][1],
            "static_mb": res["static"][1],
        }
        rows.append(row)
        print(f"{n:>5} | {row['nocache_tps']:>9.1f} {row['cache_tps']:>9.1f} {row['static_tps']:>9.1f}"
              f"         | {row['static_vs_cat']:>9.2f}x | "
              f"{row['nocache_mb']:>8.0f} {row['cat_mb']:>8.0f} {row['static_mb']:>8.0f}")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
