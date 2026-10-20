"""Benchmark decoding speed with and without the KV cache.

Usage: python benchmark.py [--new-tokens 256] [--batch 1] [--out results.csv]
"""
import argparse
import csv
import time

import torch

from kvcache import GPT, GPTConfig, generate


def timed(fn, device):
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0


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

    generate(model, prompt, 8, use_cache=True)  # warmup

    rows = []
    print(f"device={device} batch={args.batch} prompt_len={args.prompt_len}")
    print(f"{'new_tokens':>10} {'cache tok/s':>12} {'no-cache tok/s':>15} {'speedup':>8}")
    for n in args.new_tokens:
        t_c = timed(lambda: generate(model, prompt, n, use_cache=True), device)
        t_n = timed(lambda: generate(model, prompt, n, use_cache=False), device)
        tps_c, tps_n = args.batch * n / t_c, args.batch * n / t_n
        rows.append({"new_tokens": n, "cache_tps": tps_c, "nocache_tps": tps_n, "speedup": tps_c / tps_n})
        print(f"{n:>10} {tps_c:>12.1f} {tps_n:>15.1f} {tps_c / tps_n:>7.1f}x")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
