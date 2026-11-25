#!/usr/bin/env python3
"""Formal 5-run (after 1 warmup) gate benchmark suite for Inkling-2-Mini clean build."""
import re
import statistics
import subprocess

BIN = "./inkling_vk"


def run_cmd(args):
    p = subprocess.run([BIN] + args, capture_output=True, text=True, check=True)
    return p.stderr + "\n" + p.stdout


def bench_regex(label, args, pattern, runs=6):
    vals = []
    for i in range(runs):
        out = run_cmd(args)
        m = re.search(pattern, out)
        if not m:
            raise RuntimeError(f"Pattern not found for {label}:\n{out}")
        v = float(m.group(1))
        if i >= 1:  # skip warmup
            vals.append(v)
    med = statistics.median(vals)
    print(f"{label:32s}: median={med:7.1f} tok/s | spread=[{min(vals):.1f}, {max(vals):.1f}] | runs={[round(x,1) for x in vals]}")
    return med, min(vals), max(vals)


print("=== Req E: 128,000-token VRAM check ===")
out_e = run_cmd(["--max-ctx", "128000", "--spec", "--check-vram"])
for l in out_e.splitlines():
    if "VRAM" in l or "Loaded" in l:
        print(" ", l)

print("\n=== Req B1: Short-context decode (5 runs after warmup) ===")
bench_regex("B1 code_1 (--spec K=7, ctx~134)", ["--tokens-file", "code_1_toks.txt", "--max-ctx", "1024", "--steps", "200", "--spec"], r"Speculative decode.*?:\s+([0-9.]+)\s+tok/s")
bench_regex("B1 code_2 (--spec K=7, ctx~143)", ["--tokens-file", "code_2_toks.txt", "--max-ctx", "1024", "--steps", "200", "--spec"], r"Speculative decode.*?:\s+([0-9.]+)\s+tok/s")
bench_regex("B1 prose_1 (plain B=1, ctx~118)", ["--tokens-file", "prose1_toks.txt", "--max-ctx", "1024", "--steps", "200"], r"Decode wall-clock.*?\(([0-9.]+)\s+tok/s\)")
bench_regex("B1 prose_2 (plain B=1, ctx~117)", ["--tokens-file", "prose_2_toks.txt", "--max-ctx", "1024", "--steps", "200"], r"Decode wall-clock.*?\(([0-9.]+)\s+tok/s\)")

print("\n=== Req C: Cold 25,000-token prefill (5 runs after warmup) ===")
bench_regex("C cold 25k prefill (B=256)", ["--tokens-file", "long25k_toks.txt", "--max-ctx", "26000", "--steps", "1"], r"Prefill 25000 tokens.*?\(([0-9.]+)\s+tok/s\)")

print("\n=== Req B2: 40,000-token decode (5 runs after warmup) ===")
bench_regex("B2 40k decode (--spec K=7)", ["--tokens-file", "long40k_toks.txt", "--max-ctx", "41000", "--steps", "200", "--spec"], r"Speculative decode.*?:\s+([0-9.]+)\s+tok/s")
