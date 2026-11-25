#!/usr/bin/env python3
"""Requirement A (short prompt) verifier: 500/500 exact next tokens against F32 llama.cpp (-fa off)."""
import json, os, subprocess, sys

key_path = os.path.expanduser("~/src/bench/key32/prose1_500tok.json")
with open(key_path) as f:
    key = json.load(f)

bin_path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/inkling2/inkling_vk")
prompt_toks = ",".join(map(str, key["prompt_tokens"]))
ref_toks = key["generated_tokens"]
assert len(ref_toks) == 500, f"Expected 500 ref tokens, got {len(ref_toks)}"

out = subprocess.check_output(
    [bin_path, "--tokens", prompt_toks, "--steps", "500"],
    text=True,
)
gen_line = [l for l in out.splitlines() if l.startswith("GENERATED:")][0]
got_toks = [int(x) for x in gen_line.split(":", 1)[1].strip().split(",")]

matches = sum(a == b for a, b in zip(ref_toks, got_toks))
print(f"Requirement A (short, {os.path.basename(bin_path)}): {matches}/500 exact token match")
if matches != 500:
    for i, (a, b) in enumerate(zip(ref_toks, got_toks)):
        if a != b:
            print(f"  FIRST MISMATCH at step {i}: ref={a} got={b}")
            break
    sys.exit(1)
print("PASS: 500/500 exact match!")
