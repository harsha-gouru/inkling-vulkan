#!/usr/bin/env python3
"""Stage 1 gate verifier: compares ref_cpu 20-token output against F32 llama.cpp (-fa off) answer key."""
import json, os, subprocess, sys

key_path = os.path.expanduser("~/src/bench/key32/prose1_20tok.json")
with open(key_path) as f:
    key = json.load(f)

prompt_toks = ",".join(map(str, key["prompt_tokens"]))
ref_toks = key["generated_tokens"]

out = subprocess.check_output(
    [os.path.expanduser("~/inkling2/ref_cpu"), "--tokens", prompt_toks, "--steps", str(len(ref_toks))],
    text=True,
)
gen_line = [l for l in out.splitlines() if l.startswith("GENERATED:")][0]
got_toks = [int(x) for x in gen_line.split(":", 1)[1].strip().split(",")]

matches = sum(a == b for a, b in zip(ref_toks, got_toks))
print(f"Stage 1 match: {matches}/{len(ref_toks)}")
if matches != len(ref_toks):
    for i, (a, b) in enumerate(zip(ref_toks, got_toks)):
        if a != b:
            print(f"  FIRST MISMATCH at step {i}: ref={a} got={b}")
            break
    sys.exit(1)
print("PASS: 20/20 exact token match.")
