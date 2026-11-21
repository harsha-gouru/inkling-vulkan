import glob, os, subprocess, sys, time
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
from inkling_batch_vk import BatchedVulkanEngine, SEQ_CAP, SWA_WIN

MODEL_DIR = Path(os.environ.get("INKLING_MODEL_DIR", "./Inkling-2-Mini"))
MODEL_ST  = str(MODEL_DIR / "model-00001-of-00001.safetensors")
TOK_PATH  = str(MODEL_DIR / "tokenizer.json")
MODEL_F32 = os.environ.get("INKLING_F32_GGUF", "")
LLAMA_BIN = str(BASE_DIR / "llama_argmax500")
TOK_FILE  = str(BASE_DIR / "p500_tokens.txt")

tok = Tokenizer.from_file(TOK_PATH)
p500 = [int(x) for x in open(TOK_FILE).read().split()]
assert len(p500) == 500

# 1. Run BatchedVulkanEngine
eng = BatchedVulkanEngine(MODEL_ST)
vram_paths = sorted(glob.glob("/sys/class/drm/card*/device/mem_info_vram_used"))
mem_str = ""
if vram_paths:
    vp = Path(vram_paths[0])
    v = int(vp.read_text()) / (1024**3)
    g = int((vp.parent / "mem_info_gtt_used").read_text()) / (1024**3)
    mem_str = f" | real VRAM={v:.2f} GiB, GTT={g:.2f} GiB"
print(f"BatchedVulkanEngine ready | SEQ_CAP={SEQ_CAP} SWA_WIN={SWA_WIN} | init={eng.init_s:.2f}s{mem_str}\n", flush=True)

# 2. Run 500 tokens ALONE (B=1, slot 0)
eng.reset_slot(0)
b1_preds = []
t0 = time.perf_counter()
for pos, tid in enumerate(p500):
    (nxt, sc), = eng.step_batch([0], [tid], [pos])
    b1_preds.append((nxt, sc))
print(f"B=1 (alone) 500 tokens done in {time.perf_counter()-t0:.2f}s", flush=True)

# 3. Run 500 tokens INSIDE A BATCH OF 8 OTHER ACTIVE USERS (B=9 -> bucket 16)!
# Put target user on seq_slot=10 (exercises upper K1/V1 half!) while 8 other users run distinct text!
other_seqs = [(p500[i*50:] + p500[:i*50])[:500] for i in range(1, 9)]
other_slots = [0, 1, 2, 3, 4, 8, 9, 12]
target_slot = 10
all_slots = other_slots[:4] + [target_slot] + other_slots[4:] # target at batch index 4!
for s in all_slots:
    eng.reset_slot(s)

b9_preds = []
t0 = time.perf_counter()
for pos in range(500):
    tids = [other_seqs[i][pos] for i in range(4)] + [p500[pos]] + [other_seqs[i][pos] for i in range(4, 8)]
    poss = [pos] * 9
    outs = eng.step_batch(all_slots, tids, poss)
    b9_preds.append(outs[4])
print(f"B=9 (target on slot 10 + 8 active users) 500 tokens done in {time.perf_counter()-t0:.2f}s", flush=True)

# Check B=1 vs B=9 exactness first
m_19 = sum(1 for a, b in zip(b1_preds, b9_preds) if a[0] == b[0])
d_19 = max(abs(a[1] - b[1]) for a, b in zip(b1_preds, b9_preds))
print(f"B=1 vs B=9 agreement: {m_19}/500 | max |logit diff| = {d_19:.6f}", flush=True)

# 4. SWA 1,024 Ring-Buffer Wraparound Test on p1500 (1,500 tokens > 1,024 window!)
p1500 = (p500 * 3)[:1500]
eng.reset_slot(11) # slot 11 -> KCache1/VCache1
for pos, tid in enumerate(p1500[:-1]):
    eng.step_batch([11], [tid], [pos])
cur = p1500[-1]; pos = len(p1500) - 1
ring_gen = []
for _ in range(15):
    (cur, _), = eng.step_batch([11], [cur], [pos]); pos += 1; ring_gen.append(cur)

# 5. On-GPU Sampling Test (temp, top_p, deterministic seed + mixed greedy/sampled batch!)
def sample_run(seed, temp=0.8, top_p=0.9):
    eng.reset_slot(0)
    eng.step_batch([0], [32269], [0])
    cur = 150770; out = []
    for p in range(1, 21):
        (cur, _), = eng.step_batch([0], [cur], [p], temps=[temp], top_ps=[top_p], seeds=[seed])
        out.append(cur)
    return out

s_a1 = sample_run(12345, 0.8, 0.9)
s_a2 = sample_run(12345, 0.8, 0.9)
s_b  = sample_run(99999, 0.8, 0.9)
print(f"\nGPU Sampler deterministic seed (12345 vs 12345): {'PASS (20/20 IDENTICAL)' if s_a1 == s_a2 else 'FAIL'}")
print(f"GPU Sampler different seeds    (12345 vs 99999): {'PASS (DISTINCT)' if s_a1 != s_b else 'FAIL'}")
print(f"  seed=12345: {tok.decode(s_a1)!r}")
print(f"  seed=99999: {tok.decode(s_b)!r}")

# Mixed greedy + sampled in the SAME GPU pass!
eng.reset_slot(0); eng.reset_slot(1)
eng.step_batch([0, 1], [32269, 32269], [0, 0])
c0, c1 = 150770, 150770; g0, g1 = [], []
for p in range(1, 6):
    (c0, _), (c1, _) = eng.step_batch([0, 1], [c0, c1], [p, p], temps=[0.0, 0.8], top_ps=[1.0, 0.9], seeds=[0, 12345])
    g0.append(c0); g1.append(c1)
print(f"Mixed batch (user0 greedy + user1 sampled): greedy_ok={g0 == [0, 68014, 126311, 149882, 150084]} | sampled_ok={g1 == s_a1[:5]}", flush=True)
assert s_a1 == s_a2 and s_a1 != s_b and g0 == [0, 68014, 126311, 149882, 150084] and g1 == s_a1[:5]

# Compare ring_gen (slot 11 single-token) vs chunked prefill on slot 0 at depth 1500!
eng.reset_slot(0)
pos = eng.prefill_tokens(0, p1500[:-1], 0)
cur = p1500[-1]; pref_gen = []
for _ in range(15):
    (cur, _), = eng.step_batch([0], [cur], [pos]); pos += 1; pref_gen.append(cur)
print(f"\nSWA 1024 Ring-Buffer slot 11 vs slot 0 chunked prefill at depth 1500 (15 tok): {'PASS (15/15 BIT-EXACT!)' if ring_gen == pref_gen else f'DIFF: {ring_gen} vs {pref_gen}'}", flush=True)
assert ring_gen == pref_gen

eng.close()

# 6. Optional: Run llama.cpp F32 -fa off on the exact same 500 tokens if GGUF + binary are provided
if MODEL_F32 and os.path.isfile(MODEL_F32) and os.path.isfile(LLAMA_BIN):
    t0 = time.perf_counter()
    res = subprocess.run([LLAMA_BIN, MODEL_F32, "0", "0", TOK_FILE], capture_output=True, text=True, check=True)
    llama_preds = [(int(l.split()[1]), float(l.split()[2])) for l in res.stdout.strip().splitlines()]
    assert len(llama_preds) == 500
    print(f"\nllama.cpp F32 (-fa off) 500 tokens done in {time.perf_counter()-t0:.1f}s")

    for label, preds in [("Serving Engine ALONE (B=1) vs llama.cpp F32 (-fa off)", b1_preds),
                         ("Serving Engine IN BATCH OF 9 (B=9, slot 10) vs llama.cpp F32 (-fa off)", b9_preds)]:
        matches = sum(1 for a, b in zip(preds, llama_preds) if a[0] == b[0])
        diffs = [abs(a[1] - b[1]) for a, b in zip(preds, llama_preds)]
        print(f"  {label}: {matches}/500 ({100*matches/500:.2f}%) | median |diff|={np.median(diffs):.5f}, max |diff|={np.max(diffs):.5f}")
        assert matches == 500, f"Expected 500/500, got {matches}/500!"
else:
    print("\nSkipping llama.cpp F32 comparison (set INKLING_F32_GGUF and build llama_argmax500 to enable).")
