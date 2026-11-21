import glob, os, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
from inkling_batch_vk import BatchedVulkanEngine

MODEL_DIR = Path(os.environ.get("INKLING_MODEL_DIR", "./Inkling-2-Mini"))
MODEL = str(MODEL_DIR / "model-00001-of-00001.safetensors")
tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
p500 = [int(x) for x in (BASE_DIR / "p500_tokens.txt").read_text().split()]

eng = BatchedVulkanEngine(MODEL)
vram_paths = sorted(glob.glob("/sys/class/drm/card*/device/mem_info_vram_used"))
mem_str = ""
if vram_paths:
    vp = Path(vram_paths[0])
    v = int(vp.read_text()) / (1024**3)
    g = int((vp.parent / "mem_info_gtt_used").read_text()) / (1024**3)
    mem_str = f" | VRAM={v:.2f} GiB, GTT={g:.2f} GiB"
print(f"BatchedVulkanEngine (with chunked expert-grouped prefill) ready | init={eng.init_s:.2f}s{mem_str}\n", flush=True)

# 1. Bit-exactness: p100 (99 tokens prefill -> 20 decode) single-token vs chunked prefill!
p100 = p500[:100]
eng.reset_slot(0)
for pos, t in enumerate(p100[:-1]): eng.step_batch([0], [t], [pos])
cur = p100[-1]; pos = len(p100) - 1; ref100 = []
for _ in range(20): (cur, _), = eng.step_batch([0], [cur], [pos]); pos += 1; ref100.append(cur)

eng.reset_slot(1)
pos = eng.prefill_tokens(1, p100[:-1], 0)
cur = p100[-1]; chk100 = []
for _ in range(20): (cur, _), = eng.step_batch([1], [cur], [pos]); pos += 1; chk100.append(cur)
print(f"p100 Chunked Prefill vs Single-Token (20 gen): {'PASS (20/20 BIT-EXACT!)' if ref100 == chk100 else f'DIFF: {ref100[:5]} vs {chk100[:5]}'}", flush=True)
assert ref100 == chk100

# 2. Bit-exactness: p1500 (1,499 tokens crossing SWA 1024 ring wraparound -> 15 decode)!
p1500 = (p500 * 3)[:1500]
eng.reset_slot(0)
for pos, t in enumerate(p1500[:-1]): eng.step_batch([0], [t], [pos])
cur = p1500[-1]; pos = len(p1500) - 1; ref1500 = []
for _ in range(15): (cur, _), = eng.step_batch([0], [cur], [pos]); pos += 1; ref1500.append(cur)

eng.reset_slot(10) # slot 10 exercises upper KCache1/VCache1!
pos = eng.prefill_tokens(10, p1500[:-1], 0)
cur = p1500[-1]; chk1500 = []
for _ in range(15): (cur, _), = eng.step_batch([10], [cur], [pos]); pos += 1; chk1500.append(cur)
print(f"p1500 Chunked Prefill (slot 10, ring>1024) vs Single-Token (15 gen): {'PASS (15/15 BIT-EXACT!)' if ref1500 == chk1500 else f'DIFF: {ref1500[:5]} vs {chk1500[:5]}'}\n", flush=True)
assert ref1500 == chk1500

# 3. Cold TTFT & Prefill Throughput at 50, 500, 652, 2000 tokens!
long_ids = (p1500 + p1500)[:2000]
print("=== CHUNKED EXPERT-GROUPED PREFILL SPEED (C=64/16, no output head) ===")
for n_tok in (50, 500, 652, 2000):
    ids = long_ids[:n_tok]
    eng.reset_slot(0)
    t0 = time.perf_counter()
    pos = eng.prefill_tokens(0, ids[:-1], 0)
    nxt, _ = eng.step_batch([0], [ids[-1]], [pos])[0]
    dt_ms = (time.perf_counter() - t0) * 1e3
    print(f"  Prompt {n_tok:4d} tok -> Cold TTFT = {dt_ms:7.1f} ms  ({n_tok / (dt_ms / 1e3):6.1f} tok/s)", flush=True)

# 4. Sampler Statistical Verification (4,000 draws at T=0.8, top_p=0.9 vs exact theoretical probs!)
# Download exact logits for "Hello world" at pos=1 from buf_logits via vk_download_slice!
from inkling_batch_vk import _vk
eng.reset_slot(0)
eng.step_batch([0], [32269], [0])
eng.step_batch([0], [150770], [1], temps=[0.0])
logits = np.empty(200058, dtype=np.float32)
assert _vk.vk_download_slice(eng.buf_logits, 0, logits.ctypes.data, logits.nbytes) == 0

for temp, top_p in [(0.7, 0.9), (1.0, 0.9)]:
    # Exact theoretical top-p probabilities on CPU
    mx = float(np.max(logits))
    ex = np.exp((logits.astype(np.float64) - mx) / temp)
    probs_all = ex / ex.sum()
    order = np.argsort(-probs_all)[:64]
    p_top = probs_all[order]
    csum = np.cumsum(p_top)
    cut = int(np.searchsorted(csum, top_p)) + 1
    theo_ids = order[:cut]
    theo_p = p_top[:cut] / p_top[:cut].sum()

    # Draw 4,000 samples from the GPU sampler with seeds 1..4000 (in batches of 16 for speed!)
    # Reset slot 0 to pos=1 each time (snapshot conv after token 0!)
    eng.reset_slot(0)
    eng.step_batch([0], [32269], [0])
    snap0 = eng.snapshot_slot_conv(0)
    counts = Counter()
    N_DRAWS = 4000
    for s in range(1, N_DRAWS + 1):
        eng.restore_slot_conv(0, snap0)
        (tid, _), = eng.step_batch([0], [150770], [1], temps=[temp], top_ps=[top_p], seeds=[s])
        counts[tid] += 1
        if s % 1000 == 0: pass

    emp_p = np.array([counts[int(t)] / N_DRAWS for t in theo_ids])
    max_abs = float(np.max(np.abs(emp_p - theo_p)))
    # Chi-squared on categories with expected >= 5
    mask = (theo_p * N_DRAWS) >= 5.0
    if mask.sum() >= 2:
        obs = np.array([counts[int(t)] for t in theo_ids[mask]], dtype=np.float64)
        exp_c = theo_p[mask] * N_DRAWS
        chi2 = float(np.sum((obs - exp_c) ** 2 / exp_c))
        df = int(mask.sum() - 1)
    else:
        chi2, df = 0.0, 0
    print(f"\nSampler check (T={temp}, top_p={top_p}, N={N_DRAWS}, nucleus={len(theo_ids)} toks): max |emp-theo|={max_abs:.4f} | chi2={chi2:.2f} (df={df})", flush=True)
    for idx_k in range(min(5, len(theo_ids))):
        print(f"  tok={int(theo_ids[idx_k]):6d} ({tok.decode([int(theo_ids[idx_k])])!r:10s}): theo={theo_p[idx_k]:.4f}  emp={emp_p[idx_k]:.4f}")

eng.close()
