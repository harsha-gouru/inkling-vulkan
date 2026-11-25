import json, os, subprocess, sys, threading, time, urllib.request
from collections import Counter
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("INKLING_MODEL_DIR", "./Inkling-2-Mini"))
sys.path.insert(0, str(BASE_DIR))

# Part 1: Verify upgraded top-3/thread + top-128 GPU sampler chi-squared at T=1.0, top_p=0.9!
def verify_sampler():
    from inkling_batch_vk import BatchedVulkanEngine, _vk
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    eng = BatchedVulkanEngine(str(MODEL_DIR / "model-00001-of-00001.safetensors"))
    eng.reset_slot(0)
    eng.step_batch([0], [32269], [0])
    snap0 = eng.snapshot_slot_conv(0)
    eng.step_batch([0], [150770], [1], temps=[0.0])
    logits = np.empty(200058, dtype=np.float32)
    assert _vk.vk_download_slice(eng.buf_logits, 0, logits.ctypes.data, logits.nbytes) == 0

    for temp, top_p in [(0.7, 0.9), (1.0, 0.9)]:
        mx = float(np.max(logits))
        ex = np.exp((logits.astype(np.float64) - mx) / temp)
        probs_all = ex / ex.sum()
        order = np.argsort(-probs_all)[:128]
        p_top = probs_all[order]
        cut = int(np.searchsorted(np.cumsum(p_top), top_p)) + 1
        theo_ids = order[:cut]
        theo_p = p_top[:cut] / p_top[:cut].sum()

        counts = Counter()
        N = 4000
        for s in range(1, N + 1):
            eng.restore_slot_conv(0, snap0)
            (tid, _), = eng.step_batch([0], [150770], [1], temps=[temp], top_ps=[top_p], seeds=[s])
            counts[tid] += 1
        emp_p = np.array([counts[int(t)] / N for t in theo_ids])
        max_abs = float(np.max(np.abs(emp_p - theo_p)))
        mask = (theo_p * N) >= 5.0
        obs = np.array([counts[int(t)] for t in theo_ids[mask]], dtype=np.float64)
        exp_c = theo_p[mask] * N
        chi2 = float(np.sum((obs - exp_c) ** 2 / exp_c))
        df = int(mask.sum() - 1)
        print(f"Upgraded Sampler (T={temp}, top_p={top_p}, N={N}, nucleus={len(theo_ids)}): max |emp-theo|={max_abs:.4f} | chi2={chi2:.2f} (df={df})", flush=True)
    eng.close()

# Part 2: Live HTTP TTFT Matrix on serve_batch_vk.py (50 / 500 / 2,000 tok, Cold & Warm, Alone & +8 generating!)
def run_ttft_matrix():
    PORT = 8092
    BASE = f"http://127.0.0.1:{PORT}"
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    p500 = [int(x) for x in (BASE_DIR / "p500_tokens.txt").read_text().split()]
    raw_doc = tok.decode(p500 * 5)

    def make_sys(target_toks):
        # Binary-search character slice to hit target_toks closely
        ids = tok.encode(raw_doc, add_special_tokens=False).ids[:target_toks]
        return tok.decode(ids)

    def post(body):
        req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read().decode())

    proc = subprocess.Popen(
        [sys.executable, str(BASE_DIR / "serve_batch_vk.py"),
         "--model-dir", str(MODEL_DIR), "--port", str(PORT), "--timeout", "180"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        for _ in range(80):
            try:
                with urllib.request.urlopen(BASE + "/health", timeout=5) as r:
                    if r.status == 200: break
            except Exception:
                time.sleep(0.25)

        # Warmup
        post({"messages": [{"role": "user", "content": "Hi"}], "reasoning_effort": "none", "max_tokens": 4})

        print("\n=== LIVE HTTP TTFT MATRIX (serve_batch_vk.py: chunked C=64/16 prefill + expert grouping) ===")
        print(f"{'Prompt':>8s} | {'Load':>14s} | {'Cold TTFT':>10s} | {'Warm TTFT':>10s} | {'Cached/Tot':>11s} | {'BG dec tok/s':>12s}")
        print("-" * 80)

        for target in (50, 500, 2000):
            sys_txt = make_sys(max(10, target - 25))
            # Use unique prefix tag per target so Cold is genuinely cold!
            for load_name, n_bg in [("1 user alone", 0), ("+8 generating", 8)]:
                tag = f"[run-{target}-{n_bg}] "
                m_cold = [{"role": "system", "content": tag + sys_txt}, {"role": "user", "content": "Give a one-sentence summary."}]
                m_warm = [{"role": "system", "content": tag + sys_txt}, {"role": "user", "content": "Name three key terms."}]

                stop_bg = threading.Event()
                bg_tps = []
                bg_threads = []
                if n_bg > 0:
                    def bg_worker(idx):
                        while not stop_bg.is_set():
                            try:
                                r = post({"messages": [{"role": "user", "content": f"Write a short story about space traveler #{idx}."}],
                                          "reasoning_effort": "none", "max_tokens": 96})
                                bg_tps.append(r["timings"]["decode_tok_s"])
                            except Exception:
                                break
                    for bi in range(n_bg):
                        t = threading.Thread(target=bg_worker, args=(bi,), daemon=True)
                        t.start(); bg_threads.append(t)
                    time.sleep(0.4)  # let the 8 background users enter steady decode!

                r_cold = post({"messages": m_cold, "reasoning_effort": "none", "max_tokens": 16})
                r_warm = post({"messages": m_warm, "reasoning_effort": "none", "max_tokens": 16})

                if n_bg > 0:
                    stop_bg.set()
                    for t in bg_threads: t.join(timeout=15)

                pt = r_cold["usage"]["prompt_tokens"]
                c_ttft = r_cold["timings"]["ttft_ms"]
                w_ttft = r_warm["timings"]["ttft_ms"]
                cached = r_warm["timings"]["cached_tokens"]
                w_pt   = r_warm["usage"]["prompt_tokens"]
                bg_str = f"{np.median(bg_tps):6.1f}/user" if bg_tps else "—"
                print(f"{pt:5d} tk | {load_name:>14s} | {c_ttft:8.1f}ms | {w_ttft:8.1f}ms | {cached:4d}/{w_pt:<4d}  | {bg_str:>12s}", flush=True)
    finally:
        proc.terminate()
        try: proc.wait(timeout=5)
        except Exception: proc.kill()

if __name__ == "__main__":
    verify_sampler()
    run_ttft_matrix()
