import concurrent.futures, json, os, subprocess, sys, time, urllib.error, urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = os.environ.get("INKLING_MODEL_DIR", "./Inkling-2-Mini")
PORT = 8091
BASE = f"http://127.0.0.1:{PORT}"
KEY  = "test-secret-key-123"

def req_json(path, body=None, key=KEY, timeout=120):
    hdrs = {}
    if key is not None: hdrs["Authorization"] = f"Bearer {key}"
    data = None
    if body is not None:
        hdrs["Content-Type"] = "application/json"
        data = json.dumps(body).encode()
    r = urllib.request.Request(BASE + path, data=data, headers=hdrs)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())

SYS_A = "Reference A:\n" + "\n".join(
    f"Section {i}: Distributed systems rely on deterministic state machines, consensus protocols, "
    f"and bounded-latency replication queues to preserve consistency across independent worker nodes "
    f"operating under variable network jitter and partial hardware failures."
    for i in range(1, 26)
)
SYS_B = "Reference B:\n" + "\n".join(
    f"Note {i}: Modern GPU compute pipelines balance shared-memory occupancy, subgroup reductions, "
    f"and coalesced global memory transactions to sustain high bandwidth."
    for i in range(1, 10)
)

proc = subprocess.Popen(
    [sys.executable, str(BASE_DIR / "serve_batch_vk.py"),
     "--model-dir", MODEL_DIR, "--port", str(PORT), "--api-key", KEY, "--max-queue", "2", "--timeout", "60"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
)
try:
    for _ in range(80):
        try:
            code, h = req_json("/health", key=None, timeout=5)
            if code == 200: break
        except Exception:
            time.sleep(0.25)
    print(f"1. /health (no auth, 127.0.0.1 default): {code} {h}")
    assert code == 200 and h["seq_cap"] == 4096 and h["swa_window"] == 1024

    # 2. API key enforcement
    c_no, _   = req_json("/v1/models", key=None)
    c_bad, _  = req_json("/v1/models", key="wrong-key")
    c_ok, _   = req_json("/v1/models", key=KEY)
    print(f"2. API key auth: missing={c_no}, wrong={c_bad}, valid={c_ok}")
    assert c_no == 401 and c_bad == 401 and c_ok == 200

    # 3. GPU Sampling (temperature, top_p, deterministic seed)
    b_s1 = {"messages": [{"role": "user", "content": "Write a creative 1-sentence opening about Mars."}],
            "reasoning_effort": "none", "temperature": 0.8, "top_p": 0.9, "seed": 42, "max_tokens": 32}
    _, s1a = req_json("/v1/chat/completions", b_s1)
    _, s1b = req_json("/v1/chat/completions", b_s1)
    b_s2 = dict(b_s1, seed=99)
    _, s2  = req_json("/v1/chat/completions", b_s2)
    t1a, t1b, t2 = s1a["choices"][0]["message"]["content"], s1b["choices"][0]["message"]["content"], s2["choices"][0]["message"]["content"]
    print(f"3. GPU Sampling: same_seed_exact={t1a == t1b} | diff_seed_distinct={t1a != t2}")
    print(f"   seed=42: {t1a!r}\n   seed=99: {t2!r}")
    assert t1a == t1b and t1a != t2

    # 4. Multi-conversation LRU prompt caching + rollback!
    mA1 = [{"role": "system", "content": SYS_A}, {"role": "user", "content": "Summarize in one sentence."}]
    _, rA1 = req_json("/v1/chat/completions", {"messages": mA1, "reasoning_effort": "none", "temperature": 0.0, "max_tokens": 40})
    aA1 = rA1["choices"][0]["message"]["content"]

    # Interleave a completely different User B conversation (must NOT evict User A's warm slot!)
    mB1 = [{"role": "system", "content": SYS_B}, {"role": "user", "content": "Summarize in one sentence."}]
    _, rB1 = req_json("/v1/chat/completions", {"messages": mB1, "reasoning_effort": "none", "temperature": 0.0, "max_tokens": 30})

    # User A Turn 2 (continues User A's warm slot even after User B ran!)
    mA2 = mA1 + [{"role": "assistant", "content": aA1}, {"role": "user", "content": "Give 3 keywords."}]
    _, rA2 = req_json("/v1/chat/completions", {"messages": mA2, "reasoning_effort": "none", "temperature": 0.0, "max_tokens": 30})

    # User A Turn 3 (rolls back past Turn 1 & 2 to SYS_A!)
    mA3 = [{"role": "system", "content": SYS_A}, {"role": "user", "content": "What is the main topic in 5 words?"}]
    _, rA3 = req_json("/v1/chat/completions", {"messages": mA3, "reasoning_effort": "none", "temperature": 0.0, "max_tokens": 30})

    print(f"4. Multi-conv LRU prompt cache:")
    print(f"   User A Turn 1 (cold)    : cached={rA1['timings']['cached_tokens']:3d}/{rA1['usage']['prompt_tokens']} | TTFT={rA1['timings']['ttft_ms']:7.1f} ms")
    print(f"   User B Turn 1 (interlv) : cached={rB1['timings']['cached_tokens']:3d}/{rB1['usage']['prompt_tokens']} | TTFT={rB1['timings']['ttft_ms']:7.1f} ms")
    print(f"   User A Turn 2 (warm hit): cached={rA2['timings']['cached_tokens']:3d}/{rA2['usage']['prompt_tokens']} | TTFT={rA2['timings']['ttft_ms']:7.1f} ms | restore={rA2['timings']['restore_ms']:.2f} ms")
    print(f"   User A Turn 3 (rollback): cached={rA3['timings']['cached_tokens']:3d}/{rA3['usage']['prompt_tokens']} | TTFT={rA3['timings']['ttft_ms']:7.1f} ms | restore={rA3['timings']['restore_ms']:.2f} ms | reply={rA3['choices'][0]['message']['content']!r}")
    assert rA2["timings"]["cached_tokens"] > 600 and rA3["timings"]["cached_tokens"] > 600

    # 5. Request timeout (504) + slot recovery
    c_to, r_to = req_json("/v1/chat/completions", {
        "messages": [{"role": "user", "content": "Write a 500-word essay about ocean currents."}],
        "reasoning_effort": "none", "max_tokens": 256, "timeout": 0.35,
    })
    c_rec, _ = req_json("/v1/chat/completions", {
        "messages": [{"role": "user", "content": "Say OK."}],
        "reasoning_effort": "none", "max_tokens": 8,
    })
    print(f"5. Request timeout & slot recovery: timed_out_code={c_to}, next_req_code={c_rec}")
    assert c_to == 504 and c_rec == 200

    # 6. Bounded queue (429 Too Many Requests when > 16 active + 2 queued = 18 arrive)
    def flood(i):
        return req_json("/v1/chat/completions", {
            "messages": [{"role": "user", "content": f"Count from 1 to 30 slowly (request {i})."}],
            "reasoning_effort": "none", "max_tokens": 60,
        })[0]
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as ex:
        codes = list(ex.map(flood, range(24)))
    n200, n429 = codes.count(200), codes.count(429)
    print(f"6. Queue limit (--max-queue 2, 24 concurrent reqs): 200={n200}, 429={n429}")
    assert n429 >= 1 and n200 + n429 == 24
    print("\nALL PRODUCTION SERVER TESTS PASSED!")
finally:
    proc.terminate()
    try: proc.wait(timeout=5)
    except Exception: proc.kill()
