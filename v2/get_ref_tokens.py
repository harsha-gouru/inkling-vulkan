#!/usr/bin/env python3
"""Query llama-server (F32 GGUF, CPU, -fa off, F32 KV) for exact greedy reference tokens."""
import argparse, json, os, subprocess, time, urllib.request

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--n-predict", type=int, default=20)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=18089)
    args = ap.parse_args()

    srv_bin = os.path.expanduser("~/src/llama.cpp/build/bin/llama-server")
    model   = os.path.expanduser("~/models/gguf/Inkling-2-Mini-f32.gguf")
    err_log = args.out + ".srv.err"
    os.makedirs(os.path.dirname(os.path.expanduser(args.out)), exist_ok=True)
    cmd = [
        srv_bin, "-m", model,
        "-ngl", "0", "--device", "none", "-t", "16",
        "-fa", "off", "-ctk", "f32", "-ctv", "f32",
        "-c", str(args.ctx),
        "--host", "127.0.0.1", "--port", str(args.port),
    ]
    with open(os.path.expanduser(err_log), "w") as ferr:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=ferr)
    try:
        url_base = f"http://127.0.0.1:{args.port}"
        for _ in range(240):
            if proc.poll() is not None:
                raise RuntimeError(f"llama-server exited early ({proc.returncode}); see {err_log}")
            try:
                with urllib.request.urlopen(url_base + "/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(0.5)
        else:
            raise RuntimeError("llama-server did not become healthy")

        req = urllib.request.Request(
            url_base + "/tokenize",
            data=json.dumps({"content": args.prompt, "add_special": True}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            tok_resp = json.loads(r.read().decode())

        payload = {
            "prompt": tok_resp["tokens"],
            "n_predict": args.n_predict,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "min_p": 0.0,
            "repeat_penalty": 1.0,
            "cache_prompt": False,
            "return_tokens": True,
            "ignore_eos": True,
        }
        req2 = urllib.request.Request(
            url_base + "/completion",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req2, timeout=1800) as r:
            comp_resp = json.loads(r.read().decode())

        out = {
            "prompt_tokens": tok_resp["tokens"],
            "generated_tokens": comp_resp.get("tokens", []),
            "content": comp_resp.get("content", ""),
            "timings": comp_resp.get("timings", {}),
        }
        with open(os.path.expanduser(args.out), "w") as f:
            json.dump(out, f, indent=2)
        print("PROMPT_TOKENS:", ",".join(map(str, out["prompt_tokens"])))
        print("GENERATED_TOKENS:", ",".join(map(str, out["generated_tokens"])))
        print("CONTENT:", repr(out["content"]))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

if __name__ == "__main__":
    main()
