import argparse, hmac, json, os, queue, sys, threading, time, uuid
from collections import deque
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tokenizers import Tokenizer

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
from inkling_batch_vk import BatchedVulkanEngine, MAX_BATCH, SEQ_CAP, SWA_WIN

MODEL_ID   = "Inkling-2-Mini"
EFFORT_MAP = {"none": 0.0, "minimal": 0.1, "low": 0.2, "medium": 0.7, "high": 0.9, "xhigh": 0.99, "max": 0.99}
ROLE_TOK   = {"user": "<|message_user|>", "assistant": "<|message_model|>", "system": "<|message_system|>", "tool": "<|message_tool|>"}

TOK_EOT        = 199999  # <|endoftext|>
TOK_MSG_MODEL  = 200001  # <|message_model|>
TOK_TEXT       = 200004  # <|content_text|>
TOK_END_SAMPLE = 200006  # <|content_model_end_sampling|>
TOK_THINK      = 200008  # <|content_thinking|>
TOK_END_MSG    = 200010  # <|end_message|>
STOP_TOKEN_IDS = {TOK_EOT, TOK_END_SAMPLE}
SPECIAL_CTRL   = {TOK_MSG_MODEL, TOK_TEXT, TOK_THINK, TOK_END_MSG, TOK_EOT, TOK_END_SAMPLE}

def format_chat_prompt(messages, reasoning_effort=0.9):
    if isinstance(reasoning_effort, str):
        eff = EFFORT_MAP.get(reasoning_effort.strip().lower(), 0.9)
    elif reasoning_effort is None:
        eff = 0.9
    else:
        eff = float(reasoning_effort)
    eff_str = "0" if eff == 0.0 else str(eff)
    effort_block = f"<|message_system|><|content_text|>Thinking effort level: {eff_str}<|end_message|>"

    parts = []
    effort_emitted = False
    for m in messages:
        role = m.get("role", "user")
        if not effort_emitted and role != "system":
            parts.append(effort_block)
            effort_emitted = True
        rtok = ROLE_TOK.get(role, "<|message_user|>")
        if role == "assistant" and m.get("reasoning_content"):
            parts.append(f"{rtok}<|content_thinking|>{m['reasoning_content']}<|end_message|>")
        content = m.get("content", "")
        if isinstance(content, list):
            content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
        if content or role != "assistant":
            parts.append(f"{rtok}<|content_text|>{content}<|end_message|>")
        if role == "assistant":
            parts.append("<|content_model_end_sampling|>")
    if not effort_emitted:
        parts.append(effort_block)
    if eff == 0.0:
        parts.append("<|message_model|><|content_text|>")
    return "".join(parts)

class ActiveReq:
    def __init__(self, prompt_ids, max_tokens, temperature, top_p, seed, stop_strings, parse_chat, deadline):
        self.prompt_ids = prompt_ids[-(SEQ_CAP - 2):] if len(prompt_ids) >= SEQ_CAP - 1 else prompt_ids
        self.max_tokens = max_tokens
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.seed = int(seed) & 0xFFFFFFFF if seed is not None else (uuid.uuid4().int & 0xFFFFFFFF)
        self.stop_strings = stop_strings
        self.parse_chat = parse_chat
        self.deadline = deadline
        self.out_q = queue.Queue()
        self.slot = -1
        self.pos = 0
        self.cur = self.prompt_ids[0]
        self.cached_tokens = 0
        self.restore_ms = 0.0
        self.mode = "init"
        self.n_gen = 0
        self.text_accum = ""
        self.t_arrived = time.perf_counter()
        self.t_first = None

class ContinuousBatchScheduler:
    def __init__(self, model_dir, max_queue=32):
        model_dir = Path(model_dir)
        model_path = model_dir / "model-00001-of-00001.safetensors"
        tok_path = model_dir / "tokenizer.json"
        if not model_path.is_file() or not tok_path.is_file():
            raise FileNotFoundError(
                f"Inkling-2-Mini weights not found in {model_dir}.\n"
                f"Download them with:\n  hf download thinkingmachineslab/Inkling-2-Mini --local-dir {model_dir}"
            )
        self.tok = Tokenizer.from_file(str(tok_path))
        self.eng = BatchedVulkanEngine(str(model_path))
        self.max_queue = max_queue
        self.pending = queue.Queue(maxsize=max_queue)
        self.wake = threading.Event()
        self.stop_flag = False
        # Per-slot LRU prompt cache state: idle slots stay warm!
        self.slot_cache = {
            s: {"tokens": [], "pos": 0, "snaps": {}, "last_used": 0.0}
            for s in range(MAX_BATCH)
        }
        self.idle_slots = set(range(MAX_BATCH))
        self.active = []
        self.stats_path = str(BASE_DIR / "usage_stats.json")
        self.tok_events = deque()  # (timestamp, n_tokens) for rolling 2s live tok/s
        self.session_stats = {"requests": 0, "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0}
        self.persisted = self._load_stats()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        print(f"[serve_batch_vk] Ready: {MODEL_ID} | MAX_BATCH={MAX_BATCH} | SEQ_CAP={SEQ_CAP} (SWA ring={SWA_WIN}) | max_queue={max_queue} | init={self.eng.init_s:.2f}s", flush=True)

    def _load_stats(self):
        try:
            d = json.load(open(self.stats_path))
            if "all_time" in d and "daily" in d: return d
        except Exception:
            pass
        return {"all_time": {"requests": 0, "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0}, "daily": {}}

    def _record_completed(self, prompt_toks: int, cached_toks: int, comp_toks: int):
        today = date.today().isoformat()
        for bucket in (self.session_stats, self.persisted["all_time"],
                       self.persisted["daily"].setdefault(today, {"requests": 0, "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0})):
            bucket["requests"] += 1
            bucket["prompt_tokens"] += prompt_toks
            bucket["cached_tokens"] += cached_toks
            bucket["completion_tokens"] += comp_toks
        try:
            tmp = self.stats_path + ".tmp"
            with open(tmp, "w") as f: json.dump(self.persisted, f)
            os.replace(tmp, self.stats_path)
        except Exception:
            pass

    def get_stats(self):
        now = time.perf_counter()
        while self.tok_events and now - self.tok_events[0][0] > 2.0:
            self.tok_events.popleft()
        live_tps = sum(n for _, n in self.tok_events) / 2.0
        kv_used = sum(sc["pos"] for sc in self.slot_cache.values())
        warm_slots = sum(1 for sc in self.slot_cache.values() if sc["pos"] > 0)
        today = date.today().isoformat()
        day = self.persisted["daily"].get(today, {"requests": 0, "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0})
        return {
            "online": True, "model": MODEL_ID,
            "active": len(self.active), "queued": self.pending.qsize(),
            "max_batch": MAX_BATCH, "max_queue": self.max_queue,
            "live_tps": round(live_tps, 1),
            "kv_used_tokens": kv_used, "kv_cap_tokens": MAX_BATCH * SEQ_CAP,
            "warm_slots": warm_slots,
            "session": self.session_stats,
            "today": day,
            "all_time": self.persisted["all_time"],
        }

    def submit(self, req: ActiveReq) -> bool:
        try:
            self.pending.put_nowait(req)
            self.wake.set()
            return True
        except queue.Full:
            return False

    def _save_snap(self, slot: int):
        sc = self.slot_cache[slot]
        pos = sc["pos"]
        if pos > 0 and pos not in sc["snaps"]:
            sc["snaps"][pos] = self.eng.snapshot_slot_conv(slot)
            # Prune snapshots overwritten by the 1024-token SWA ring buffer, and cap count at 16/slot
            min_valid = max(1, pos - SWA_WIN + 1)
            for k in [k for k in sc["snaps"] if k < min_valid]:
                del sc["snaps"][k]
            if len(sc["snaps"]) > 16:
                keys = sorted(sc["snaps"].keys())
                del sc["snaps"][keys[len(keys) // 2]]

    def _pick_and_prepare_slot(self, prompt_ids):
        # Scan all idle slots for the longest ring-safe restorable shared prefix!
        best_slot, best_cp = -1, 0
        for s in self.idle_slots:
            sc = self.slot_cache[s]
            ctoks = sc["tokens"]
            s_pos = sc["pos"]
            if not ctoks:
                continue
            max_lcp = min(len(ctoks), len(prompt_ids) - 1)
            lcp = 0
            while lcp < max_lcp and ctoks[lcp] == prompt_ids[lcp]:
                lcp += 1
            if lcp == 0:
                continue
            cp = 0
            if lcp == s_pos:
                cp = lcp
            else:
                # Ring-buffer invariant: rollback to cp is exact iff s_pos - cp < SWA_WIN (1024)!
                valid = [k for k in sc["snaps"].keys() if k <= lcp and (s_pos - k) < SWA_WIN]
                if valid:
                    cp = max(valid)
            if cp > best_cp:
                best_cp, best_slot = cp, s

        t_r0 = time.perf_counter()
        if best_slot >= 0 and best_cp > 0:
            slot = best_slot
            sc = self.slot_cache[slot]
            if best_cp != sc["pos"]:
                self.eng.restore_slot_conv(slot, sc["snaps"][best_cp])
            sc["tokens"] = sc["tokens"][:best_cp]
            sc["pos"] = best_cp
            for k in [k for k in sc["snaps"] if k > best_cp]:
                del sc["snaps"][k]
            cp = best_cp
        else:
            # Evict LRU idle slot
            slot = min(self.idle_slots, key=lambda s: self.slot_cache[s]["last_used"])
            self.eng.reset_slot(slot)
            self.slot_cache[slot] = {"tokens": [], "pos": 0, "snaps": {}, "last_used": time.time()}
            cp = 0

        self.idle_slots.remove(slot)
        restore_ms = (time.perf_counter() - t_r0) * 1e3
        return slot, cp, restore_ms

    def _release_slot(self, slot: int, timed_out: bool = False):
        sc = self.slot_cache[slot]
        sc["last_used"] = time.time()
        if not timed_out:
            self._save_snap(slot)
        self.idle_slots.add(slot)

    def _admit_pending(self):
        now = time.perf_counter()
        while self.idle_slots:
            try:
                req = self.pending.get_nowait()
            except queue.Empty:
                break
            if now > req.deadline:
                req.out_q.put(("timeout", "Request timed out in queue", None))
                continue
            slot, cp, restore_ms = self._pick_and_prepare_slot(req.prompt_ids)
            req.slot = slot
            req.pos = cp
            req.cur = req.prompt_ids[cp]
            req.cached_tokens = cp
            req.restore_ms = restore_ms
            self.active.append(req)

    def _loop(self):
        while not self.stop_flag:
            self._admit_pending()
            if not self.active:
                self.wake.wait(timeout=0.02)
                self.wake.clear()
                continue

            now = time.perf_counter()
            live = []
            for r in self.active:
                if now > r.deadline:
                    r.out_q.put(("timeout", "Request timed out during generation", None))
                    self._release_slot(r.slot, timed_out=True)
                else:
                    live.append(r)
            self.active = live
            if not self.active:
                continue

            # Chunked prefill (Fix 1 + Fix 2 + Fix 4 + Fix 5):
            # Advance any newcomer with remaining prefill tokens in fast C=64/16 chunks (no output head!),
            # interleaving with active decoders so streams stay responsive.
            prefill_req = None
            for r in self.active:
                rem = (len(r.prompt_ids) - 1) - r.pos
                if rem >= 4:
                    prefill_req = r
                    break
            if prefill_req is not None:
                r = prefill_req
                rem = (len(r.prompt_ids) - 1) - r.pos
                chunk_len = min(64, rem)
                chunk = r.prompt_ids[r.pos : r.pos + chunk_len]
                self.eng.prefill_chunk(r.slot, chunk, r.pos)
                sc = self.slot_cache[r.slot]
                sc["tokens"].extend(chunk)
                r.pos += chunk_len
                sc["pos"] = r.pos
                r.cur = r.prompt_ids[r.pos]
                self.tok_events.append((time.perf_counter(), chunk_len))
                # Save conv snapshot after chunk for prefix caching & rollback!
                self._save_snap(r.slot)
                # If no other requests are ready to decode, loop immediately to finish prefill fast!
                decoders = [x for x in self.active if (len(x.prompt_ids) - 1) - x.pos < 4]
                if not decoders:
                    continue
                step_list = decoders
            else:
                step_list = self.active

            slots  = [r.slot for r in step_list]
            tids   = [r.cur  for r in step_list]
            poss   = [r.pos  for r in step_list]
            temps  = [r.temperature if r.pos >= len(r.prompt_ids) - 1 else 0.0 for r in step_list]
            top_ps = [r.top_p for r in step_list]
            seeds  = [r.seed  for r in step_list]

            outs = self.eng.step_batch(slots, tids, poss, temps=temps, top_ps=top_ps, seeds=seeds)
            self.tok_events.append((time.perf_counter(), len(slots)))

            stepped_set = set(id(x) for x in step_list)
            still_active = [x for x in self.active if id(x) not in stepped_set]
            for r, (nxt, _) in zip(step_list, outs):
                sc = self.slot_cache[r.slot]
                sc["tokens"].append(r.cur)
                r.pos += 1
                sc["pos"] = r.pos

                # Still prefilling prompt?
                if r.pos < len(r.prompt_ids):
                    if r.cur in (TOK_END_MSG, TOK_END_SAMPLE) or (r.pos % 64 == 0):
                        self._save_snap(r.slot)
                    r.cur = r.prompt_ids[r.pos]
                    still_active.append(r)
                    continue

                if r.cur in (TOK_END_MSG, TOK_END_SAMPLE):
                    self._save_snap(r.slot)

                if r.t_first is None:
                    r.t_first = time.perf_counter()
                r.n_gen += 1
                r.cur = nxt
                done_reason = None

                if nxt in STOP_TOKEN_IDS:
                    done_reason = "stop"
                elif r.parse_chat and nxt in SPECIAL_CTRL:
                    if nxt == TOK_THINK: r.mode = "thinking"
                    elif nxt == TOK_TEXT: r.mode = "text"
                else:
                    piece = self.tok.decode([nxt], skip_special_tokens=False)
                    if not r.parse_chat or r.mode != "thinking":
                        r.text_accum += piece
                        if any(s and s in r.text_accum for s in r.stop_strings):
                            done_reason = "stop"
                        else:
                            r.out_q.put(("content", piece, None))
                    else:
                        r.out_q.put(("reasoning", piece, None))

                if done_reason is None and (r.n_gen >= r.max_tokens or r.pos >= SEQ_CAP - 1):
                    done_reason = "length"

                if done_reason is not None:
                    t_end = time.perf_counter()
                    ttft_ms = ((r.t_first or t_end) - r.t_arrived) * 1e3
                    dec_s = max(1e-6, t_end - (r.t_first or r.t_arrived))
                    dec_tps = max(0, r.n_gen - 1) / dec_s if r.n_gen > 1 else r.n_gen / dec_s
                    self._release_slot(r.slot, timed_out=False)
                    self._record_completed(len(r.prompt_ids), r.cached_tokens, r.n_gen)
                    r.out_q.put(("done", done_reason, {
                        "prompt_tokens": len(r.prompt_ids),
                        "cached_tokens": r.cached_tokens,
                        "prefill_tokens": len(r.prompt_ids) - r.cached_tokens,
                        "restore_ms": round(r.restore_ms, 2),
                        "completion_tokens": r.n_gen,
                        "total_tokens": len(r.prompt_ids) + r.n_gen,
                        "ttft_ms": round(ttft_ms, 2),
                        "decode_tok_s": round(dec_tps, 2),
                    }))
                else:
                    still_active.append(r)

            self.active = still_active

    def close(self):
        self.stop_flag = True
        self.wake.set()
        self.thread.join(timeout=2)
        self.eng.close()

SCHED = None
API_KEY = None
REQ_TIMEOUT = 60.0

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args): pass

    def _send_json(self, code, obj, extra_headers=None):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _check_auth(self) -> bool:
        if not API_KEY:
            return True
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer ") and hmac.compare_digest(auth[7:], API_KEY):
            return True
        self._send_json(401, {"error": {"message": "Invalid or missing API key", "type": "authentication_error"}})
        return False

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {
                "status": "ok", "model": MODEL_ID,
                "max_batch": MAX_BATCH, "seq_cap": SEQ_CAP, "swa_window": SWA_WIN,
                "active": len(SCHED.active), "queued": SCHED.pending.qsize(), "max_queue": SCHED.max_queue,
            })
        elif self.path == "/stats":
            self._send_json(200, SCHED.get_stats())
        elif self.path == "/v1/models":
            if not self._check_auth(): return
            self._send_json(200, {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local-vk-batch"}]})
        else:
            self._send_json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if not self._check_auth(): return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            req_j = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except Exception:
            self._send_json(400, {"error": {"message": "invalid JSON"}}); return

        if self.path == "/v1/chat/completions":
            prompt = format_chat_prompt(req_j.get("messages", []), reasoning_effort=req_j.get("reasoning_effort", "none"))
            prompt_ids = SCHED.tok.encode(prompt, add_special_tokens=False).ids
            parse_chat = True
        elif self.path == "/v1/completions":
            prompt_ids = SCHED.tok.encode(req_j.get("prompt", ""), add_special_tokens=False).ids
            parse_chat = False
        else:
            self._send_json(404, {"error": {"message": "not found"}}); return

        if not prompt_ids:
            self._send_json(400, {"error": {"message": "empty prompt"}}); return

        max_tokens  = int(req_j.get("max_tokens", 256))
        temperature = float(req_j.get("temperature", 0.0))
        top_p       = float(req_j.get("top_p", 1.0))
        seed        = req_j.get("seed", None)
        stop        = req_j.get("stop") or []
        if isinstance(stop, str): stop = [stop]
        stream      = bool(req_j.get("stream", False))
        timeout_s   = min(REQ_TIMEOUT, float(req_j.get("timeout", REQ_TIMEOUT)))
        deadline    = time.perf_counter() + timeout_s
        req_id      = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created     = int(time.time())

        ar = ActiveReq(prompt_ids, max_tokens, temperature, top_p, seed, stop, parse_chat, deadline)
        if not SCHED.submit(ar):
            self._send_json(429, {"error": {"message": "Server queue full", "type": "rate_limit_error"}}, {"Retry-After": "1"})
            return

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            while True:
                rem = max(0.05, deadline - time.perf_counter() + 1.0)
                try:
                    kind, payload, meta = ar.out_q.get(timeout=rem)
                except queue.Empty:
                    kind, payload, meta = "timeout", "Request timed out", None
                if kind == "content":
                    chunk = {
                        "id": req_id, "object": "chat.completion.chunk" if parse_chat else "text_completion",
                        "created": created, "model": MODEL_ID,
                        "choices": [{"index": 0, "delta": {"content": payload}, "finish_reason": None}] if parse_chat else [{"index": 0, "text": payload, "finish_reason": None}],
                    }
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode()); self.wfile.flush()
                elif kind == "reasoning":
                    chunk = {
                        "id": req_id, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
                        "choices": [{"index": 0, "delta": {"reasoning_content": payload}, "finish_reason": None}],
                    }
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode()); self.wfile.flush()
                elif kind == "timeout":
                    chunk = {"error": {"message": payload, "type": "timeout_error"}}
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
                    break
                elif kind == "done":
                    chunk = {
                        "id": req_id, "object": "chat.completion.chunk" if parse_chat else "text_completion",
                        "created": created, "model": MODEL_ID,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": payload}] if parse_chat else [{"index": 0, "text": "", "finish_reason": payload}],
                        "usage": {k: meta[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
                        "timings": {k: meta[k] for k in ("ttft_ms", "decode_tok_s", "cached_tokens", "prefill_tokens", "restore_ms")},
                    }
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
                    break
        else:
            c_parts, r_parts = [], []
            finish, meta = "stop", {}
            while True:
                rem = max(0.05, deadline - time.perf_counter() + 1.0)
                try:
                    kind, payload, m = ar.out_q.get(timeout=rem)
                except queue.Empty:
                    kind, payload, m = "timeout", "Request timed out", None
                if kind == "content": c_parts.append(payload)
                elif kind == "reasoning": r_parts.append(payload)
                elif kind == "timeout":
                    self._send_json(504, {"error": {"message": payload, "type": "timeout_error"}})
                    return
                elif kind == "done":
                    finish, meta = payload, m; break
            if parse_chat:
                msg = {"role": "assistant", "content": "".join(c_parts)}
                if r_parts: msg["reasoning_content"] = "".join(r_parts)
                resp = {
                    "id": req_id, "object": "chat.completion", "created": created, "model": MODEL_ID,
                    "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                    "usage": {k: meta[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
                    "timings": {k: meta[k] for k in ("ttft_ms", "decode_tok_s", "cached_tokens", "prefill_tokens", "restore_ms")},
                }
            else:
                resp = {
                    "id": req_id, "object": "text_completion", "created": created, "model": MODEL_ID,
                    "choices": [{"index": 0, "text": "".join(c_parts), "finish_reason": finish}],
                    "usage": {k: meta[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
                    "timings": {k: meta[k] for k in ("ttft_ms", "decode_tok_s", "cached_tokens", "prefill_tokens", "restore_ms")},
                }
            self._send_json(200, resp)

def main():
    global SCHED, API_KEY, REQ_TIMEOUT
    ap = argparse.ArgumentParser(description="Inkling-2-Mini Continuous-Batching Raw Vulkan Server")
    ap.add_argument("--model-dir", default=os.environ.get("INKLING_MODEL_DIR", "./Inkling-2-Mini"), help="Path to Inkling-2-Mini weights directory (or INKLING_MODEL_DIR env)")
    ap.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--api-key", default=os.environ.get("INKLING_API_KEY", ""), help="Bearer API key (or INKLING_API_KEY env)")
    ap.add_argument("--max-queue", type=int, default=32, help="Max queued requests beyond 16 active slots (429 when full)")
    ap.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout in seconds (504 on timeout)")
    args = ap.parse_args()
    API_KEY = args.api_key or None
    REQ_TIMEOUT = args.timeout
    SCHED = ContinuousBatchScheduler(model_dir=args.model_dir, max_queue=args.max_queue)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[serve_batch_vk] Listening on http://{args.host}:{args.port} | auth={'ON' if API_KEY else 'OFF'} | timeout={REQ_TIMEOUT}s", flush=True)
    try: srv.serve_forever()
    except KeyboardInterrupt: pass
    finally: SCHED.close()

if __name__ == "__main__":
    main()
