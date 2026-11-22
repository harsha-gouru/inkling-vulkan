import argparse, hmac, json, os, queue, sys, threading, time, uuid
from collections import deque
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from tokenizers import Tokenizer

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
from inkling_batch_vk import BatchedVulkanEngine, MAX_BATCH, SEQ_CAP, SWA_WIN
from inkling_vit import InklingViT

MODEL_ID = "Inkling-2-Mini"

EFFORT_MAP = {"none": 0.0, "minimal": 0.1, "low": 0.2, "medium": 0.7, "high": 0.9, "xhigh": 0.99, "max": 0.99}
ROLE_TOK   = {"user": "<|message_user|>", "assistant": "<|message_model|>", "system": "<|message_system|>", "tool": "<|message_tool|>"}

TOK_EOT        = 199999  # <|endoftext|>
TOK_MSG_MODEL  = 200001  # <|message_model|>
TOK_TEXT       = 200004  # <|content_text|>
TOK_END_SAMPLE = 200006  # <|content_model_end_sampling|>
TOK_THINK      = 200008  # <|content_thinking|>
TOK_END_MSG    = 200010  # <|end_message|>
TOK_IMAGE_PH   = 200054  # <|unused_200054|>
STOP_TOKEN_IDS = {TOK_EOT, TOK_END_SAMPLE}
SPECIAL_CTRL   = {TOK_MSG_MODEL, TOK_TEXT, TOK_THINK, TOK_END_MSG, TOK_EOT, TOK_END_SAMPLE}

def _canon_tool_args(raw):
    if isinstance(raw, dict):
        obj = raw
    elif isinstance(raw, str):
        raw_s = raw.strip()
        if not raw_s:
            obj = {}
        else:
            try:
                parsed = json.loads(raw_s)
                obj = parsed if isinstance(parsed, dict) else {"value": parsed}
            except Exception:
                obj = {}
    else:
        obj = {}
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)

def _extract_image_url(part):
    if not isinstance(part, dict):
        return None
    ptype = part.get("type", "")
    if ptype == "image_url":
        iu = part.get("image_url")
        if isinstance(iu, dict):
            return iu.get("url")
        if isinstance(iu, str):
            return iu
    elif ptype in ("image", "input_image"):
        for k in ("image_url", "url", "image", "data"):
            v = part.get(k)
            if isinstance(v, dict) and v.get("url"):
                return v["url"]
            if isinstance(v, str) and v:
                return v
    return None

def format_chat_prompt_segments(messages, reasoning_effort=0.9, tools=None):
    """Return list of segments: ("text", str), ("image", url), or ("assistant", msg, fallback_str)."""
    if isinstance(reasoning_effort, str):
        eff = EFFORT_MAP.get(reasoning_effort.strip().lower(), 0.9)
    elif reasoning_effort is None:
        eff = 0.9
    else:
        eff = float(reasoning_effort)
    eff_str = "0" if eff == 0.0 else str(eff)
    effort_block = f"<|message_system|><|content_text|>Thinking effort level: {eff_str}<|end_message|>"

    segs = []
    def emit_text(s):
        if not s:
            return
        if segs and segs[-1][0] == "text":
            segs[-1] = ("text", segs[-1][1] + s)
        else:
            segs.append(("text", s))

    if tools:
        specs = []
        for tool in tools:
            fn = tool.get("function") if isinstance(tool, dict) and "function" in tool else tool
            if not isinstance(fn, dict) or not fn.get("name"):
                continue
            specs.append({
                "description": fn.get("description") or "",
                "name": fn["name"],
                "parameters": fn.get("parameters") or {},
                "type": tool.get("type") or "function",
            })
        if specs:
            specs_json = json.dumps(specs, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            emit_text(f"<|message_system|>tool_declare<|content_xml|>{specs_json}<|end_message|>")

    tc_id_to_name = {}
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                tid = tc.get("id")
                fn = tc.get("function") or {}
                if tid and fn.get("name"):
                    tc_id_to_name[tid] = fn["name"]

    effort_emitted = False
    for m in messages:
        role = m.get("role", "user")
        if role == "developer":
            role = "system"
        if not effort_emitted and role != "system":
            emit_text(effort_block)
            effort_emitted = True
        rtok = ROLE_TOK.get(role, "<|message_user|>")
        if role == "tool":
            tname = m.get("name") or tc_id_to_name.get(m.get("tool_call_id", ""), "")
            rtok = rtok + tname
        elif role == "assistant" and m.get("name"):
            rtok = rtok + m["name"]

        if role == "assistant":
            aparts = []
            if m.get("reasoning_content"):
                aparts.append(f"{rtok}<|content_thinking|>{m['reasoning_content']}<|end_message|>")
            content = m.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, str):
                        aparts.append(f"{rtok}<|content_text|>{part}<|end_message|>")
                    elif isinstance(part, dict):
                        ptype = part.get("type", "text")
                        if ptype in ("text", "input_text"):
                            aparts.append(f"{rtok}<|content_text|>{part.get('text', '')}<|end_message|>")
            else:
                content_s = content or ""
                has_other = bool(m.get("tool_calls") or m.get("reasoning_content"))
                if content_s or not has_other:
                    aparts.append(f"{rtok}<|content_text|>{content_s}<|end_message|>")
            if m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    fn = tc.get("function") or {}
                    fname = fn.get("name", "")
                    if not fname:
                        continue
                    args_json = _canon_tool_args(fn.get("arguments", {}))
                    fname_json = json.dumps(fname, ensure_ascii=False)
                    aparts.append(
                        f"<|message_model|>{fname}<|content_invoke_tool_json|>"
                        f'{{"name":{fname_json},"args":{args_json}}}<|end_message|>'
                    )
            aparts.append("<|content_model_end_sampling|>")
            segs.append(("assistant", m, "".join(aparts)))
            continue

        content = m.get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, str):
                    emit_text(f"{rtok}<|content_text|>{part}<|end_message|>")
                elif isinstance(part, dict):
                    ptype = part.get("type", "text")
                    if ptype in ("text", "input_text"):
                        emit_text(f"{rtok}<|content_text|>{part.get('text', '')}<|end_message|>")
                    elif ptype in ("image", "input_image", "image_url"):
                        url = _extract_image_url(part)
                        if url and role in ("user", "tool"):
                            emit_text(f"{rtok}<|content_image|>")
                            segs.append(("image", url))
                            emit_text("<|end_message|>")
        else:
            content_s = content or ""
            emit_text(f"{rtok}<|content_text|>{content_s}<|end_message|>")

    if not effort_emitted:
        emit_text(effort_block)
    if eff == 0.0 and not tools:
        emit_text("<|message_model|><|content_text|>")
    return segs

def format_chat_prompt(messages, reasoning_effort=0.9, tools=None):
    out = []
    for seg in format_chat_prompt_segments(messages, reasoning_effort=reasoning_effort, tools=tools):
        if seg[0] == "text":
            out.append(seg[1])
        elif seg[0] == "image":
            out.append("<|unused_200054|>")
        elif seg[0] == "assistant":
            out.append(seg[2])
    return "".join(out)

TOK_MSG_USER         = 200000
TOK_INVOKE_TOOL_JSON = 200049
TOK_INVOKE_TOOL_TEXT = 200057

def _norm_tool_args_fp(raw):
    if isinstance(raw, dict):
        obj = raw
    elif isinstance(raw, str):
        s = raw.strip()
        if not s:
            obj = {}
        else:
            try:
                obj = json.loads(s)
            except Exception:
                return s
    else:
        obj = {}
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def _asst_msg_fingerprint(reasoning_s, content, tool_calls):
    if isinstance(content, list):
        c_parts = []
        for p in content:
            if isinstance(p, str):
                c_parts.append(p)
            elif isinstance(p, dict) and p.get("type", "text") in ("text", "input_text"):
                c_parts.append(p.get("text", ""))
        c_str = "".join(c_parts)
    else:
        c_str = content or ""
    tc_fp = []
    for tc in (tool_calls or []):
        fn = tc.get("function") or {}
        tc_fp.append((fn.get("name", ""), _norm_tool_args_fp(fn.get("arguments", {}))))
    return (reasoning_s or "", c_str, tuple(tc_fp))

class ActiveReq:
    def __init__(self, prompt_ids, max_tokens, temperature, top_p, seed, stop_strings, parse_chat, deadline,
                 cache_keys=None, embed_overrides=None):
        self.prompt_ids = prompt_ids
        self.cache_keys = cache_keys if cache_keys is not None else list(prompt_ids)
        self.embed_overrides = embed_overrides or {}
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
        self.mode = "text" if (parse_chat and prompt_ids and prompt_ids[-1] == TOK_TEXT) else "header"
        self.header_buf = ""
        self.tool_buf = ""
        self.tool_calls = []
        self.tc_active = None  # dict when streaming an in-progress tool_call
        self.n_gen = 0
        self.gen_ids = []
        self.reasoning_acc = []
        self.content_acc = []
        self.text_accum = ""
        self.sys_prefix_len = 0
        self.t_arrived = time.perf_counter()
        self.t_first = None

class ContinuousBatchScheduler:
    def __init__(self, model_dir, max_queue=32, max_batch=None, seq_cap=None):
        model_dir = Path(model_dir)
        model_path = model_dir / "model-00001-of-00001.safetensors"
        tok_path = model_dir / "tokenizer.json"
        if not model_path.is_file() or not tok_path.is_file():
            raise FileNotFoundError(
                f"Inkling-2-Mini weights not found in {model_dir}.\n"
                f"Download them with:\n  hf download thinkingmachines/Inkling-2-Mini --local-dir {model_dir}"
            )
        self.tok = Tokenizer.from_file(str(tok_path))
        self.vit = InklingViT(safetensors_path=str(model_path), max_pixels=1024 * 1024, cache_size=16, lazy=True)
        self.asst_token_cache = {}  # key -> (gen_ids, fingerprint), bounded LRU (64 entries)
        self.asst_cache_lock = threading.Lock()
        self.eng = BatchedVulkanEngine(str(model_path), max_batch=max_batch, seq_cap=seq_cap)
        self.max_queue = max_queue
        self.pending = queue.Queue(maxsize=max_queue)
        self.wake = threading.Event()
        self.stop_flag = False
        # Per-slot LRU prompt cache state: idle slots stay warm!
        self.slot_cache = {
            s: {"tokens": [], "pos": 0, "snaps": {}, "last_used": 0.0}
            for s in range(self.eng.max_batch)
        }
        self.idle_slots = set(range(self.eng.max_batch))
        self.vram_prefixes = [{"keys": [], "pos": 0, "last_used": 0.0, "hits": 0} for _ in range(getattr(self.eng, "n_prefix_slots", 0))]
        self.active = []
        self.stats_path = str(BASE_DIR / "usage_stats.json")
        self.tok_events = deque()  # (timestamp, n_tokens) for rolling 2s live tok/s
        self.session_stats = {"requests": 0, "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0}
        self.persisted = self._load_stats()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        print(f"[serve_batch_vk] Ready: {MODEL_ID} | MAX_BATCH={self.eng.max_batch} | SEQ_CAP={self.eng.seq_cap} (SWA ring={SWA_WIN}) | MTP={'ON' if getattr(self.eng, 'has_mtp', False) else 'OFF'} | max_queue={max_queue} | init={self.eng.init_s:.2f}s", flush=True)

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
            "max_batch": self.eng.max_batch, "max_queue": self.max_queue,
            "live_tps": round(live_tps, 1),
            "kv_used_tokens": kv_used, "kv_cap_tokens": self.eng.max_batch * self.eng.seq_cap,
            "warm_slots": warm_slots,
            "vram_prefixes": [{"pos": vp["pos"], "hits": vp["hits"]} for vp in self.vram_prefixes if vp["pos"] > 0],
            "session": self.session_stats,
            "today": day,
            "all_time": self.persisted["all_time"],
        }

    def _cache_asst_turn(self, r: ActiveReq):
        if not r.parse_chat or not r.gen_ids or r.gen_ids[-1] != TOK_END_SAMPLE:
            return
        fp = _asst_msg_fingerprint("".join(r.reasoning_acc), "".join(r.content_acc), r.tool_calls)
        if len(r.prompt_ids) >= 2 and r.prompt_ids[-2] == TOK_MSG_MODEL and r.prompt_ids[-1] == TOK_TEXT:
            ids_copy = [TOK_MSG_MODEL, TOK_TEXT] + list(r.gen_ids)
        else:
            ids_copy = list(r.gen_ids)
        keys = []
        if r.tool_calls:
            keys.append(("tc", tuple(tc["id"] for tc in r.tool_calls)))
        keys.append(("fp", fp))
        with self.asst_cache_lock:
            for k in keys:
                self.asst_token_cache.pop(k, None)
                self.asst_token_cache[k] = (ids_copy, fp)
            while len(self.asst_token_cache) > 128:
                self.asst_token_cache.pop(next(iter(self.asst_token_cache)))

    def encode_chat_prompt(self, messages, reasoning_effort="none", tools=None):
        segs = format_chat_prompt_segments(messages, reasoning_effort=reasoning_effort, tools=tools)
        prompt_ids = []
        cache_keys = []
        embed_overrides = {}
        for seg in segs:
            kind = seg[0]
            if kind == "text":
                ids = self.tok.encode(seg[1], add_special_tokens=False).ids
                prompt_ids.extend(ids)
                cache_keys.extend(ids)
            elif kind == "image":
                sha_hex, emb = self.vit.encode_data_url(seg[1])
                base = len(prompt_ids)
                n_img = emb.shape[0]
                prompt_ids.extend([TOK_IMAGE_PH] * n_img)
                for j in range(n_img):
                    cache_keys.append((sha_hex, j))
                    embed_overrides[base + j] = emb[j]
            elif kind == "assistant":
                m, fallback_text = seg[1], seg[2]
                fp = _asst_msg_fingerprint(m.get("reasoning_content", ""), m.get("content", ""), m.get("tool_calls") or [])
                cached_ids = None
                with self.asst_cache_lock:
                    if m.get("tool_calls"):
                        tc_key = ("tc", tuple(tc.get("id") for tc in m["tool_calls"]))
                        hit = self.asst_token_cache.get(tc_key)
                        if hit and hit[1] == fp:
                            cached_ids = hit[0]
                    if cached_ids is None:
                        hit = self.asst_token_cache.get(("fp", fp))
                        if hit and hit[1] == fp:
                            cached_ids = hit[0]
                ids = cached_ids if cached_ids is not None else self.tok.encode(fallback_text, add_special_tokens=False).ids
                prompt_ids.extend(ids)
                cache_keys.extend(ids)
        return prompt_ids, cache_keys, embed_overrides

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

        # Check GPU VRAM L1 Prefix Cache Pool for an instant (~1ms) system-prompt hit!
        best_vp, best_vp_pos = -1, 0
        for pi, vp in enumerate(self.vram_prefixes):
            vpos = vp["pos"]
            if vpos > best_vp_pos and vpos < len(prompt_ids) and prompt_ids[:vpos] == vp["keys"]:
                best_vp, best_vp_pos = pi, vpos

        t_r0 = time.perf_counter()
        if best_slot >= 0 and best_cp >= best_vp_pos and best_cp > 0:
            slot = best_slot
            sc = self.slot_cache[slot]
            if best_cp != sc["pos"]:
                self.eng.restore_slot_conv(slot, sc["snaps"][best_cp])
            sc["tokens"] = sc["tokens"][:best_cp]
            sc["pos"] = best_cp
            for k in [k for k in sc["snaps"] if k > best_cp]:
                del sc["snaps"][k]
            cp = best_cp
        elif best_vp >= 0 and best_vp_pos > 0:
            slot = min(self.idle_slots, key=lambda s: self.slot_cache[s]["last_used"])
            vp = self.vram_prefixes[best_vp]
            self.eng.load_vram_prefix(slot, best_vp, best_vp_pos)
            vp["last_used"] = time.time()
            vp["hits"] += 1
            self.slot_cache[slot] = {"tokens": list(vp["keys"]), "pos": best_vp_pos, "snaps": {}, "last_used": time.time()}
            self._save_snap(slot)
            cp = best_vp_pos
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
            slot, cp, restore_ms = self._pick_and_prepare_slot(req.cache_keys)
            req.slot = slot
            req.pos = cp
            req.cur = req.prompt_ids[cp]
            req.cached_tokens = cp
            req.restore_ms = restore_ms
            # Detect system/tools prefix boundary for auto-saving into GPU VRAM prefix pool
            if self.vram_prefixes and req.parse_chat:
                p_cap = getattr(self.eng, "prefix_cap", 32768)
                try:
                    u_idx = req.prompt_ids.index(TOK_MSG_USER)
                except ValueError:
                    u_idx = 0
                if 512 <= u_idx <= p_cap and u_idx < len(req.prompt_ids) - 1:
                    # Only capture if not already stored in vram_prefixes
                    if not any(vp["pos"] == u_idx and vp["keys"] == req.cache_keys[:u_idx] for vp in self.vram_prefixes):
                        req.sys_prefix_len = u_idx
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
                if r.sys_prefix_len > r.pos and r.pos + chunk_len > r.sys_prefix_len:
                    chunk_len = r.sys_prefix_len - r.pos
                chunk = r.prompt_ids[r.pos : r.pos + chunk_len]
                chunk_keys = r.cache_keys[r.pos : r.pos + chunk_len]
                chunk_ov = None
                if r.embed_overrides:
                    chunk_ov = {i: r.embed_overrides[r.pos + i] for i in range(chunk_len) if (r.pos + i) in r.embed_overrides}
                    if not chunk_ov:
                        chunk_ov = None
                self.eng.prefill_chunk(r.slot, chunk, r.pos, embed_overrides=chunk_ov)
                sc = self.slot_cache[r.slot]
                sc["tokens"].extend(chunk_keys)
                r.pos += chunk_len
                sc["pos"] = r.pos
                r.cur = r.prompt_ids[r.pos]
                self.tok_events.append((time.perf_counter(), chunk_len))
                # Save conv snapshot after chunk for prefix caching & rollback!
                self._save_snap(r.slot)
                if r.sys_prefix_len > 0 and r.pos == r.sys_prefix_len:
                    pi = min(range(len(self.vram_prefixes)), key=lambda i: (self.vram_prefixes[i]["pos"] > 0, self.vram_prefixes[i]["last_used"]))
                    ms = self.eng.save_vram_prefix(r.slot, pi, r.pos)
                    self.vram_prefixes[pi] = {"keys": list(r.cache_keys[:r.pos]), "pos": r.pos, "last_used": time.time(), "hits": 0}
                    print(f"[serve_batch_vk] Saved VRAM prefix slot {pi} ({r.pos} toks) in {ms:.2f} ms", flush=True)
                    r.sys_prefix_len = 0
                # If no other requests are ready to decode, loop immediately to finish prefill fast!
                decoders = [x for x in self.active if (len(x.prompt_ids) - 1) - x.pos < 4]
                if not decoders:
                    continue
                step_list = decoders
            else:
                step_list = self.active

            use_spec = (
                len(step_list) == 1
                and getattr(self.eng, "has_mtp", False)
                and step_list[0].pos >= len(step_list[0].prompt_ids)
                and step_list[0].temperature <= 0.0
                and step_list[0].pos < self.eng.seq_cap - 9
                and (step_list[0].max_tokens - step_list[0].n_gen) >= 2
            )
            if use_spec:
                r0 = step_list[0]
                spec_outs = self.eng.step_speculative(r0.slot, r0.cur, r0.pos, adaptive=True)
                self.tok_events.append((time.perf_counter(), len(spec_outs)))
                batch_results = [(r0, spec_outs)]
            else:
                slots  = [r.slot for r in step_list]
                tids   = [r.cur  for r in step_list]
                poss   = [r.pos  for r in step_list]
                temps  = [r.temperature if r.pos >= len(r.prompt_ids) - 1 else 0.0 for r in step_list]
                top_ps = [r.top_p for r in step_list]
                seeds  = [r.seed  for r in step_list]

                batch_ov = None
                for bi, r in enumerate(step_list):
                    if r.embed_overrides and r.pos in r.embed_overrides:
                        if batch_ov is None:
                            batch_ov = {}
                        batch_ov[bi] = r.embed_overrides[r.pos]
                outs = self.eng.step_batch(slots, tids, poss, temps=temps, top_ps=top_ps, seeds=seeds, embed_overrides=batch_ov)
                self.tok_events.append((time.perf_counter(), len(slots)))
                batch_results = [(r, [o]) for r, o in zip(step_list, outs)]

            stepped_set = set(id(x) for x in step_list)
            still_active = [x for x in self.active if id(x) not in stepped_set]
            for r, token_outs in batch_results:
                sc = self.slot_cache[r.slot]
                n_blk = len(token_outs)
                done_reason = None
                still_prefilling = False
                had_boundary = False
                for j, (nxt, _) in enumerate(token_outs):
                    cur_tok = r.cur
                    cur_key = r.cache_keys[r.pos] if r.pos < len(r.cache_keys) else cur_tok
                    sc["tokens"].append(cur_key)
                    r.pos += 1
                    sc["pos"] = r.pos

                    # Still prefilling prompt?
                    if r.pos < len(r.prompt_ids):
                        if cur_tok in (TOK_END_MSG, TOK_END_SAMPLE) or (r.pos % 64 == 0):
                            self._save_snap(r.slot)
                        r.cur = r.prompt_ids[r.pos]
                        still_prefilling = True
                        break

                    if cur_tok in (TOK_END_MSG, TOK_END_SAMPLE):
                        had_boundary = True

                    if r.t_first is None:
                        r.t_first = time.perf_counter()
                    r.n_gen += 1
                    r.gen_ids.append(nxt)
                    r.cur = nxt
                    done_reason = None

                    def _feed_tool_piece(piece):
                        r.tool_buf += piece
                        tc = r.tc_active
                        if tc is None:
                            tc = {
                                "index": len(r.tool_calls),
                                "id": f"call_{uuid.uuid4().hex[:12]}",
                                "name": r.header_buf.strip(),
                                "started": False,
                                "done": False,
                                "depth": 0,
                                "in_str": False,
                                "esc": False,
                                "args_acc": [],
                                "scan_pos": 0,
                            }
                            r.tc_active = tc
                        if tc["done"]:
                            return
                        if not tc["started"]:
                            # Wait until {"name":"...","args": (or "arguments":) appears
                            buf = r.tool_buf
                            idx = buf.find('"args":')
                            key_len = 7
                            if idx < 0:
                                idx = buf.find('"arguments":')
                                key_len = 12
                            if idx < 0:
                                return
                            # Try extracting name from prefix if header_buf was empty
                            if not tc["name"]:
                                ni = buf.find('"name":')
                                if ni >= 0:
                                    try:
                                        q1 = buf.find('"', ni + 7)
                                        q2 = buf.find('"', q1 + 1)
                                        if q1 >= 0 and q2 > q1:
                                            tc["name"] = buf[q1 + 1 : q2]
                                    except Exception:
                                        pass
                            if not tc["name"]:
                                return
                            tc["started"] = True
                            tc["scan_pos"] = idx + key_len
                            r.out_q.put(("tool_call", {
                                "index": tc["index"],
                                "id": tc["id"],
                                "type": "function",
                                "function": {"name": tc["name"], "arguments": ""},
                            }, None))
                        # Stream characters inside the args JSON object live!
                        buf = r.tool_buf
                        pos = tc["scan_pos"]
                        out_chars = []
                        while pos < len(buf) and not tc["done"]:
                            ch = buf[pos]
                            pos += 1
                            if tc["depth"] == 0:
                                if ch in " \t\r\n":
                                    continue
                                if ch == "{":
                                    tc["depth"] = 1
                                    out_chars.append(ch)
                                else:
                                    # Non-object args fallback; wait for end_message
                                    tc["done"] = True
                                    break
                            else:
                                out_chars.append(ch)
                                if tc["in_str"]:
                                    if tc["esc"]:
                                        tc["esc"] = False
                                    elif ch == "\\":
                                        tc["esc"] = True
                                    elif ch == '"':
                                        tc["in_str"] = False
                                else:
                                    if ch == '"':
                                        tc["in_str"] = True
                                    elif ch == "{":
                                        tc["depth"] += 1
                                    elif ch == "}":
                                        tc["depth"] -= 1
                                        if tc["depth"] == 0:
                                            tc["done"] = True
                        tc["scan_pos"] = pos
                        if out_chars:
                            chunk_s = "".join(out_chars)
                            tc["args_acc"].append(chunk_s)
                            r.out_q.put(("tool_call_delta", {
                                "index": tc["index"],
                                "function": {"arguments": chunk_s},
                            }, None))

                    def _flush_tool_call():
                        tc = r.tc_active
                        r.tc_active = None
                        raw = r.tool_buf.strip()
                        r.tool_buf = ""
                        if tc is not None and tc["started"]:
                            r.tool_calls.append({
                                "index": tc["index"],
                                "id": tc["id"],
                                "type": "function",
                                "function": {"name": tc["name"], "arguments": "".join(tc["args_acc"])},
                            })
                            return
                        fname = (tc["name"] if tc else "") or r.header_buf.strip()
                        args_obj = {}
                        if raw:
                            try:
                                obj = json.loads(raw)
                                if isinstance(obj, dict):
                                    fname = obj.get("name") or fname
                                    a = obj.get("args", obj.get("arguments", {}))
                                    if isinstance(a, str):
                                        try: a = json.loads(a)
                                        except Exception: pass
                                    args_obj = a if isinstance(a, dict) else {"value": a}
                            except Exception:
                                args_obj = {"raw": raw}
                        if fname:
                            tc_full = {
                                "index": len(r.tool_calls),
                                "id": f"call_{uuid.uuid4().hex[:12]}",
                                "type": "function",
                                "function": {
                                    "name": fname,
                                    "arguments": json.dumps(args_obj, ensure_ascii=False),
                                },
                            }
                            r.tool_calls.append(tc_full)
                            r.out_q.put(("tool_call", tc_full, None))

                    if nxt in STOP_TOKEN_IDS:
                        if r.parse_chat and r.mode == "tool_json" and r.tool_buf.strip():
                            _flush_tool_call()
                        done_reason = "tool_calls" if r.tool_calls else "stop"
                    elif r.parse_chat and (nxt in SPECIAL_CTRL or nxt in (TOK_INVOKE_TOOL_JSON, TOK_INVOKE_TOOL_TEXT)):
                        if nxt == TOK_THINK:
                            r.mode = "thinking"; r.header_buf = ""
                        elif nxt == TOK_TEXT:
                            r.mode = "text"; r.header_buf = ""
                        elif nxt in (TOK_INVOKE_TOOL_JSON, TOK_INVOKE_TOOL_TEXT):
                            r.mode = "tool_json"; r.tool_buf = ""
                        elif nxt == TOK_END_MSG:
                            if r.mode == "tool_json":
                                _flush_tool_call()
                            r.mode = "header"; r.header_buf = ""
                        elif nxt == TOK_MSG_MODEL:
                            r.mode = "header"; r.header_buf = ""; r.tool_buf = ""
                    else:
                        piece = self.tok.decode([nxt], skip_special_tokens=False)
                        if not r.parse_chat:
                            r.text_accum += piece
                            if any(s and s in r.text_accum for s in r.stop_strings):
                                done_reason = "stop"
                            else:
                                r.out_q.put(("content", piece, None))
                        elif r.mode == "thinking":
                            r.reasoning_acc.append(piece)
                            r.out_q.put(("reasoning", piece, None))
                        elif r.mode == "tool_json":
                            _feed_tool_piece(piece)
                        elif r.mode == "header":
                            r.header_buf += piece
                        else:
                            r.text_accum += piece
                            if any(s and s in r.text_accum for s in r.stop_strings):
                                done_reason = "stop"
                            else:
                                r.content_acc.append(piece)
                                r.out_q.put(("content", piece, None))


                    if done_reason is None and (r.n_gen >= r.max_tokens or r.pos >= self.eng.seq_cap - 1):
                        done_reason = "length"
                    if done_reason is not None:
                        # Keep slot_cache truthful to GPU KV/conv position if stopping mid-block
                        if j < n_blk - 1:
                            for rem_t, _ in token_outs[j:-1]:
                                sc["tokens"].append(rem_t)
                            sc["pos"] = len(sc["tokens"])
                        break

                if still_prefilling:
                    still_active.append(r)
                    continue

                if done_reason is not None:
                    t_end = time.perf_counter()
                    ttft_ms = ((r.t_first or t_end) - r.t_arrived) * 1e3
                    dec_s = max(1e-6, t_end - (r.t_first or r.t_arrived))
                    dec_tps = max(0, r.n_gen - 1) / dec_s if r.n_gen > 1 else r.n_gen / dec_s
                    self._cache_asst_turn(r)
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
                    if had_boundary or (r.pos % 256 < n_blk):
                        self._save_snap(r.slot)
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
                "max_batch": SCHED.eng.max_batch, "seq_cap": SCHED.eng.seq_cap, "swa_window": SWA_WIN,
                "mtp": bool(getattr(SCHED.eng, "has_mtp", False)),
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

        cache_keys = None
        embed_overrides = None
        if self.path == "/v1/chat/completions":
            try:
                prompt_ids, cache_keys, embed_overrides = SCHED.encode_chat_prompt(
                    req_j.get("messages", []),
                    reasoning_effort=req_j.get("reasoning_effort", "none"),
                    tools=req_j.get("tools"),
                )
            except Exception as e:
                self._send_json(400, {"error": {"message": f"Prompt/image encode error: {e}", "type": "invalid_request_error"}})
                return
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

        ar = ActiveReq(prompt_ids, max_tokens, temperature, top_p, seed, stop, parse_chat, deadline, cache_keys=cache_keys, embed_overrides=embed_overrides)
        if not SCHED.submit(ar):
            self._send_json(429, {"error": {"message": "Server queue full", "type": "rate_limit_error"}}, {"Retry-After": "1"})
            return

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
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
                elif kind in ("tool_call", "tool_call_delta"):
                    chunk = {
                        "id": req_id, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
                        "choices": [{"index": 0, "delta": {"tool_calls": [payload]}, "finish_reason": None}],
                    }
                    try:
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode()); self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        break
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
                        "usage": {**{k: meta[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")}, "prompt_tokens_details": {"cached_tokens": meta.get("cached_tokens", 0)}},
                        "timings": {k: meta[k] for k in ("ttft_ms", "decode_tok_s", "cached_tokens", "prefill_tokens", "restore_ms")},
                    }
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
                    break
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            c_parts, r_parts, tc_parts = [], [], []
            finish, meta = "stop", {}
            while True:
                rem = max(0.05, deadline - time.perf_counter() + 1.0)
                try:
                    kind, payload, m = ar.out_q.get(timeout=rem)
                except queue.Empty:
                    kind, payload, m = "timeout", "Request timed out", None
                if kind == "content": c_parts.append(payload)
                elif kind == "reasoning": r_parts.append(payload)
                elif kind == "tool_call": tc_parts.append({k: v for k, v in payload.items() if k != "index"})
                elif kind == "tool_call_delta":
                    if tc_parts: tc_parts[-1]["function"]["arguments"] += payload["function"]["arguments"]
                elif kind == "timeout":
                    self._send_json(504, {"error": {"message": payload, "type": "timeout_error"}})
                    return
                elif kind == "done":
                    finish, meta = payload, m; break
            if parse_chat:
                msg = {"role": "assistant", "content": "".join(c_parts) if c_parts or not tc_parts else None}
                if r_parts: msg["reasoning_content"] = "".join(r_parts)
                if tc_parts: msg["tool_calls"] = tc_parts
                resp = {
                    "id": req_id, "object": "chat.completion", "created": created, "model": MODEL_ID,
                    "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                    "usage": {**{k: meta[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")}, "prompt_tokens_details": {"cached_tokens": meta.get("cached_tokens", 0)}},
                    "timings": {k: meta[k] for k in ("ttft_ms", "decode_tok_s", "cached_tokens", "prefill_tokens", "restore_ms")},
                }
            else:
                resp = {
                    "id": req_id, "object": "text_completion", "created": created, "model": MODEL_ID,
                    "choices": [{"index": 0, "text": "".join(c_parts), "finish_reason": finish}],
                    "usage": {**{k: meta[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")}, "prompt_tokens_details": {"cached_tokens": meta.get("cached_tokens", 0)}},
                    "timings": {k: meta[k] for k in ("ttft_ms", "decode_tok_s", "cached_tokens", "prefill_tokens", "restore_ms")},
                }
            self._send_json(200, resp)

def main():
    global SCHED, API_KEY, REQ_TIMEOUT
    ap = argparse.ArgumentParser(description="Inkling-2-Mini Continuous-Batching Raw Vulkan Server")
    ap.add_argument("--model-dir", default=os.environ.get("INKLING_MODEL_DIR", "./Inkling-2-Mini"), help="Directory containing model-00001-of-00001.safetensors and tokenizer.json (or INKLING_MODEL_DIR env)")
    ap.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--api-key", default=os.environ.get("INKLING_API_KEY", ""), help="Bearer API key (or INKLING_API_KEY env)")
    ap.add_argument("--max-queue", type=int, default=32, help="Max queued requests beyond active slots (429 when full)")
    ap.add_argument("--timeout", type=float, default=600.0, help="Per-request timeout in seconds (504 on timeout)")
    ap.add_argument("--max-batch", type=int, default=MAX_BATCH, help="Concurrent GPU slots")
    ap.add_argument("--seq-cap", type=int, default=SEQ_CAP, help="Max context length per slot")
    args = ap.parse_args()
    API_KEY = args.api_key or None
    REQ_TIMEOUT = args.timeout
    SCHED = ContinuousBatchScheduler(model_dir=args.model_dir, max_queue=args.max_queue, max_batch=args.max_batch, seq_cap=args.seq_cap)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[serve_batch_vk] Listening on http://{args.host}:{args.port} | auth={'ON' if API_KEY else 'OFF'} | timeout={REQ_TIMEOUT}s", flush=True)
    try: srv.serve_forever()
    except KeyboardInterrupt: pass
    finally: SCHED.close()

if __name__ == "__main__":
    main()
