#!/usr/bin/env python3
"""OpenAI-compatible localhost server for Inkling-2-Mini (clean build, zero external deps)."""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tokenizer import InklingTokenizer

EFFORT_MAP = {
    "none": 0.0, "minimal": 0.1, "low": 0.2, "medium": 0.7,
    "high": 0.9, "xhigh": 0.99, "max": 0.99,
}
ROLE_TOKEN = {
    "user": "<|message_user|>",
    "assistant": "<|message_model|>",
    "system": "<|message_system|>",
    "tool": "<|message_tool|>",
}
EOS_ID = 200006  # <|content_model_end_sampling|>


def format_chat_prompt(messages, tools=None, reasoning_effort=0.0):
    """Exact port of ~/models/Inkling-2-Mini/chat_template.jinja."""
    parts = []
    if tools:
        specs = []
        for t in tools:
            fn = t.get("function", t)
            specs.append({
                "description": fn.get("description") or "",
                "name": fn["name"],
                "parameters": fn.get("parameters") or {},
                "type": t.get("type") or "function",
            })
        parts.append(
            "<|message_system|>tool_declare<|content_xml|>"
            + json.dumps(specs, sort_keys=True, separators=(",", ":"))
            + "<|end_message|>"
        )

    def emit_effort():
        eff = reasoning_effort
        if isinstance(eff, str):
            num = EFFORT_MAP[eff.strip()]
        else:
            num = float(eff)
        s_num = "0" if num == 0.0 else str(num)
        return f"<|message_system|><|content_text|>Thinking effort level: {s_num}<|end_message|>"

    effort_emitted = False
    for msg in messages:
        role = msg["role"]
        if not effort_emitted and role != "system":
            parts.append(emit_effort())
            effort_emitted = True

        header = ROLE_TOKEN[role]
        if role == "tool":
            name = msg.get("name")
            if not name and msg.get("tool_call_id"):
                tcid = msg["tool_call_id"]
                for prev in messages:
                    if prev.get("role") == "assistant" and prev.get("tool_calls"):
                        for tc in prev["tool_calls"]:
                            if tc.get("id") == tcid:
                                name = tc.get("function", {}).get("name")
            if name:
                header += name
        elif role == "assistant" and msg.get("name"):
            header += msg["name"]

        if role == "assistant" and msg.get("reasoning_content"):
            parts.append(f"{header}<|content_thinking|>{msg['reasoning_content']}<|end_message|>")

        content = msg.get("content")
        if isinstance(content, str) and (
            content or not (role == "assistant" and (msg.get("tool_calls") or msg.get("reasoning_content")))
        ):
            parts.append(f"{header}<|content_text|>{content}<|end_message|>")
        elif isinstance(content, list):
            for p in content:
                if isinstance(p, str):
                    parts.append(f"{header}<|content_text|>{p}<|end_message|>")
                elif isinstance(p, dict):
                    parts.append(f"{header}<|content_text|>{p.get('text', '')}<|end_message|>")

        if role == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc["function"]
                fn_name = fn["name"]
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                arg_items = ",".join(
                    f"{json.dumps(k)}:{json.dumps(args[k], separators=(',', ':'))}"
                    for k in sorted(args.keys())
                )
                parts.append(
                    f"<|message_model|>{fn_name}<|content_invoke_tool_json|>"
                    f'{{"name":{json.dumps(fn_name, sort_keys=True, separators=(",", ":"))},"args":{{{arg_items}}}}}'
                    f"<|end_message|>"
                )

        if role == "assistant":
            parts.append("<|content_model_end_sampling|>")

    if not effort_emitted:
        parts.append(emit_effort())
    return "".join(parts)


def parse_model_output(raw_text: str):
    """Parse Inkling-2-Mini output segments into content, reasoning_content, and tool_calls."""
    text = raw_text.replace("<|content_model_end_sampling|>", "")
    reasoning_parts = []
    content_parts = []
    tool_calls = []

    pat = re.compile(
        r"<\|message_model\|>([^<]*)(<\|content_thinking\|>|<\|content_text\|>|<\|content_invoke_tool_json\|>)(.*?)(?:<\|end_message\|>|$)",
        re.S,
    )
    matches = list(pat.finditer(text))
    if not matches:
        clean = text.replace("<|end_message|>", "").strip()
        return clean, None, []

    for m in matches:
        header_name, tag, body = m.group(1), m.group(2), m.group(3)
        if tag == "<|content_thinking|>":
            reasoning_parts.append(body)
        elif tag == "<|content_text|>":
            content_parts.append(body)
        elif tag == "<|content_invoke_tool_json|>":
            try:
                obj = json.loads(body.strip())
                fn_name = obj.get("name") or header_name.strip()
                fn_args = obj.get("args", {})
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {
                        "name": fn_name,
                        "arguments": json.dumps(fn_args, separators=(",", ":")),
                    },
                })
            except Exception:
                content_parts.append(body)

    content = "".join(content_parts) if content_parts else (None if tool_calls else "")
    reasoning = "".join(reasoning_parts) if reasoning_parts else None
    return content, reasoning, tool_calls


class EngineBackend:
    def __init__(self, bin_path, max_ctx=32768, use_spec=True, use_fp8=False):
        cmd = [bin_path, "--stdio-server", "--max-ctx", str(max_ctx)]
        if use_spec:
            cmd.append("--spec")
        if use_fp8:
            cmd.append("--fp8")
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        self.lock = threading.Lock()
        while True:
            line = self.proc.stderr.readline()
            if not line:
                raise RuntimeError("inkling_vk exited before READY")
            sys.stderr.write(f"[engine] {line}")
            sys.stderr.flush()
            if line.strip() == "READY":
                break

    def generate(self, tokens, max_tokens=256, stop_tokens=None, spec=True, adaptive=True):
        req = {
            "tokens": tokens,
            "max_tokens": int(max_tokens),
            "stop_tokens": stop_tokens if stop_tokens is not None else [EOS_ID],
            "spec": bool(spec),
            "adaptive": bool(adaptive),
        }
        with self.lock:
            self.proc.stdin.write(json.dumps(req, separators=(",", ":")) + "\n")
            self.proc.stdin.flush()
            resp_line = self.proc.stdout.readline()
            if not resp_line:
                raise RuntimeError("inkling_vk crashed")
            return json.loads(resp_line)

    def reset(self):
        with self.lock:
            self.proc.stdin.write('{"cmd":"reset"}\n')
            self.proc.stdin.flush()
            return json.loads(self.proc.stdout.readline())


def make_handler(engine: EngineBackend, tok: InklingTokenizer, api_key: str):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            sys.stderr.write(f"[http] {self.address_string()} - {fmt % args}\n")

        def _send_json(self, code, obj):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _check_auth(self):
            auth = self.headers.get("Authorization", "")
            if auth != f"Bearer {api_key}":
                self._send_json(401, {"error": {"message": "Invalid or missing API key", "type": "authentication_error"}})
                return False
            return True

        def do_GET(self):
            if self.path == "/health":
                self._send_json(200, {"status": "ok"})
                return
            if not self._check_auth():
                return
            if self.path == "/v1/models":
                self._send_json(200, {
                    "object": "list",
                    "data": [{"id": "inkling-2-mini", "object": "model", "owned_by": "local"}],
                })
            else:
                self._send_json(404, {"error": {"message": "Not found"}})

        def do_POST(self):
            if not self._check_auth():
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))

            if self.path == "/v1/reset":
                self._send_json(200, engine.reset())
                return

            if self.path == "/v1/completions":
                prompt = body.get("prompt", "")
                if isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
                    ptoks = prompt
                else:
                    ptoks = tok.encode(str(prompt))
                max_toks = int(body.get("max_tokens", 256))
                stops = body.get("stop_token_ids", [])
                spec = body.get("spec", True)
                adaptive = body.get("adaptive", True)
                res = engine.generate(ptoks, max_tokens=max_toks, stop_tokens=stops, spec=spec, adaptive=adaptive)
                if "error" in res:
                    self._send_json(400, {"error": {"message": res["error"]}})
                    return
                gen_ids = res["tokens"]
                text = tok.decode(gen_ids)
                self._send_json(200, {
                    "id": f"cmpl-{uuid.uuid4().hex[:12]}",
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": body.get("model", "inkling-2-mini"),
                    "choices": [{"index": 0, "text": text, "token_ids": gen_ids, "finish_reason": "length"}],
                    "usage": {
                        "prompt_tokens": res["prompt_tokens"],
                        "completion_tokens": res["completion_tokens"],
                        "total_tokens": res["prompt_tokens"] + res["completion_tokens"],
                        "prompt_reread_tokens": res["reread"],
                        "prefill_ms": res["prefill_ms"],
                        "decode_ms": res["decode_ms"],
                    },
                })
                return

            if self.path == "/v1/chat/completions":
                messages = body.get("messages", [])
                tools = body.get("tools")
                effort = body.get("reasoning_effort", 0.0)
                max_toks = int(body.get("max_tokens", 512))
                prompt_str = format_chat_prompt(messages, tools=tools, reasoning_effort=effort)
                ptoks = tok.encode(prompt_str, allow_special=True)
                res = engine.generate(ptoks, max_tokens=max_toks, stop_tokens=[EOS_ID], spec=True, adaptive=True)
                if "error" in res:
                    self._send_json(400, {"error": {"message": res["error"]}})
                    return
                gen_ids = res["tokens"]
                hit_eos = bool(gen_ids and gen_ids[-1] == EOS_ID)
                raw_text = tok.decode(gen_ids)
                content, reasoning, tool_calls = parse_model_output(raw_text)
                msg_obj = {"role": "assistant", "content": content}
                if reasoning is not None:
                    msg_obj["reasoning_content"] = reasoning
                if tool_calls:
                    msg_obj["tool_calls"] = tool_calls
                finish_reason = "tool_calls" if tool_calls else ("stop" if hit_eos else "length")
                self._send_json(200, {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": body.get("model", "inkling-2-mini"),
                    "choices": [{"index": 0, "message": msg_obj, "finish_reason": finish_reason}],
                    "usage": {
                        "prompt_tokens": res["prompt_tokens"],
                        "completion_tokens": res["completion_tokens"],
                        "total_tokens": res["prompt_tokens"] + res["completion_tokens"],
                        "prompt_reread_tokens": res["reread"],
                        "prefill_ms": res["prefill_ms"],
                        "decode_ms": res["decode_ms"],
                    },
                })
                return

            self._send_json(404, {"error": {"message": "Not found"}})

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--api-key", default=os.environ.get("INKLING_API_KEY", ""))
    ap.add_argument("--max-ctx", type=int, default=45000)
    ap.add_argument("--bin", default=os.path.expanduser("~/inkling2/inkling_vk"))
    ap.add_argument("--no-spec", action="store_true")
    ap.add_argument("--fp8", action="store_true")
    args = ap.parse_args()
    if not args.api_key:
        sys.stderr.write("ERROR: --api-key or INKLING_API_KEY is required\n")
        sys.exit(2)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        sys.stderr.write("ERROR: server must bind to localhost only\n")
        sys.exit(2)

    tok = InklingTokenizer()
    engine = EngineBackend(args.bin, max_ctx=args.max_ctx, use_spec=not args.no_spec, use_fp8=args.fp8)
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(engine, tok, args.api_key))
    sys.stderr.write(f"Listening on http://{args.host}:{args.port} (API key required)\n")
    sys.stderr.flush()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        engine.proc.terminate()


if __name__ == "__main__":
    main()
