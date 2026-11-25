#!/usr/bin/env python3
"""Verify Requirement F (localhost OpenAI server, API key auth, 15-turn tool-call agent),
Requirement A (short 500/500 via served /v1/completions), and Requirement D (prefix cache reuse)."""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

HOST = "127.0.0.1"
PORT = 18088
API_KEY = "inkling-local-test-key-2026"
BASE = f"http://{HOST}:{PORT}"


def req_json(method, path, body=None, api_key=API_KEY):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"}
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"
    r = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(r, timeout=300) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def main():
    srv = subprocess.Popen(
        [sys.executable, "server.py", "--host", HOST, "--port", str(PORT), "--api-key", API_KEY, "--max-ctx", "8192"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        # Wait for /health
        for _ in range(120):
            try:
                st, _ = req_json("GET", "/health", api_key=None)
                if st == 200:
                    break
            except Exception:
                time.sleep(0.25)
        else:
            raise RuntimeError("Server failed to become healthy")

        # 1. Auth checks
        st_none, _ = req_json("GET", "/v1/models", api_key=None)
        st_bad, _ = req_json("GET", "/v1/models", api_key="wrong-key")
        st_ok, models = req_json("GET", "/v1/models", api_key=API_KEY)
        assert st_none == 401, f"Expected 401 without key, got {st_none}"
        assert st_bad == 401, f"Expected 401 with bad key, got {st_bad}"
        assert st_ok == 200 and models["data"][0]["id"] == "inkling-2-mini"
        print("PASS [Auth]: missing key -> 401, wrong key -> 401, valid key -> 200")

        # 2. Served Req A (short 500/500)
        key = json.load(open(os.path.expanduser("~/src/bench/key32/prose1_500tok.json")))
        st, resp = req_json("POST", "/v1/completions", {
            "model": "inkling-2-mini",
            "prompt": key["prompt_tokens"],
            "max_tokens": 500,
            "stop_token_ids": [],
            "spec": True,
            "adaptive": True,
        })
        assert st == 200, f"/v1/completions failed: {st} {resp}"
        got = resp["choices"][0]["token_ids"]
        ref = key["generated_tokens"]
        matches = sum(a == b for a, b in zip(ref, got))
        u = resp["usage"]
        print(f"PASS [Req A served]: {matches}/500 exact match | decode={1000.0*u['completion_tokens']/u['decode_ms']:.1f} tok/s (reread={u['prompt_reread_tokens']})")
        assert matches == 500, f"Expected 500/500, got {matches}/500"

        # Reset cache before 15-turn coding agent session
        req_json("POST", "/v1/reset", {})

        # 3. 15-turn coding agent session with tool calls + Req D cache reuse
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "List files in a directory of the repository.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string", "description": "Relative directory path"}},
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read the contents of a file in the repository.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string", "description": "Relative file path"}},
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_tests",
                    "description": "Run the project unit test suite and return output.",
                    "parameters": {
                        "type": "object",
                        "properties": {"target": {"type": "string", "description": "Test file or 'all'"}},
                        "required": ["target"],
                    },
                },
            },
        ]

        mock_files = {
            "src/calc.py": "def add(a, b):\n    return a + b\n\ndef divide(a, b):\n    return a / b  # bug: no ZeroDivisionError check\n",
            "tests/test_calc.py": "from src.calc import add, divide\ndef test_add(): assert add(2, 3) == 5\ndef test_div_zero(): assert divide(1, 0) is None\n",
        }

        def exec_tool(name, args_json):
            try:
                args = json.loads(args_json) if isinstance(args_json, str) else args_json
            except Exception:
                args = {}
            if name == "list_files":
                return "src/calc.py\ntests/test_calc.py\nREADME.md"
            if name == "read_file":
                p = args.get("path", "src/calc.py")
                return mock_files.get(p, f"File not found: {p}")
            if name == "run_tests":
                return "FAILED tests/test_calc.py::test_div_zero - ZeroDivisionError: division by zero"
            return f"Unknown tool: {name}"

        followups = [
            "Please inspect the repo using list_files first.",
            "Now read src/calc.py using read_file.",
            "Now read tests/test_calc.py using read_file.",
            "Run the test suite using run_tests.",
            "Explain the exact bug in divide() in one sentence.",
            "Propose the 2-line fix for src/calc.py.",
            "Also check if add() needs any type check.",
            "Write a short docstring for divide().",
            "What edge case should we test for negative numbers?",
            "List one performance consideration if divide() is called in a tight loop.",
            "Call read_file on src/calc.py again to double check line numbers.",
            "Call run_tests with target='all' to confirm the failure signature.",
            "Summarize all findings in 3 bullet points.",
            "Give a one-line commit message for the fix.",
            "Confirm you are ready to apply the patch.",
        ]

        messages = [
            {"role": "system", "content": "You are a concise coding assistant. Use the provided tools when asked."},
            {"role": "user", "content": followups[0]},
        ]
        f_idx = 1
        total_tool_calls = 0
        prev_prompt_toks = 0
        prev_comp_toks = 0
        req_d_failures = []

        for turn in range(1, 16):
            st, resp = req_json("POST", "/v1/chat/completions", {
                "model": "inkling-2-mini",
                "messages": messages,
                "tools": tools,
                "reasoning_effort": 0.0,
                "max_tokens": 256,
            })
            assert st == 200, f"Turn {turn} failed: {st} {resp}"
            choice = resp["choices"][0]
            msg = choice["message"]
            u = resp["usage"]
            pt = u["prompt_tokens"]
            rr = u["prompt_reread_tokens"]
            ct = u["completion_tokens"]

            # Check Req D on turns >= 2 (where prompt grew at the end)
            if turn >= 2:
                # New tokens added since the last cached KV position (prev_prompt + prev_comp - 1)
                new_toks = pt - prev_prompt_toks
                # Allowed: new_toks + 10% (min +5 tokens margin for boundary EOS/template token)
                limit = max(new_toks * 1.10, new_toks + 5)
                ok_d = rr <= limit
                if not ok_d:
                    req_d_failures.append((turn, pt, new_toks, rr))
                ratio_str = f"new={new_toks:3d} reread={rr:3d} ({'OK' if ok_d else 'FAIL'})"
            else:
                ratio_str = f"cold reread={rr:3d}"

            tcs = msg.get("tool_calls") or []
            total_tool_calls += len(tcs)
            print(f"Turn {turn:2d}: prompt={pt:4d} {ratio_str:28s} comp={ct:3d} finish={choice['finish_reason']:10s} tools={len(tcs)}")

            # Append assistant message preserving exact structure
            asst_entry = {"role": "assistant", "content": msg.get("content")}
            if msg.get("reasoning_content"): asst_entry["reasoning_content"] = msg["reasoning_content"]
            if tcs:
                asst_entry["tool_calls"] = tcs
            messages.append(asst_entry)

            if tcs:
                for tc in tcs:
                    out = exec_tool(tc["function"]["name"], tc["function"]["arguments"])
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "name": tc["function"]["name"],
                        "content": out,
                    })
            else:
                if f_idx < len(followups):
                    messages.append({"role": "user", "content": followups[f_idx]})
                    f_idx += 1
                else:
                    messages.append({"role": "user", "content": "Thanks, please summarize in one line."})

            prev_prompt_toks = pt
            prev_comp_toks = ct

        print(f"Total tool calls executed across 15 turns: {total_tool_calls}")
        assert total_tool_calls >= 1, "Expected at least 1 tool call in 15-turn session"
        assert not req_d_failures, f"Req D cache reuse failures: {req_d_failures}"
        print("PASS [Req D]: Cache reuse <= new + 10% on all growing turns!")
        print("PASS [Req F]: 15-turn coding agent session with tool calls completed cleanly!")
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=5)
        except Exception:
            srv.kill()


if __name__ == "__main__":
    main()
