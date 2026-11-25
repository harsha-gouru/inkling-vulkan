#!/usr/bin/env python3
"""Zero-dependency BPE tokenizer for Inkling-2-Mini (reads tiktoken/tokenizer.model + tokenizer.json)."""
import base64
import json
import os
import re
import unicodedata

MODEL_DIR = os.path.expanduser("~/models/Inkling-2-Mini")


def _build_pretok_split(text: str):
    """Pure-Python equivalent of the o200k/inkling2 Unicode pre-tokenizer regex."""
    # Categories: L (Lu/Lt/Lm/Lo/Ll + M), N, whitespace, punctuation/other
    # Using a clean scanner that matches the exact regex in tokenizer.json:
    # [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
    # | [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
    # | \p{N}{1,3}
    # |  ?[^\s\p{L}\p{N}]+[\r\n/]*
    # | \s*[\r\n]+
    # | \s+(?!\S)
    # | \s+
    n = len(text)
    i = 0
    out = []

    def cat(ch):
        return unicodedata.category(ch)

    def is_L(ch):
        return cat(ch)[0] == "L"

    def is_N(ch):
        return cat(ch)[0] == "N"

    def is_upper(ch):
        return cat(ch) in ("Lu", "Lt", "Lm", "Lo") or cat(ch)[0] == "M"

    def is_lower(ch):
        return cat(ch) in ("Ll", "Lm", "Lo") or cat(ch)[0] == "M"

    def is_space(ch):
        return ch.isspace()

    def match_contraction(pos):
        rem = text[pos:pos + 3].lower()
        for c in ("'re", "'ve", "'ll"):
            if rem.startswith(c):
                return 3
        rem2 = text[pos:pos + 2].lower()
        for c in ("'s", "'t", "'m", "'d"):
            if rem2.startswith(c):
                return 2
        return 0

    while i < n:
        ch = text[i]
        # Branch 1 & 2: optional non-newline/L/N prefix followed by letters
        has_prefix = (ch not in ("\r", "\n") and not is_L(ch) and not is_N(ch))
        start_let = i + 1 if has_prefix else i
        if start_let < n and is_L(text[start_let]):
            # Try branch 1: upper* lower+ contraction?
            j = start_let
            while j < n and is_upper(text[j]) and not (cat(text[j]) == "Ll"):
                j += 1
            k = j
            while k < n and is_lower(text[k]):
                k += 1
            if k > j:
                k += match_contraction(k)
                out.append(text[i:k])
                i = k
                continue
            # Try branch 2: upper+ lower* contraction?
            if j > start_let:
                k = j
                while k < n and is_lower(text[k]):
                    k += 1
                k += match_contraction(k)
                out.append(text[i:k])
                i = k
                continue

        # Branch 3: \p{N}{1,3}
        if is_N(ch):
            j = i + 1
            while j < n and j - i < 3 and is_N(text[j]):
                j += 1
            out.append(text[i:j])
            i = j
            continue

        # Branch 4: ' '? [^\s\p{L}\p{N}]+ [\r\n/]*
        j = i + 1 if ch == " " else i
        if j < n and not is_space(text[j]) and not is_L(text[j]) and not is_N(text[j]):
            k = j + 1
            while k < n and not is_space(text[k]) and not is_L(text[k]) and not is_N(text[k]):
                k += 1
            while k < n and text[k] in ("\r", "\n", "/"):
                k += 1
            out.append(text[i:k])
            i = k
            continue

        # Branch 5/6/7: whitespace
        if is_space(ch):
            j = i
            last_nl = -1
            while j < n and is_space(text[j]):
                if text[j] in ("\r", "\n"):
                    last_nl = j
                j += 1
            if last_nl >= i:
                out.append(text[i:last_nl + 1])
                i = last_nl + 1
                continue
            if j < n and j - i > 1:
                # \s+(?!\S) leaves the last space before non-space
                out.append(text[i:j - 1])
                i = j - 1
                continue
            out.append(text[i:j])
            i = j
            continue

        out.append(ch)
        i += 1
    return out


class InklingTokenizer:
    def __init__(self, model_dir: str = MODEL_DIR):
        self.ranks: dict[bytes, int] = {}
        self.id_to_bytes: dict[int, bytes] = {}
        with open(os.path.join(model_dir, "tiktoken", "tokenizer.model"), "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                b64, r = line.split()
                tok_b = base64.b64decode(b64)
                rid = int(r)
                self.ranks[tok_b] = rid
                self.id_to_bytes[rid] = tok_b

        self.special_to_id: dict[str, int] = {}
        self.id_to_special: dict[int, str] = {}
        with open(os.path.join(model_dir, "tokenizer.json"), "r") as f:
            tj = json.load(f)
        for item in tj.get("added_tokens", []):
            cid = int(item["id"])
            content = item["content"]
            self.special_to_id[content] = cid
            self.id_to_special[cid] = content
        if self.special_to_id:
            self._special_re = re.compile(
                "(" + "|".join(re.escape(k) for k in sorted(self.special_to_id, key=len, reverse=True)) + ")"
            )
        else:
            self._special_re = None

    def _bpe(self, piece: bytes) -> list[int]:
        if piece in self.ranks:
            return [self.ranks[piece]]
        parts = [bytes([b]) for b in piece]
        while len(parts) >= 2:
            best_rank = None
            best_idx = -1
            for idx in range(len(parts) - 1):
                merged = parts[idx] + parts[idx + 1]
                r = self.ranks.get(merged)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_idx = idx
            if best_idx < 0:
                break
            parts[best_idx:best_idx + 2] = [parts[best_idx] + parts[best_idx + 1]]
        return [self.ranks[p] for p in parts]

    def encode(self, text: str, allow_special: bool = True) -> list[int]:
        ids: list[int] = []
        chunks = self._special_re.split(text) if (allow_special and self._special_re) else [text]
        for ch in chunks:
            if not ch:
                continue
            if allow_special and ch in self.special_to_id:
                ids.append(self.special_to_id[ch])
                continue
            for piece in _build_pretok_split(ch):
                ids.extend(self._bpe(piece.encode("utf-8")))
        return ids

    def decode(self, ids: list[int]) -> str:
        out = bytearray()
        for tid in ids:
            if tid in self.id_to_bytes:
                out.extend(self.id_to_bytes[tid])
            elif tid in self.id_to_special:
                out.extend(self.id_to_special[tid].encode("utf-8"))
        return out.decode("utf-8", errors="replace")


def load_prompts(path: str = os.path.expanduser("~/inkling2/prompts.txt")) -> dict[str, str]:
    with open(path) as f:
        text = f.read()
    res = {}
    for m in re.finditer(r"=== PROMPT \d+: (\S+) ===\n(.*?)\n=== END PROMPT ===", text, re.S):
        res[m.group(1)] = m.group(2)
    return res


if __name__ == "__main__":
    tok = InklingTokenizer()
    prompts = load_prompts()
    for name, body in prompts.items():
        ids = tok.encode(body)
        rt = tok.decode(ids)
        assert rt == body, f"Roundtrip mismatch on {name}"
        print(f"{name:8s}: {len(ids):6d} tokens (first 5: {ids[:5]})")

    # Verify against llama.cpp PROSE_1 reference
    with open(os.path.expanduser("~/src/bench/key32/prose1_20tok.json")) as f:
        ref = json.load(f)
    got_p = tok.encode(prompts["PROSE_1"])
    assert got_p == ref["prompt_tokens"], f"Prompt token mismatch: {got_p} vs {ref['prompt_tokens']}"
    got_c = tok.decode(ref["generated_tokens"])
    assert got_c == ref["content"], f"Decode mismatch: {got_c!r} vs {ref['content']!r}"
    print("PASS: Exact encode & decode match with llama.cpp on PROSE_1!")
