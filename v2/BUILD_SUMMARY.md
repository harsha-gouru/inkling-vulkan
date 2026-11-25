# Inkling-2-Mini Clean Build (`~/inkling2`) — Plain-English Walkthrough

Built from scratch on `omarchy` (AMD Radeon AI PRO R9700 32 GB, Linux, Vulkan RADV, no PyTorch / llama.cpp / vLLM inside the engine).

## How the Engine Works (End-to-End Flow)

```
HTTP client (OpenAI API + Bearer key)
   │
   ▼
server.py  (127.0.0.1, pure Python stdlib)
   │  formats chat_template.jinja + parses <|content_invoke_tool_json|> tool calls
   │  encodes/decodes tokens with tokenizer.py (pure Python BPE)
   ▼  (JSON-lines over stdin/stdout)
inkling_vk --stdio-server  (single C++17 binary + 42 GLSL/SPIR-V Vulkan shaders)
   │
   ├─► Prefill (B = 256):
   │     1. LCP prefix check against cached_tokens (turn-boundary short_conv ckpt)
   │     2. Hardware WMMA (16×16×16 FP16→FP32) FlashAttention-2 + NVFP4 MoE GEMM
   │     3. Writes Hybrid KV cache (40 SWA 1024-ring + 8 Global) + MTP KV cache
   │
   └─► Decode (adaptive speculative decoding):
         1. MTP DFlare 3-layer + Markov RNN drafts K=7 tokens (~3.6 ms short, ~5.3 ms @ 40k)
         2. Target verifies all 8 tokens in 1 batch (Split-K shared-KV at depth)
         3. Rolls back short_conv history to exact accepted length
         4. If 4-round rolling acceptance < 3.0 (e.g. prose), auto-switches to fast B=1
```

## Stage-by-Stage Timeline (What Was Built)

1. **Stage 0 — Weight Map (`inspect_weights.py`, `MODEL_MAP.md`):**
   - Parsed `model-00001-of-00001.safetensors` (18.89 GB) by hand and mapped Inkling-2-Mini's unique architecture: 24 physical layers unrolled into **48 logical layers** (4 prelude + 12 core looped 3× with cross-pass controller messages + 8 epilogue), 40 SWA-1024 + 8 Global layers, depth-4 `short_conv`, AttnRes residual mixing, and 128-expert top-8 MoE (NVFP4 in prelude/core, MXFP8 in epilogue).
2. **Stage 1 — Slow CPU Reference (`ref_cpu.cpp`, `tokenizer.py`, `prompts.txt`):**
   - Built a pure-C++ FP32 CPU reference and verified **20/20** exact token match against `llama.cpp` (`-fa off`, F32 GGUF).
   - Generated the 500-token short F32 answer key (`~/src/bench/key32/prose1_500tok.json`).
3. **Stage 2 — First Vulkan GPU Engine (`vk_abi.h`, `inkling_vk.cpp`, `shaders/*.comp`):**
   - Built a zero-dependency Vulkan 1.3 compute engine using raw `libvulkan.so.1` + Buffer Device Address (`uint64_t` pointers, no descriptor sets) and `glslc`-compiled GLSL shaders.
   - Measured R9700 raw VRAM bandwidth at **637.0 GB/s** (`vk_probe`), setting the theoretical `B=1` decode ceiling at **115.9 tok/s (8.62 ms/tok)** for 5.49 GB read per token.
   - Verified **500/500 exact match** on `PROSE_1` (`verify_req_a_short.py`).
4. **Stage 3 — Short-Context Decode & DFlare MTP Speculative Decoding:**
   - **Plain `B=1` decode:** Pushed from ~30 tok/s → **86.8 tok/s** (`11.5 ms/tok`, 74.9% of ceiling) via `uvec4` coalesced BF16 matvecs, flattened 576-block NVFP4 `w2` (zero tail waste), transposed `rel_logits_proj` + Split-K attention, compute-buffer copies replacing transfer barriers, and fused 8-projection `matvec_bf16_attn.comp`.
   - **DFlare MTP speculative decoding (`--spec`):** Reverse-engineered `mtp.safetensors` (635.8 MB: 4 fused target hidden states → 3 Qwen3-like bidirectional draft layers → `rnn_hidden` Markov head). Added per-step `short_conv` history rollback (`hist_steps_buf`, 29.2 MiB) and folded `hidden_out.bias` into a top-1024 shortlist, cutting draft overhead to **3.64 ms/round** and pushing code decode to **133.5–142.7 tok/s** (`500/500` exact match preserved).
5. **Stage 4 — Long Context (25k Prefill, 40k Decode, 128k VRAM, Cache Reuse):**
   - **25k Prefill (Req C):** Enabled `VK_KHR_cooperative_matrix` on R9700 RADV (Wave64 `16×16×16` FP16→FP32 WMMA). Rewrote `attn_prefill_fa2.comp` (64 queries/WG, `1/16`-scaled FP16 to prevent outlier overflow while keeping `500/500` exactness) and `moe_nvfp4_gemm.comp` (`64×32×32`, `outer` factored into FP32 epilogue), jumping cold 25k prefill from `241 → 807.7 tok/s` (target ≥ 600).
   - **40k Decode (Req B2):** Added 32-split shared-KV attention (`attn_verify_splitk/reduce` and `mtp_attn_splitk/reduce`) so all 8 verify/draft queries share each Global KV read once, jumping 40k decode from `31.7 → 85.1 tok/s` (target ≥ 70).
   - **128k VRAM (Req E):** Full 128k Hybrid KV + MTP weights + 128k MTP KV + checkpoints use **28.09 / 31.86 GiB (3.77 GiB free)** (target ≥ 1 GiB free).
   - **Cache Reuse (Req D):** Added a 3.8 MiB `hist_ckpt_buf` turn-boundary `short_conv` checkpoint so growing conversations reuse 99%+ of prior KV even if a client reformats the previous assistant turn.
6. **Stage 5 — Localhost OpenAI Server (`server.py`, `verify_req_f.py`):**
   - Added `--stdio-server` mode with adaptive `--spec` fallback and built `server.py` (127.0.0.1 only, Bearer API key required, `/v1/models`, `/v1/completions`, `/v1/chat/completions` with Inkling tool calls).
   - Verified auth (`401`/`200`), served **500/500** exactness, 15-turn coding agent session (7 tool calls), and 14/14 growing-turn cache reuse (`verify_req_f.py`).

## Key Files & How to Run

- Start server: `python3 ~/inkling2/server.py --host 127.0.0.1 --port 8080 --api-key YOUR_KEY --max-ctx 45000`
- Verify short exactness (500/500): `python3 ~/inkling2/verify_req_a_short.py`
- Verify serving + auth + 15-turn tool calls + Req D cache reuse: `python3 ~/inkling2/verify_req_f.py`
- Run full 5-run benchmark suite (B1, B2, C, E): `python3 ~/inkling2/bench_final_gates.py`
- Full optimization log (all 37 optimizations with predictions & measurements): `~/inkling2/LOG.md`
