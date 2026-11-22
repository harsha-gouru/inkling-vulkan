# Inkling-2-Mini Raw Vulkan Engine & Server (RDNA 4 / R9700)

Standalone inference engine and OpenAI-compatible continuous-batching server for Thinking Machines Lab's **[Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)** (text + vision + tool calling + DSpark MTP speculative decoding, Apache 2.0) on AMD Radeon AI PRO R9700 (32 GiB VRAM, RADV GFX1201).

**Native quantized weights + FP8 E4M3 KV & attention/draft GEMMs + DSpark MTP:** reads the published safetensors weights directly (`NVFP4` MoE layers 0–19, `MXFP8` MoE layers 20–23, exact `BF16` MoE routers, and fast 24-core OpenMP `FP8 E4M3FN` target attention & MTP draft projections with per-row `float32` scale) with **FP8 E4M3FN** KV cache (`1,024 B/tok/layer`, zero scale buffers), **FP8 E4M3FN** `lm_head` (including a specialized 16-row/WG `B=1` draft unembed kernel), **DSpark MTP** (`block_size=8` 3-layer draft + Markov RNN with adaptive `ema_acc` guard, 8-token batch-tiled & 32× coalesced loop controllers, fused `P_CONV_RESID`, active-expert MoE compaction, and 4-token scalar-register Split-K attention), a GPU VRAM L1 prefix cache pool (`4 × 32K`), and a pure-NumPy/OpenBLAS CPU **Inkling2V** vision encoder (`0 MiB` VRAM).

> **Note:** Model weights are **not** included in this repository. Download them directly from Hugging Face (see below).

## Verified Numbers (R9700 32 GiB, `MAX_BATCH=2`, `SEQ_CAP=131072`)
- **Startup & VRAM (`2 × 128K` = `262,144` live KV tokens + DSpark MTP + `4 × 32K` VRAM prefix pool):** **`2.04 – 2.17 s`** cold start, **`25.14 GiB`** live DRM VRAM (**`6.72 GiB` free** on 32 GiB R9700; 40 SWA layers use a `1,024`-token ring buffer, 8 full-attention layers use linear `131,072`-token FP8 KV).
- **Correctness vs. `llama.cpp` F32 (`-fa off`, 500 tokens):**
  - **FP8 E4M3 KV + FP8 attention/draft GEMMs + FP8 `lm_head`:** **480 / 500 (96.00%)** next-token agreement vs. `llama.cpp` F32 (`490 / 500 = 98.00%` vs. BF16 attention with `median |logit diff| = 0.0598`; `64 / 64 = 100.0%` token-exact agreement on real 50,000-token code prompts).
  - **DSpark MTP vs. plain `B=1` (`step_batch`):** **120 / 120 (100.00% bit-exact greedy parity)** on Python and C11 coding benchmarks and **64 / 64 (100.00%)** at 50K context depth.
  - **FP16 KV + BF16 `lm_head`:** **495 / 500 (99.00%)**; **Pure F32 KV + BF16 `lm_head`:** **500 / 500 (100.00%)** (`max |logit diff| = 0.00004`).
- **Single-stream decode (`B=1` plain vs. DSpark MTP):**
  - **Short coding prompts (120 tokens greedy):**
    - Python (`merge sort + binary search`): **`91.3 tok/s`** (`B=1`) → **`117.8 tok/s`** (`MTP Adaptive`, `114.5 tok/s` `MTP Forced`, `120/120` exact token match, `TTFT = 105.8 ms`).
    - C11 (`lock-free SPSC ring buffer`): **`90.3 tok/s`** (`B=1`) → **`88.6 tok/s`** (`MTP Adaptive`, `120/120` exact token match, `TTFT = 127.2 ms`).
  - **Real 32,768-token & 50,000-token codebase prompts (64 tokens greedy):**
    - `D = 32,768` (real repo corpus): **`52.5 tok/s`** (`B=1`, `19.05 ms/tok`) → **`61.8 tok/s`** (`MTP Forced`, `acc=3.56`) / **`53.3 tok/s`** (`MTP Adaptive`).
    - `D = 50,000` (real repo corpus): **`44.2 tok/s`** (`B=1`, `22.63 ms/tok`) → **`50.3 tok/s`** (`MTP Forced`, `acc=3.20`) / **`50.4 tok/s`** (`MTP Adaptive`, `64/64` exact match).
  - **MTP pass latency (`draft7 + verify8 + rollback`) across context depth:**
    - `D = 9,216`: **`44.11 ms/pass`** (`draft7 = 6.96 ms, verify8 = 36.80 ms, rb = 0.35 ms`) → **`124.7–147.4 tok/s`** (`@5.5–6.5 acc`).
    - `D = 32,768`: **`53.66 ms/pass`** (`draft7 = 7.72 ms, verify8 = 45.70 ms, rb = 0.25 ms`) → **`102.5–121.1 tok/s`** (`@5.5–6.5 acc`).
    - `D = 50,000`: **`59.46 ms/pass`** (`draft7 = 8.28 ms, verify8 = 50.93 ms, rb = 0.25 ms`) → **`92.5–109.3 tok/s`** (`@5.5–6.5 acc`).
- **Chunked expert-grouped prefill (`C=64/16`, no output head):** **`443.6 – 465.7 tok/s`** @ 512 tokens (`1.15 s`), **`388.4 tok/s`** @ 4,096 tokens (`10.55 s`), **`291.0 tok/s`** @ 32,768 tokens (`112.6 s`), **`252.6 tok/s`** @ 50,000 tokens (`197.9 s`).
- **Tool calling & multi-tier prefix caching:** OpenAI `tools` + incremental streaming `tool_calls`, per-slot LRU KV + `4 × 32K` GPU VRAM L1 prefix cache pool (`~2 ms` device-to-device copy via `vk_copy_batch`), `3.66 MiB` conv-history snapshot/rollback, and exact raw-token assistant turn replay (`asst_token_cache`) delivering **99.5–99.9%** KV cache hits across multi-turn tool calls and multimodal follow-ups (`<100 ms` warm TTFT).
- **Multimodal vision (`Inkling2V` CPU ViT):** OpenAI `image_url` (base64 data URLs / local paths) with SHA-256 LRU embedding cache (`0 MiB` VRAM, `~0.65 s` @ 480×320, `~0.03–8 ms` cached), bypassing `embed_norm` via zero-shader `embed_overrides` with collision-free `(sha256, patch_idx)` KV prefix caching.

## Quick Start

### 1. Prerequisites
- Linux with Vulkan 1.3 driver + headers (`libvulkan`, `gcc` with OpenMP)
- Python 3.10+ and packages in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 2. Download Model Weights
Weights are published by Thinking Machines Lab under the Apache 2.0 license:
```bash
hf download thinkingmachines/Inkling-2-Mini --local-dir ./Inkling-2-Mini
```

### 3. Build Native Libraries & SPIR-V Shaders
```bash
make && make shaders
```
`make` compiles `libvk_inkling.so` (Vulkan 1.3 command-buffer runner + 24-core OpenMP BF16→FP8 quantizer) and `libdump_spv.so` (explicit Vulkan layer for capturing compiled SPIR-V). `make shaders` runs `build_batched_spv.py`, `build_prefill_spv.py`, and `build_dspark_spv.py` with `VK_LAYER_DUMP_SPV` to generate `spv_batch/*.spv`, `spv_prefill/*.spv`, and `spv_dspark/*.spv`.

### 4. Run the Server
```bash
INKLING_MODEL_DIR=./Inkling-2-Mini   python serve_batch_vk.py --host 127.0.0.1 --port 8080 --max-batch 2 --seq-cap 131072 --timeout 600
```
Set `INKLING_API_KEY="your-secret"` (or `--api-key`) to require Bearer auth, or pass `--model-dir /path/to/Inkling-2-Mini`.

## Key Files
- `vk_inkling.c` — C Vulkan 1.3 pre-recorded command-buffer runner, batched GPU buffer copy (`vk_copy_batch`), and OpenMP BF16→FP8 E4M3 quantizer (`libvk_inkling.so`)
- `dump_spv.c` & `VK_LAYER_DUMP_SPV.json` — Vulkan explicit layer used by `make shaders` to capture SPIR-V binaries
- `build_batched_spv.py` / `build_prefill_spv.py` / `build_dspark_spv.py` — RDNA 4 Wave64 WGSL → SPIR-V shader generators (FP8 E4M3 KV, hybrid FP8/BF16 GEMMs, 8-token batch-tiled & 32× coalesced loop controllers, fused `P_CONV_RESID`, race-free MoE active-expert compaction, 4-token scalar-register Split-K attention, and DSpark MTP draft/verify kernels)
- `inkling_batch_vk.py` — multi-bucket (`1, 2, 4, 8, 16`) + chunked expert-grouped prefill (`64, 16`) + DSpark MTP `step_speculative` + GPU VRAM L1 prefix pool + `embed_overrides` Vulkan engine
- `inkling_vit.py` — pure NumPy/OpenBLAS + Pillow Inkling2V CPU vision encoder (23-layer 2D axial RoPE ViT + SHA-256 LRU cache)
- `serve_batch_vk.py` — OpenAI-compatible HTTP/SSE server (LRU prompt cache, GPU VRAM L1 prefix cache, raw-token assistant replay, DSpark MTP single-stream decode, streaming `tool_calls`, multimodal `image_url`, GPU top-128 sampler, `/health`, `/stats`)
- `verify_prod_engine.py`, `verify_prefill_vk.py`, `bench_ttft_matrix.py`, `test_prod_server.py` — verification & benchmark scripts
- `Makefile` — builds `libvk_inkling.so`, `libdump_spv.so`, and all SPIR-V shaders

## Attribution & License
- Engine & server code: [Apache License 2.0](LICENSE).
- Model architecture & weights: **[Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)** by Thinking Machines Lab, released under the Apache License 2.0.
