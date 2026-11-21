# Inkling-2-Mini Raw Vulkan Engine & Server (RDNA 4 / R9700)

Standalone inference engine and OpenAI-compatible continuous-batching server for Thinking Machines Lab's **[Inkling-2-Mini](https://huggingface.co/thinkingmachineslab/Inkling-2-Mini)** (Apache 2.0) on AMD Radeon AI PRO R9700 (32 GiB VRAM, RADV GFX1201).

**Zero re-quantization:** reads the exact published safetensors weights (`NVFP4` layers 0–19, `MXFP8` layers 20–23, `BF16` dense/attention/unembed) directly on GPU with a pure **F32** KV cache and attention path.

> **Note:** Model weights are **not** included in this repository. Download them directly from Hugging Face (see below).

## Verified Numbers (R9700 32 GiB, exact native weights)
- **Correctness vs. `llama.cpp` F32 (`-fa off`, pure F32):** **500 / 500 (100.00%)** next-token agreement (`max |logit diff| = 0.00004`), both alone (`B=1`) and inside a batch of 9 active users (`B=9`, `B=1 vs B=9 max |diff| = 0.000000`).
- **Single-stream decode:** **80.6 tok/s** @ depth 50, **70.4 tok/s** @ depth 2,000, **68.9 tok/s** @ depth 3,000 (vs. `llama.cpp` Vulkan Q6_K: `20.4` @ 50, `18.3` @ 2,000).
- **Chunked expert-grouped prefill (`C=64/16`, no output head):** **455–464 tok/s** (`652` tok cold TTFT `1.43 s`, down from `9.28 s`; vs. `llama.cpp` `386 tok/s`).
- **Continuous batching (1–16 concurrent users):** `80` (1) → `130` (2) → `189` (4) → `241` (8) → `273 tok/s` (16); flattens between 8–16 users due to 8-of-128 MoE expert fan-out (`step_ms(B) ≈ 9.5 + 3.05·B`).
- **Context & VRAM:** `16` slots × `4,096` tokens (`65,536` live KV tokens; 40 SWA slots use a `1,024` ring buffer, 8 full-attn slots use linear `4,096`), `26.84 GiB` real VRAM (`0.11 GiB` GTT), `2.1 s` cold start.
- **Prompt caching:** per-slot LRU KV + `3.66 MiB` conv-history snapshot/rollback (`2,008`-tok warm TTFT `106.6 ms` alone, `176.1 ms` with 8 users generating).

## Quick Start

### 1. Prerequisites
- Linux with Vulkan 1.3 driver + headers (`libvulkan`, `gcc`)
- Python 3.10+ and packages in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 2. Download Model Weights
Weights are published by Thinking Machines Lab under the Apache 2.0 license:
```bash
hf download thinkingmachineslab/Inkling-2-Mini --local-dir ./Inkling-2-Mini
```

### 3. Build Native Libraries & SPIR-V Shaders
```bash
make && make shaders
```
`make` compiles `libvk_inkling.so` (Vulkan 1.3 command-buffer runner) and `libdump_spv.so` (explicit Vulkan layer for capturing compiled SPIR-V). `make shaders` runs `build_batched_spv.py` and `build_prefill_spv.py` with `VK_LAYER_DUMP_SPV` to generate `spv_batch/*.spv` and `spv_prefill/*.spv`.

### 4. Run the Server
```bash
INKLING_MODEL_DIR=./Inkling-2-Mini INKLING_API_KEY="your-secret" \
  python serve_batch_vk.py --host 127.0.0.1 --port 8090 --max-queue 32 --timeout 60
```
You can also pass `--model-dir /path/to/Inkling-2-Mini` instead of `INKLING_MODEL_DIR`.

## Key Files
- `vk_inkling.c` — C Vulkan 1.3 pre-recorded command-buffer runner (`libvk_inkling.so`)
- `dump_spv.c` & `VK_LAYER_DUMP_SPV.json` — Vulkan explicit layer used by `make shaders` to capture SPIR-V binaries
- `build_batched_spv.py` / `build_prefill_spv.py` — RDNA 4 Wave64 WGSL → SPIR-V shader generators
- `inkling_batch_vk.py` — multi-bucket (`1, 2, 4, 8, 16`) + chunked expert-grouped prefill (`64, 16`) engine
- `serve_batch_vk.py` — OpenAI-compatible HTTP/SSE server (LRU prompt cache, GPU top-128 top-p/temp/seed sampler, API key, queue cap, timeout, `/stats`)
- `verify_prod_engine.py`, `verify_prefill_vk.py`, `bench_ttft_matrix.py`, `test_prod_server.py` — verification & benchmark scripts
- `Makefile` — builds `libvk_inkling.so`, `libdump_spv.so`, and all SPIR-V shaders

## Attribution & License
- Engine & server code: [Apache License 2.0](LICENSE).
- Model architecture & weights: **[Inkling-2-Mini](https://huggingface.co/thinkingmachineslab/Inkling-2-Mini)** by Thinking Machines Lab, released under the Apache License 2.0.
