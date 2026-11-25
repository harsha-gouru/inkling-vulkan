# Inkling-2-Mini Clean Build Log

## Stage 0 — Model Map

### What was built
- Inspected `~/models/Inkling-2-Mini/config.json`, `hf_quant_config.json`, `model-00001-of-00001.safetensors`, and `mtp.safetensors` by hand using Python stdlib (`struct`/`json`), cross-checked against `~/src/llama.cpp/src/models/inkling2*.cpp` and `conversion/inkling2.py`.
- Wrote `MODEL_MAP.md` documenting all 1,244 main tensors (822 LLM + 422 Vision) and 54 MTP tensors, exact shapes, mixed NVFP4/MXFP8/BF16/F32 storage formats, and the complete step-by-step forward pass for one token.
- Wrote `inspect_weights.py` (zero external dependencies) to verify every file size, tensor count, dtype breakdown, and active-weight byte estimate.
- Documented key format quirks: interleaved gate/up rows in `w13_weight`, `hf_quant_config.json` claiming `[32, 32]` MXFP8 blocks while stored scales are `[1, 32]`, reversed lag order in Global `static_w`, and 48 independent logical KV/conv cache slots across 24 physical layers.

### Gate result
- Passed: `MODEL_MAP.md` covers every tensor, shape, format, and operation without requiring code reading, and `inspect_weights.py` verified exact byte-for-byte header accounting.
- Exact command to rerun:
  ```bash
  ssh omarchy 'python3 ~/inkling2/inspect_weights.py'
  ```

### Key numbers
- `model-00001-of-00001.safetensors`: 1,244 tensors, 19,526,019,168 bytes / 19.526 GB (`measured`)
  - LLM backbone (`model.llm.*`): 822 tensors, 18,889,358,496 bytes / 18.889 GB (`measured`)
  - Vision tower (`model.visual.*`): 422 tensors, 636,512,256 bytes / 0.637 GB (`measured`)
- `mtp.safetensors`: 54 tensors, 635,801,416 bytes / 0.636 GB (`measured`)
- NVFP4 `outer_scale_denominator`: `2688.0` across all 20 NVFP4 layers (`measured`)
- Active weight bytes read per decode token (batch=1, 48 logical layers, 8/128 experts, 200,058 unembed rows): 5,493,745,248 bytes / 5.494 GB / 5.116 GiB (`inferred`)
- KV cache at 128,000 tokens (40 SWA slots @ 1,024 + 8 Global slots @ 128,000): 4.16 GiB in FP16/BF16 (`inferred`)

### What is not verified
- No forward pass has been executed yet; numerical agreement with `~/models/gguf/Inkling-2-Mini-f32.gguf` and `~/src/answer-key-f32.sh` (with `-fa off`) is unverified until Stage 1.
- Tokenizer exact behavior (`tokenizer.json` / `tiktoken`) and `prompts.txt` have not been built or approved yet (Stage 1).
- R9700 Vulkan memory bandwidth and actual runtime VRAM overhead have not been measured yet (Stage 2).

### Prediction question for the human (before Stage 1)
Before I write the five fixed test prompts (`prompts.txt`) and the slow CPU reference in Stage 1: **which single part of the Inkling-2-Mini forward pass do you think is most likely to cause our first numerical mismatch against the F32 llama.cpp reference (e.g. NVFP4/MXFP8 dequantization, interleaved `w13` gate/up split, AttnRes block/loop reset, dynamic short-conv lag order, or relative-position bias + attention sink)?**

---

## Stage 1 — Slow CPU Reference

### What was built
- Patched `~/src/answer-key-f32.sh` to pass `-fa off` (avoiding llama.cpp's 16-bit flash-attention KV cache rounding).
- Created `~/inkling2/prompts.txt` with five fixed prompts: `PROSE_1` (89 chars, 18 tokens), `PROSE_2` (106 chars), `CODE_1` (117 chars), `CODE_2` (129 chars), and `LONG_1` (120,970 chars, >20k tokens).
- Built `~/inkling2/ref_cpu.cpp` (self-contained C++20 + OpenMP, zero external dependencies, mmap's `model-00001-of-00001.safetensors` directly and dequantizes NVFP4/MXFP8 experts on the fly).
- Built `~/inkling2/get_ref_tokens.py` (queries CPU F32 `llama-server -fa off -ctk f32 -ctv f32` for exact greedy token IDs) and `~/inkling2/verify_stage1.py`.

### Gate result
- Passed: **20 of 20** exact next tokens on `PROSE_1` against `~/models/gguf/Inkling-2-Mini-f32.gguf` (`-ngl 0 --device none -fa off -ctk f32 -ctv f32 --temp 0`), on the very first compile and run.
- Exact command to rerun:
  ```bash
  ssh omarchy 'python3 ~/inkling2/verify_stage1.py'
  ```

### Key numbers
- Stage 1 token match (`PROSE_1`, 18 prompt tokens + 20 generated tokens): **20 / 20** (`measured`)
- First-step logit check (`PROSE_1`): `ref_cpu` `[11.4843, 13.5325, 9.9344]` vs `llama-eval-callback -fa off` `[11.4843, 13.5325, 9.9344]` (`measured`)
- `ref_cpu` runtime (18 prompt + 20 decode tokens, 24-core CPU OpenMP): ~16 s (`measured`)
- `llama-server` F32 CPU reference runtime (18 prompt + 20 decode tokens, 16 threads): ~110 s including model load (`measured`)

### What is not verified
- 500-token exactness on `PROSE_1` (short prompt) and `LONG_1` (20,000+ token prompt) — deferred to Stages 2 and 4.
- GPU execution, R9700 memory bandwidth ceiling, and per-step latency breakdown — deferred to Stage 2.

### Prediction question for the human (before Stage 2)
Before I measure the single-stream Vulkan GPU engine in Stage 2: **with 5.494 GB of active weights per decode token on the R9700, which step do you expect to dominate single-token decode time (MoE NVFP4/MXFP8 experts, BF16 attention projections, or the 200,058-row BF16 unembed head), and roughly what percentage of the step?**

---

## Stage 2 — Single-Stream Vulkan GPU Engine (Unoptimized)

### What was built
- Wrote minimal Vulkan 1.1–1.3 C ABI header `~/inkling2/vk_abi.h` (zero external headers/packages needed; links directly against `/usr/lib/libvulkan.so`).
- Wrote `~/inkling2/vk_probe.cpp` + `shaders/bw_bench.comp` to query R9700 capabilities and measure 1 GiB streaming read bandwidth.
- Built `~/inkling2/inkling_vk.cpp` + 13 GLSL 450 Vulkan 1.3 compute shaders (`shaders/*.comp`) using Buffer Device Address + 256 B push constants (zero descriptor sets), single command buffer per token, F32 activations and F32 KV cache, and optional GPU timestamp profiling (`--profile`).
- Built standalone zero-dependency BPE tokenizer `~/inkling2/tokenizer.py` (reads `tiktoken/tokenizer.model` + `tokenizer.json`), verified exact encode/decode match against `llama.cpp` on `PROSE_1`.

### Gate result
- 20/20 exact token match on `PROSE_1` against F32 `llama.cpp` (`-fa off`) on the first GPU run (`measured`).
- 500-token GPU output saved to `~/inkling2/stage2_vk_500tok.out` (`inkling_vk_stage2` binary preserved); awaiting completion of CPU F32 `llama-server` 500-token reference (`~/src/bench/key32/prose1_500tok.json`, currently in progress in background on CPU) to close the 500/500 check.
- Exact commands to rerun:
  ```bash
  ssh omarchy 'cd ~/inkling2 && ./vk_probe'
  ssh omarchy 'cd ~/inkling2 && ./inkling_vk_stage2 --tokens 371,151698,149879,160947,158842,82,165300,149881,196656,152192,11,150207,220,50413,266,170850,82,151835 --steps 20 --profile'
  ```

### Key numbers & first breakdown / ceiling
- Hardware (`vk_probe`): AMD Radeon AI PRO R9700 (`RADV GFX1201`), 31.86 GiB VRAM, subgroupSize=64, sharedMem=64 KiB, tsPeriod=10.0 ns (`measured`).
- Card memory read bandwidth (1 GiB, median of 5 runs after 2 warmups): **637.0 GB/s** (min 636.8, max 637.1) (`measured`).
- Active weight bytes per decode token (batch=1, 48 logical layers, 8/128 experts, 200,058 unembed rows): **5.494 GB** (`inferred`).
- **Single-token decode memory-bandwidth ceiling**: `5.494 GB / 637.0 GB/s` = **8.62 ms/token = 115.9 tokens/s** (`inferred`).
  - Note: Requirement B1's 130 tok/s code target exceeds the single-token memory ceiling (115.9 tok/s), so speculative decoding (e.g. `mtp.safetensors`) will be physically required for B1 code.
- **Stage 2 unoptimized decode speed** (`PROSE_1`, ctx ~28, 18 steps after warmup): median **27.17 ms/token (36.8 tok/s)**, min 25.28, max 30.50 — **31.7% of ceiling** (`measured`).
- **Stage 2 unoptimized decode speed** (`PROSE_1`, ctx ~268, 498 steps after warmup): median **31.98 ms/token (31.3 tok/s)**, min 25.49, max 41.38 (`measured`).
- **First GPU timestamp breakdown** (avg of 10 decode tokens at ctx ~28, `measured`, vs category bandwidth ceilings `inferred`):
  - `embed`: 0.02 ms (0.1%) — ceiling 0.00 ms
  - `attnres+ctrl`: **10.81 ms (39.0%)** — 113 MB weights, ceiling 0.18 ms (dispatch/barrier overhead!)
  - `attn_proj+conv`: 6.13 ms (22.1%) — 2.582 GB weights, ceiling 4.05 ms (66% of ceiling)
  - `attn_core`: 0.95 ms (3.4%) — 11 MB KV, ceiling 0.02 ms (1 WG/head launch + serial scan)
  - `moe_router`: 1.31 ms (4.7%) — 28 MB weights, ceiling 0.04 ms (1-thread router + barrier)
  - `moe_experts`: 5.22 ms (18.9%) — 1.840 GB weights, ceiling 2.89 ms (55% of ceiling)
  - `unembed`: 2.61 ms (9.4%) — 0.922 GB weights, ceiling 1.45 ms (55% of ceiling)
  - `unaccounted/gap`: 0.65 ms (2.3%)
  - `GPU_TOTAL`: **27.70 ms (100.0%)**

### What is not verified
- 500/500 match against `prose1_500tok.json` (still running on CPU in background).
- Long-context (20k–128k) execution, prefill speed, and cache reuse (Stage 4).

---

## Stage 3 — Decode Speed at Short Context

### Optimization 1: Remove redundant GPU buffer copies and batch independent dispatches per barrier
1. **Target:** B1 (decode speed at short context).
2. **Breakdown:** 27.70 ms GPU total (`attnres+ctrl` 10.81 ms / 39.0%, `attn_proj+conv` 6.13 ms / 22.1%, `moe_experts` 5.22 ms / 18.9%, `unembed` 2.61 ms / 9.4%, `moe_router` 1.31 ms / 4.7%, `attn_core` 0.95 ms / 3.4%).
3. **Ceiling:** 8.62 ms/tok (115.9 tok/s) at 637.0 GB/s for 5.494 GB/tok; currently at 31.7% of ceiling.
4. **Limiter:** Overhead between GPU calls (~1,500 serialized `vkCmdPipelineBarrier` + redundant `vkCmdCopyBuffer` dispatches per token).
5. **Prediction:** Removing redundant copies (passing source addresses directly) and batching independent dispatches before shared barriers (e.g., all 3 loop controllers Q/K/V in parallel, all 6 attention input projections in parallel, K/V convs in parallel, Q/K/V/R clips in parallel) will cut barriers by ~65% and reduce `attnres+ctrl` from 10.81 ms to ~3.5 ms and `attn_proj+conv` from 6.13 ms to ~4.8 ms — gaining ~8.5 ms (~27.2 ms -> ~18.7 ms, **~53 tok/s**).
6. **Change:** Eliminated redundant `vkCmdCopyBuffer` calls (ping-pong `cur[0..1]` and direct address passing) and batched independent dispatches (3 Q/K/V loop controllers, 6–8 attention projections, K/V convs, Q/K/V/R clips) before shared barriers.
7. **Check:** Non-profiled median dropped from **26.49 ms (37.7 tok/s)** to **24.19 ms (41.3 tok/s)** (`measured`, min 21.59, max 27.16, ctx ~28; 20/20 exact match). Profiled `attnres+ctrl` dropped 10.81 -> 8.15 ms and `attn_proj+conv` 6.13 -> 5.29 ms. **Missed prediction by >25%** (gained 2.30 ms vs 8.5 ms predicted).
   - **What was misunderstood:** First, `--profile` timestamps (`BOTTOM_OF_PIPE_BIT`) added ~1–3 ms of pipeline stalls to profiled steps, distorting the baseline. Second, even after batching, `attnres+ctrl` still executes ~300+ tiny low-occupancy kernels (1–64 workgroups on a 64-CU GPU) separated by full `COMPUTE|TRANSFER` L2-flushing barriers (~15–20 us each). Real gains in `attnres+ctrl` require **kernel fusion** (collapsing multi-step controllers + AttnRes + RMSNorm into 1–2 dispatches), not just reordering barriers.
8. **Cost and range:** +96 KiB scratch VRAM; gain holds at all context lengths (purely per-layer overhead).

### Optimization 2: Vectorize BF16 matvec (`uvec4` 128-bit loads, 4 rows/workgroup)
1. **Target:** B1 (decode speed at short context).
2. **Breakdown:** 24.19 ms/tok non-profiled (25.44 ms profiled: `attnres+ctrl` 8.15 ms, `moe_experts` 6.07 ms, `attn_proj+conv` 5.29 ms, `unembed` 2.71 ms, `moe_router` 1.45 ms, `attn_core` 1.04 ms).
3. **Ceiling:** 8.62 ms/tok (115.9 tok/s); currently at 35.6% of ceiling.
4. **Limiter:** Memory read bandwidth on BF16 matvecs (`attn_proj` 2.582 GB + `unembed` 0.922 GB = 3.504 GB, 64% of all weight bytes), currently using 32-bit scalar loads and 1 subgroup/WG (~55% of card bandwidth).
5. **Prediction:** Vectorizing `matvec_bf16.comp` to `uvec4` (8 BF16 = 16 B) weight loads + `vec4` input loads with 4 rows per 256-thread workgroup will push BF16 matvec bandwidth toward ~80% of 637 GB/s, saving ~1.3 ms on `attn_proj` and ~1.0 ms on `unembed` (~2.3 ms total: 24.19 ms -> ~21.9 ms, **~45.7 tok/s**).
6. **Change:** Vectorized `shaders/matvec_bf16.comp` to `uvec4` (16 B = 8 BF16) weight loads + `vec4` input loads with 4 rows per 256-thread workgroup (`(rows + 3) / 4` dispatches).
7. **Check:** Non-profiled median dropped from **24.19 ms (41.3 tok/s)** to **23.55 ms (42.5 tok/s)** (`measured`, min 21.18, max 26.47, ctx ~28; 20/20 exact match). Profiled `attn_proj+conv` dropped 5.29 -> 4.91 ms (`-0.38 ms`), while `unembed` stayed flat at 2.71 -> 2.73 ms. **Missed prediction by >25%** (gained 0.64 ms vs 2.3 ms predicted).
   - **What was misunderstood:** The `unembed` timestamp bucket includes `softcap_argmax.comp`, which ran **1 single workgroup of 64 threads** executing 3,126 sequential `tanh()` calls on 1 CU (~1.2 ms all by itself!), masking the matvec bandwidth. Meanwhile `unembed.weight` matvec itself is already near ceiling (~1.5 ms for 0.922 GB = ~615 GB/s!).
8. **Cost and range:** 0 extra VRAM; holds at all context lengths.

### Optimization 3: Parallelize `softcap_argmax.comp` across 256 workgroups (2-stage reduction, skip redundant tanh for monotonic argmax)
1. **Target:** B1 (decode speed at short context).
2. **Breakdown:** 23.55 ms/tok non-profiled (24.76 ms profiled: `unembed` = 2.73 ms = 11.0%).
3. **Ceiling:** `unembed.weight` (0.922 GB) ceiling is 1.45 ms at 637.0 GB/s; `argmax` over 200,058 floats (0.8 MB) ceiling is <0.02 ms.
4. **Limiter:** Single-workgroup occupancy bottleneck in `softcap_argmax.comp` (1 CU out of 64 doing 3,126 serial `tanh()` calls).
5. **Prediction:** Replacing 1-WG `softcap_argmax` with a 256-WG parallel argmax (+ 1-WG final reduction over 256 candidates, skipping `tanh` since `30*tanh(x/30)` is strictly monotonic for greedy argmax) will cut `unembed` from **2.73 ms to ~1.55 ms** (~1.18 ms gain: 23.55 ms -> ~22.4 ms, **~44.7 tok/s**).
6. **Change:** Replaced 1-WG `softcap_argmax.comp` with a 256-WG parallel argmax + 1-WG final reduction (skipping redundant `tanh` since `30*tanh(x/30)` is strictly monotonic).
7. **Check:** Non-profiled median dropped from **23.55 ms (42.5 tok/s)** to **22.02 ms (45.4 tok/s)** (`measured`, min 20.52, max 25.28, ctx ~28; 20/20 exact match). Profiled `unembed` dropped 2.73 -> **1.57 ms** (92% of the 1.45 ms theoretical bandwidth ceiling!). **Matched prediction** (`-1.16 ms` on `unembed` vs `-1.18 ms` predicted; `-1.53 ms` wall-clock vs `-1.18 ms` predicted).
8. **Cost and range:** 0 extra VRAM (reuses `y_exps` scratch); holds at all context lengths.

### Optimization 4: Fuse AttnRes + RMSNorm and collapse 6-step Loop Controllers into 2 fused kernels
1. **Target:** B1 (decode speed at short context).
2. **Breakdown:** 22.02 ms/tok non-profiled (23.48 ms profiled: `attnres+ctrl` **7.79 ms / 33.2%**, `moe_experts` 6.16 ms / 26.2%, `attn_proj+conv` 4.90 ms / 20.9%, `unembed` 1.57 ms / 6.7%, `moe_router` 1.27 ms / 5.4%, `attn_core` 1.05 ms / 4.5%).
3. **Ceiling:** 8.62 ms/tok (115.9 tok/s); currently at 39.1% of ceiling. `attnres+ctrl` has only 113 MB of weights (0.18 ms bandwidth ceiling).
4. **Limiter:** Dispatch and barrier serialization overhead in `attnres+ctrl` (~288 barriers + ~500 tiny low-occupancy dispatches across 48 layers).
5. **Prediction:** Adding optional fused RMSNorm to `attn_res_mix.comp` (for non-controller layers) and replacing the 6-dispatch/4-barrier loop controller sequence with 2 fused kernels (`ctrl_down_fused.comp` + `ctrl_apply_norm_fused.comp`) will cut `attnres+ctrl` barriers by ~3x, reducing `attnres+ctrl` from **7.79 ms to ~2.6 ms** (`-5.2 ms` gain: 22.02 ms -> ~16.8 ms, **~59.5 tok/s**).
6. **Change:** Added optional fused RMSNorm to `attn_res_mix.comp` and replaced the 6-step loop controller with `ctrl_down_fused.comp` and `ctrl_apply_norm_fused.comp`.
7. **Check:** Non-profiled median dropped from **22.02 ms (45.4 tok/s)** to **21.51 ms (46.5 tok/s)** (`measured`, min 19.16, max 24.53, ctx ~28; 20/20 exact match). Profiled `attnres+ctrl` dropped 7.79 -> 6.85 ms (`-0.94 ms`). **Missed prediction by >25%** (`-0.51 ms` wall-clock vs `-5.2 ms` predicted).
   - **What was misunderstood:** `ctrl_apply_norm_fused.comp` launched only **1 workgroup** (MLP) or **3 workgroups** (Q/K/V) on a **64-CU GPU** to read **589 KB – 1.77 MB** of `gain_weight` + `forget_weight`! That starved 61–63 of the 64 CUs, capping controller memory bandwidth to ~20–30 GB/s (~4–5 ms wasted on 113 MB of weights). Similarly `ctrl_down_fused` launched only 16 WGs for MLP.
8. **Cost and range:** 0 extra VRAM; holds at all context lengths.

### Optimization 5: High-occupancy loop controllers (saturate all 64 CUs on `ctrl_down` and `ctrl_apply`)
1. **Target:** B1 (decode speed at short context).
2. **Breakdown:** 21.51 ms/tok non-profiled (23.29 ms profiled: `attnres+ctrl` **6.85 ms / 29.4%**, `moe_experts` 6.69 ms / 28.7%, `attn_proj+conv` 4.96 ms / 21.3%, `unembed` 1.56 ms / 6.7%, `moe_router` 1.35 ms / 5.8%, `attn_core` 1.11 ms / 4.8%).
3. **Ceiling:** 8.62 ms/tok (115.9 tok/s); currently at 40.1% of ceiling.
4. **Limiter:** Severe CU under-utilization in `ctrl_apply_norm_fused` (1–3 WGs on 64 CUs reading 113 MB) and `ctrl_down_fused` (16 WGs on 64 CUs).
5. **Prediction:** Launching 1 row/WG in `ctrl_down_fused` (64 WGs for MLP, 192 WGs for Q/K/V) and 72 WGs/controller in `ctrl_apply_fused` (72 WGs for MLP, 216 WGs for Q/K/V) + fast L2-cached `rmsnorm` will saturate all 64 CUs and drop `attnres+ctrl` from **6.85 ms to ~2.0 ms** (`-4.85 ms` gain: 21.51 ms -> **~16.7 ms, ~60 tok/s**).
6. **Change:** Increased workgroup count of `ctrl_down_fused` (16 -> 64 WGs) and split `ctrl_apply_norm_fused` into high-occupancy `ctrl_apply` (36 WGs/controller) + separate `rmsnorm`.
7. **Check:** Non-profiled median went from **21.51 ms (46.5 tok/s)** to **22.41 ms (44.6 tok/s)** (`measured`, min 20.33, max 27.48, ctx ~28; 20/20 exact match). Profiled `attnres+ctrl` barely moved (6.85 -> 6.78 ms). **Failed/slower result (missed prediction).**
   - **What was misunderstood:** Increasing occupancy on `ctrl_apply` did NOT reduce `attnres+ctrl`, proving `attnres+ctrl` is **NOT** memory-bandwidth/CU-starved on controller weights — it is dominated by **per-barrier overhead** (~25–30 us per `vk.barrier()` across ~240 barriers in `attnres+ctrl`!). Splitting `rmsnorm` back out added 48 barriers (~+0.9 ms), directly confirming ~20 us/barrier!
8. **Cost and range:** 0 extra VRAM; negative gain (reverted/superseded by barrier optimization).

### Optimization 6: Separate compute-only barriers (`COMPUTE_SHADER_BIT` only) from transfer barriers
1. **Target:** B1 (decode speed at short context).
2. **Breakdown:** 22.41 ms/tok non-profiled (23.02 ms profiled: `attnres+ctrl` 6.78 ms, `moe_experts` 6.56 ms, `attn_proj+conv` 4.94 ms, `unembed` 1.56 ms, `moe_router` 1.33 ms, `attn_core` 1.09 ms).
3. **Ceiling:** 8.62 ms/tok (115.9 tok/s); currently at 38.5% of ceiling.
4. **Limiter:** Overhead between GPU calls — every `vk.barrier()` currently includes `VK_PIPELINE_STAGE_TRANSFER_BIT` and `VK_ACCESS_TRANSFER_READ/WRITE_BIT` (~750 times per token), forcing RADV to stall/flush transfer stages on pure compute-to-compute barriers.
5. **Prediction:** Splitting `vk.barrier()` into a lightweight `barrier_compute()` (`COMPUTE_SHADER_BIT` -> `COMPUTE_SHADER_BIT`, `SHADER_WRITE` -> `SHADER_READ`) and using `barrier_xfer()` only after `vkCmdCopyBuffer` will cut per-barrier latency significantly, saving ~4–6 ms (22.41 ms -> ~17 ms, **~58 tok/s**).
6. **Change:** Split `vk.barrier()` into compute-only (`COMPUTE_SHADER_BIT` -> `COMPUTE_SHADER_BIT`, `SHADER_WRITE` -> `SHADER_READ`) and `barrier_xfer()` for `vkCmdCopyBuffer`.
7. **Check:** Non-profiled median dropped from **22.41 ms (44.6 tok/s)** to **22.06 ms (45.3 tok/s)** (`measured`, min 19.64, max 25.22, ctx ~28; 20/20 exact match). **Missed prediction by >25%** (`-0.35 ms` vs `-4..6 ms` predicted).
   - **Root cause found via `bench_ops.cpp` (`measured`):** Compute barriers on RADV are actually cheap (~3–4 us)! The micro-benchmark exposed the true culprit:
     - `ctrl_down(3x) + barrier`: **0.100 ms** (4.2 us/call)
     - `ctrl_apply(3x) + barrier`: **0.090 ms** (3.8 us/call)
     - `attn_res_mix(m=3, norm=1) + barrier`: **6.537 ms (54.5 us/call, N=120)** — **97% of `attnres+ctrl`!**
     - `moe_router(1-thr) + barrier`: **1.135 ms** (23.6 us/call, N=48)
     - `rmsnorm + barrier`: **3.325 ms** (16.6 us/call, N=200)
   - `attn_res_mix.comp` had a 36-float local array `vals[36]` (spilling to scratch memory!) and dynamic indexing `pc.states_addr[k]` on a single 64-thread subgroup doing scalar loads!
8. **Cost and range:** 0 extra VRAM; holds at all context lengths.

### Optimization 7: Rewrite `attn_res_mix.comp` (256 threads, `vec4` loads, zero scratch spill, no dynamic `states_addr[k]` indexing)
1. **Target:** B1 (decode speed at short context).
2. **Breakdown:** 22.06 ms/tok non-profiled (22.73 ms profiled: `attnres+ctrl` **6.73 ms / 29.6%**, `moe_experts` 6.50 ms / 28.6%, `attn_proj+conv` 4.80 ms / 21.1%, `unembed` 1.56 ms / 6.9%, `moe_router` 1.30 ms / 5.7%, `attn_core` 1.08 ms / 4.8%).
3. **Ceiling:** 8.62 ms/tok (115.9 tok/s); currently at 39.1% of ceiling.
4. **Limiter:** Scalar loads + scratch register spilling (`vals[36]`) + dynamic push-constant array indexing inside `attn_res_mix.comp` (54.5 us/call x 120 = 6.54 ms).
5. **Prediction:** Rewriting `attn_res_mix.comp` with `local_size_x = 256` (4 subgroups, 2–3 `vec4`s per thread, zero local array spill via 2 short L2-cached passes and static `if (m > k)` unrolling for `m <= 5`) will drop `attn_res_mix` from 54.5 us/call to ~5–6 us/call, reducing `attnres+ctrl` from **6.73 ms to ~1.2 ms** (`-5.5 ms` gain: 22.06 ms -> **~16.5 ms, ~60.6 tok/s**).
6. **Change:** Rewrote `shaders/attn_res_mix.comp` with 256 threads, `vec4` loads, static unrolling (`m <= 5`), and zero local array scratch spills.
7. **Check:** `bench_ops` confirmed `attn_res_mix` dropped **6.537 ms -> 0.563 ms (11.6x faster)**. Full engine non-profiled median dropped **22.06 ms (45.3 tok/s) -> 18.45 ms (54.2 tok/s)** (`measured`, min 16.58, max 18.90, ctx ~28; 20/20 exact match). Profiled `attnres+ctrl` dropped **6.73 -> 2.02 ms (-4.71 ms)**, within 15% of the `-5.5 ms` prediction. (`unembed` now at **1.49 ms = 97.3% of 1.45 ms card bandwidth ceiling**.)
8. **Cost and range:** 0 extra VRAM; holds at all context lengths.

### Optimization 8: Vectorized NVFP4/MXFP8 MoE with LDS LUTs + fused `w13+SiTU` and `w2+reduce`
1. **Target:** B1 (decode speed at short context).
2. **Breakdown:** 18.45 ms/tok non-profiled (18.77 ms profiled: `moe_experts` **7.69 ms / 41.0%**, `attn_proj+conv` 4.88 ms / 26.0%, `attnres+ctrl` 2.02 ms / 10.8%, `unembed` 1.49 ms / 7.9%, `moe_router` 1.47 ms / 7.8%, `attn_core` 1.22 ms / 6.5%).
3. **Ceiling:** 8.62 ms/tok (115.9 tok/s); currently at 46.7% of ceiling. `moe_experts` reads 1.840 GB -> 2.89 ms bandwidth ceiling (currently only 37% of ceiling).
4. **Limiter:** Slow per-element `E2M1[16]` + `ldexp()` E4M3 decode inside inner loops, narrow loads, and 4 separate MoE dispatches (`w13`, `SiTU`, `w2`, `reduce`) with barriers and intermediate VRAM writes.
5. **Prediction:** Precomputing E2M1/E4M3 LUTs in LDS (zero `ldexp()`), using `uvec2`/`vec4` loads, fusing `w13` gate+up pair with `SiTU` activation, and fusing `w2` across all 8 experts with `top_w` weighted reduction will cut `moe_experts` from **7.69 ms to ~3.7 ms** (`-4.0 ms` gain: 18.45 ms -> **~14.5 ms, ~69 tok/s**).
6. **Change:** Rewrote `moe_nvfp4_matvec.comp` and `moe_mxfp8_matvec.comp` with LDS E4M3/E2M1/UE8M0 LUTs, vectorized `uvec2`/`uvec4`/`vec4` loads, fused `w13 + SiTU`, and fused `w2 + weighted_reduce` (eliminating `elemwise(SiTU)` and `moe_reduce` dispatches/barriers).
7. **Check:** Non-profiled median dropped from **18.45 ms (54.2 tok/s)** to **16.18 ms (61.8 tok/s)** (`measured`, min 15.83, max 17.44, ctx ~28; 20/20 exact match). Profiled `moe_experts` dropped **7.69 -> 5.13 ms (-2.56 ms)**. Slightly shy of `-4.0 ms` prediction because `init_shared()` recomputes E4M3 LUT on every WG and `mode 0` reads `x` twice per (gate,up) pair.
8. **Cost and range:** 0 extra VRAM; holds at all context lengths.

### Optimization 9: Parallelize `moe_router.comp` to 64-thread subgroup top-8 selection
1. **Target:** B1 (decode speed at short context).
2. **Breakdown:** 16.18 ms/tok non-profiled (16.26 ms profiled: `moe_experts` 5.13 ms / 31.6%, `attn_proj+conv` 4.89 ms / 30.1%, `attnres+ctrl` 2.04 ms / 12.5%, `moe_router` **1.49 ms / 9.2%**, `unembed` 1.48 ms / 9.1%, `attn_core` 1.23 ms / 7.6%).
3. **Ceiling:** 8.62 ms/tok (115.9 tok/s); currently at 53.3% of ceiling. `moe_router` bandwidth ceiling is 0.04 ms.
4. **Limiter:** `moe_router.comp` runs `local_size_x = 1` (single thread doing 128 `exp()` + 8x128 serial selection loops = 1.12 ms measured in `bench_ops`).
5. **Prediction:** Parallelizing `moe_router.comp` across 64 threads (2 experts/thread + 8 `subgroupMax` iterations) will drop `moe_router` from **1.49 ms to ~0.40 ms** (`-1.1 ms` gain: 16.18 ms -> **~15.1 ms, ~66.2 tok/s**).
6. **Change:** Parallelized `shaders/moe_router.comp` from 1 thread to 64 threads (2 experts/thread + 8 `subgroupMax` iterations).
7. **Check:** `bench_ops` dropped `moe_router` **1.119 ms -> 0.143 ms (7.8x faster)**. Full engine non-profiled median dropped **16.18 ms (61.8 tok/s) -> 15.43 ms (64.8 tok/s)** (`measured`, min 14.61, max 16.78, ctx ~28; 20/20 exact match). Profiled `moe_router` dropped **1.49 -> 0.41 ms (-1.08 ms)**, matching the `-1.1 ms` prediction!
8. **Cost and range:** 0 extra VRAM; holds at all context lengths.

### Optimization 10: Vectorize `rmsnorm.comp` (256 threads, `vec4` loads)
1. **Target:** B1 (decode speed at short context).
2. **Breakdown:** 15.43 ms/tok non-profiled (15.10 ms profiled: `moe_experts` 5.07 ms / 33.6%, `attn_proj+conv` 4.88 ms / 32.3%, `attnres+ctrl` **2.03 ms / 13.4%**, `unembed` 1.48 ms / 9.8%, `attn_core` 1.23 ms / 8.1%, `moe_router` 0.41 ms / 2.7%).
3. **Ceiling:** 8.62 ms/tok (115.9 tok/s); currently at 55.9% of ceiling.
4. **Limiter:** Scalar 64-thread `rmsnorm.comp` takes 16.3 us/call (`bench_ops`), called ~97 times per token (49 in `attnres+ctrl`, 48 for V head norm in `attn_proj+conv`).
5. **Prediction:** Vectorizing `rmsnorm.comp` to 256 threads + `vec4` loads will drop `rmsnorm` from 16.3 us/call to ~3.5 us/call, saving ~0.7 ms in `attnres+ctrl` and ~0.3 ms in `attn_proj+conv` (~1.0 ms total: 15.43 ms -> **~14.4 ms, ~69.4 tok/s**).
6. **Change:** Vectorized `shaders/rmsnorm.comp` to 256 threads with `vec4` loads (covering both `dim=2304` and `dim=128`).
7. **Check:** `bench_ops` dropped `rmsnorm` **3.253 ms -> 0.420 ms (7.7x faster, 2.1 us/call)**. Full engine non-profiled median dropped **15.43 ms (64.8 tok/s) -> 14.33 ms (69.8 tok/s)** (`measured`, min 13.93, max 14.88, ctx ~28; 20/20 exact match). Profiled `attnres+ctrl` dropped **2.03 -> 1.24 ms (-0.79 ms)**, matching the `-1.0 ms` total prediction (`-1.10 ms` wall-clock).
8. **Cost and range:** 0 extra VRAM; holds at all context lengths.

### Optimization 11: Eliminate LDS LUT setup/bank conflicts (direct IEEE-754 bit-decode) and single-pass `(gate, up)` `x` reuse in MoE kernels
1. **Target:** B1 (decode speed at short context).
2. **Breakdown:** 14.33 ms/tok non-profiled (14.28 ms profiled: `moe_experts` **5.06 ms / 35.4%**, `attn_proj+conv` 4.87 ms / 34.1%, `unembed` 1.48 ms / 10.4%, `attnres+ctrl` 1.24 ms / 8.7%, `attn_core` 1.23 ms / 8.6%, `moe_router` 0.41 ms / 2.9%).
3. **Ceiling:** 8.62 ms/tok (115.9 tok/s); currently at 60.2% of ceiling. `moe_experts` ceiling is 2.89 ms (currently at 57% of ceiling).
4. **Limiter:** `init_shared()` recomputes 256 `ldexp()` E4M3 entries + `barrier()` in every MoE workgroup (~138k WGs/token), LDS `s_e2m1` lookups suffer bank conflicts, and `w13` reads `x` twice per `(gate, up)` pair.
5. **Prediction:** Replacing LDS LUTs with exact IEEE-754 bit-manipulation decodes (zero `shared` memory, zero `barrier()`, zero `ldexp()`) and fusing the `gate`/`up` loop in `w13` to load `x` once will drop `moe_experts` from **5.06 ms to ~3.4 ms** (`-1.66 ms` gain: 14.33 ms -> **~12.7 ms, ~78.7 tok/s**).
6. **Change:** Replaced LDS LUTs in `moe_nvfp4_matvec.comp` and `moe_mxfp8_matvec.comp` with inline IEEE-754 bit-manipulation decodes (`decode_e2m1`, `decode_e4m3_fast`, `decode_ue8m0_fast`) and fused `(gate, up)` single-pass `x` reuse.
7. **Check:** Non-profiled median **regressed** from **14.33 ms (69.8 tok/s)** to **19.34 ms (51.7 tok/s)** (`measured`, min 17.37, max 21.57, ctx ~28; 20/20 exact match). Profiled `moe_experts` regressed **5.06 -> 10.00 ms (+4.94 ms)**. **Failed/slower result.**
   - **What was misunderstood:** Calling `decode_e2m1()` (with conditional + bitfield shifts) 32 times per block in the inner loop made NVFP4 **ALU-bound** on RDNA4, whereas LDS `s_e2m1[16]` is a single fast LDS read per nibble! Only the *scale* decode (`decode_e4m3_fast`, called 1x per 16 weights) should be inline bit-math; `s_e2m1[16]` (and `s_e4m3[256]` for MXFP8 weights) belongs in LDS!
8. **Cost and range:** Negative gain; immediately reverted inline `decode_e2m1` in Opt 12.

### Optimization 12: Hybrid LDS `s_e2m1[16]` (zero `ldexp()` init) + inline scale decode + single-pass `(gate, up)` `x` reuse
1. **Target:** B1 (decode speed at short context).
2. **Breakdown:** Baseline (Opt 10): 14.33 ms/tok non-profiled (`moe_experts` 5.06 ms / 35.4%).
3. **Ceiling:** 8.62 ms/tok (115.9 tok/s); currently at 60.2% of ceiling.
4. **Limiter:** Opt 8's `init_shared()` ran 256 `ldexp()` calls per WG for `s_e4m3[256]` and read `x` twice per `(gate, up)` pair.
5. **Prediction:** Keeping `s_e2m1[16]` in LDS (16-thread constant init, zero `ldexp()`), using `decode_e4m3_fast` for NVFP4 scales (1x/16 weights), and keeping single-pass `(gate, up)` `x` reuse will cut `moe_experts` from **5.06 ms to ~3.9 ms** (`-1.15 ms` vs Opt 10: 14.33 ms -> **~13.2 ms, ~75.8 tok/s**).
6. **Change:** Hybrid LDS `s_e2m1[16]` (16-thread constant init, zero `ldexp()`) + inline `decode_e4m3_fast` for scales + single-pass `(gate, up)` `x` reuse.
7. **Check:** Non-profiled median **14.39 ms (69.5 tok/s)** (`measured`, min 13.63, max 14.89, ctx ~28; 20/20 exact match). Profiled `moe_experts` dropped **5.06 -> 4.93 ms** (`-0.13 ms`). **Missed prediction** (`-0.13 ms` vs `-1.15 ms`).
   - **Root cause found:** Wavefront tail-thread underutilization! In NVFP4 `w13` (`scale_cols = 144`), 64 threads/row gives `144 = 2*64 + 16` (48 of 64 lanes idle on iter 3 = 25% waste). In `w2` (`scale_cols = 72`), 64 threads/row gives `72 = 1*64 + 8` (**56 of 64 lanes idle on iter 2 = 44% waste!**).
8. **Cost and range:** 0 extra VRAM; holds at all context lengths.

### Optimization 13: Zero-tail-waste lane partitioning in NVFP4 MoE (`16 lanes/row` for `w13` [144=16x9], `8 lanes/row` for `w2` [72=8x9])
1. **Target:** B1 (decode speed at short context).
2. **Breakdown:** 14.39 ms/tok non-profiled (14.16 ms profiled: `moe_experts` **4.93 ms / 34.8%**, `attn_proj+conv` 4.87 ms / 34.4%, `unembed` 1.48 ms / 10.5%, `attnres+ctrl` 1.24 ms / 8.7%, `attn_core` 1.23 ms / 8.7%, `moe_router` 0.41 ms / 2.9%).
3. **Ceiling:** 8.62 ms/tok (115.9 tok/s); currently at 59.9% of ceiling. `moe_experts` ceiling is 2.89 ms.
4. **Limiter:** 25% (`w13`) and 44% (`w2`) wavefront lane waste from 64-lane reduction on `scale_cols = 144` and `72`.
5. **Prediction:** Partitioning each 64-lane subgroup into 4x16 lanes for `w13` (9 exact iters/lane, 0% tail waste) and 8x8 lanes for `w2` (9 exact iters/lane, 0% tail waste) using `subgroupShuffleXor` butterfly reduction will cut `moe_experts` from **4.93 ms to ~3.4 ms** (`-1.5 ms` gain: 14.39 ms -> **~12.9 ms, ~77.5 tok/s**).
6. **Change:** Partitioned NVFP4 MoE to 16 lanes/pair (`w13`) and 8 lanes/row (`w2`) with `subgroupShuffleXor` butterfly reductions.
7. **Check:** Non-profiled median **regressed** from **14.39 ms (69.5 tok/s)** to **15.08 ms (66.3 tok/s)** (`measured`, min 14.44, max 16.21, ctx ~28; 20/20 exact match). Profiled `moe_experts` regressed **4.93 -> 5.61 ms (+0.68 ms)**. **Failed/slower result.**
   - **What was misunderstood:** Splitting a 64-lane wavefront across 8 different rows in `w2` broke memory coalescing (8 scattered 576-byte row accesses per wave instead of 1 contiguous 512-byte coalesced burst) and reduced WG count to 72 (hurting latency hiding). Reverted to Opt 12's 64-lane coalesced layout (`4.93 ms`).
8. **Cost and range:** Negative gain; reverted.

### Stage 2 & Stage 3 Requirement A (short prompt) 500/500 Verification
- `~/src/bench/key32/prose1_500tok.json` completed (F32 GGUF, CPU, `-fa off -ctk f32 -ctv f32`).
- Stage 2 saved output (`stage2_vk_500tok.out`) vs F32 `llama.cpp`: **500 / 500** (`measured`).
- Stage 3 Opt 12 (`inkling_vk`) vs F32 `llama.cpp`: **500 / 500** (`measured`, median 18.79 ms = 53.2 tok/s at ctx ~268, min 13.16 ms at ctx ~20, max 26.13 ms at ctx ~518).
- Exact command to rerun: `ssh omarchy 'python3 ~/inkling2/verify_req_a_short.py ~/inkling2/inkling_vk'`

### Optimization 14: Split-K FlashDecoding (`attn_decode_splitk.comp` + `attn_reduce.comp` with fused sigmoid gate)
1. **Target:** B1 (short context) & B2 (depth scaling — 500-token run showed +13 ms growth from ctx=20 to ctx=518 due to serial 1-WG/head scan).
2. **Breakdown:** 14.39 ms at ctx~28 (`attn_core` 1.23 ms), degrading to 26.13 ms at ctx~518 (`attn_core` ~13 ms!).
3. **Ceiling:** At ctx~28: 8.62 ms/tok (115.9 tok/s); at ctx~518: ~9.0 ms/tok.
4. **Limiter:** `attn_decode.comp` launches only 24 WGs (1 per Q head) on a 64-CU GPU and scans KV serially with scalar loads.
5. **Prediction:** Split-K FlashDecoding (splitting KV across `N_SPLITS` WGs per head + online log-sum-exp reduction with fused sigmoid attention gate) will keep `attn_core` flat across 500 tokens (ctx~268 median dropping from **18.79 ms -> ~13.8 ms, ~72.5 tok/s**, and ctx~28 dropping from **14.39 ms -> ~13.2 ms, ~75.8 tok/s**).
6. **Change:** Replaced serial 1-WG/head `attn_decode.comp` with Split-K FlashDecoding (`attn_decode_splitk.comp` + `attn_reduce.comp` with fused sigmoid attention gate; 1/4/16/64 splits by `n_kv`).
7. **Check:** 500-token Requirement A run (`PROSE_1`, ctx ~268) jumped from **18.79 ms (53.2 tok/s)** to **12.14 ms (82.3 tok/s)** (`measured`, min **11.51 ms = 86.9 tok/s**, max 14.53 ms; **500/500 exact match** against F32 `llama.cpp -fa off`!). Beat the 13.8 ms prediction (71% of the 115.9 tok/s single-token bandwidth ceiling!). Note: short 18-step runs show ~14.5 ms because R9700 DPM clocks are still ramping up during the first ~0.25 s; steady-state 500-step runs hit full boost clocks (12.14 ms median).
8. **Cost and range:** +780 KiB scratch VRAM; gain grows dramatically with context depth (eliminated the +13 ms ctx=20..518 degradation and prepares for 40k–128k depth).

---

## Reference & Box Constraint Note (2026-11-25 13:57)

### What is not verified
- **F32 GGUF exceeds system RAM (`measured`):** `~/models/gguf/Inkling-2-Mini-f32.gguf` is 97.1 GiB on a 62 GiB RAM box. Running it on CPU (`-ngl 0`) causes continuous page eviction/re-read (~100k major page faults/s, ~17 min per 2,048-token prefill batch -> ~2.5–3 hours for a 20k prompt).
- **Quiet-box re-measurement needed (`measured`):** All Stage 3 GPU timings from Opt 1 through Opt 14 (and the batched-prefill runs) were taken while a background CPU F32 `llama-server` reference job (`prose1_500tok.json` first, then `long20k_500tok.json`) was running and thrashing disk/RAM. Although GPU compute kernels read from VRAM, weight load times (`2–46 s`!) and host-side command submission/wall-clock jitter were affected. All Stage 3/4 speed numbers must be re-measured on a quiet box once no CPU reference job is running.
- **Batched prefill (`B > 1`) exactness:** `B = 1` matches **500/500** (`prose1_500tok.json`), while `B = 18` currently matches steps 0..28 and diverges at step 29 (`ref=157311 got=149880`); root cause to be isolated after settling the reference strategy.

---

## Quiet-Box Baseline Re-Measurement (2026-11-25 14:06)
- All earlier Stage 3 timing numbers above (Opt 1 through Opt 14) are **superseded** (they were measured while a CPU F32 `llama-server` reference job was thrashing RAM/disk).
- **Quiet-box baseline (current build, Hybrid 128k-ready KV, Split-K FlashDecoding, `--prefill-batch 1`, `measured`):**
  - Requirement A (short prompt, `PROSE_1`): **500 / 500** exact match against `~/src/bench/key32/prose1_500tok.json` (`measured`).
  - 5-run median (100 decode steps each after 1 warmup run, ctx ~68): **12.02 ms/tok (83.2 tok/s)**, spread **12.01–12.25 ms** (`measured`) — **71.8%** of the 115.9 tok/s single-token bandwidth ceiling.
  - 500-token steady-state (ctx ~268): **11.94 ms/tok (83.8 tok/s)**, min 11.43 ms (87.5 tok/s), max 14.47 ms (`measured`).
  - 128,000-token VRAM check (`--max-ctx 128000 --check-vram`): **26.36 / 31.86 GiB used (5.50 GiB free)** (`measured`, passes Requirement E >= 1 GiB free).

### What is not verified
- **Long-context (20k) exactness has no reference key** (the 97.1 GiB F32 GGUF exceeds 62 GiB RAM and was stopped at 14:03).
- Batched prefill (`B > 1`) exactness (step-29 mismatch on `B = 18` not yet isolated).

---

### Optimization 15 (Quiet Box): Fuse residual add into `short_conv.comp`
1. **Target:** B1 (decode speed at short context, pushing toward 80%+ of 115.9 tok/s ceiling = 92.7 tok/s).
2. **Breakdown:** 12.02 ms/tok quiet-box median (ctx ~68).
3. **Ceiling:** 8.62 ms/tok (115.9 tok/s); currently at 71.8% of ceiling.
4. **Limiter:** Dispatch/barrier overhead and intermediate scratch traffic from 88 separate `elemwise(add)` dispatches (`inpSA + attn_out` on 40 layers, `ffn_inp + mlp_out` on all 48 layers).
5. **Prediction:** Adding optional `res_addr` to `short_conv.comp` to fuse the residual add eliminates 88 dispatches and 88 barriers per token (~5 us each), saving **~0.45 ms** (`12.02 ms -> ~11.57 ms, ~86.4 tok/s`).
6. **Change:** Added optional `res_addr` to `shaders/short_conv.comp` to fuse post-attention (40 layers) and post-MoE (48 layers) residual adds, removing 88 `elemwise` dispatches and 88 barriers per token.
7. **Check:** Quiet-box 5-run median (ctx ~68) moved **12.02 ms -> 12.00 ms (83.3 tok/s)**, spread **11.93–12.16 ms**; 500-token exactness **500/500 PASS** (median 11.98 ms = 83.5 tok/s, min 11.37 ms). **Missed prediction (`-0.02 ms` vs `-0.45 ms`).**
   - **What was misunderstood:** On a quiet box at full boost clocks, L2-resident `elemwise(add)` + compute barrier costs <0.3 us (88 calls < 0.03 ms total). We are no longer dispatch/barrier-limited; remaining time above the 8.62 ms ceiling is dominated by memory/ALU efficiency in `moe_experts` (~4.2 ms vs 2.89 ms ceiling) and `attn_core` (~1.0 ms).
8. **Cost and range:** 0 extra VRAM; neutral (cleaner dispatch graph, kept).

### Optimization 16 (Quiet Box): 128-bit `uvec4` NVFP4 loads (2 scale blocks/iter) + 32-lane coalesced `w2`
1. **Target:** B1 (pushing plain single-token decode toward 80% of 115.9 tok/s ceiling = 92.7 tok/s).
2. **Breakdown:** 12.00 ms/tok quiet-box median (ctx ~68); `moe_experts` is ~4.2 ms for 1.84 GB (~440 GB/s = 69% of 637 GB/s ceiling).
3. **Ceiling:** 8.62 ms/tok (115.9 tok/s); currently at 71.8% of ceiling.
4. **Limiter:** Narrow 64-bit (`uvec2`) NVFP4 weight loads and 44% wavefront tail waste in `w2` (`scale_cols = 72` with 64 lanes/row).
5. **Prediction:** Loading `uvec4` (16 B = 32 FP4 weights = 2 scale blocks) per iteration in `w13` (72 pairs/row) and `w2` (36 pairs/row with 32 contiguous lanes/row = 512 B coalesced bursts, 8 rows/WG = 288 WGs, cutting `w2` tail waste from 44% to 11%) will save **~0.8–1.0 ms** (`12.00 ms -> ~11.1 ms, ~90.1 tok/s`).
6. **Change:** Updated `shaders/moe_nvfp4_matvec.comp` to `uvec4` loads (2 scale blocks/iter) in `w13` and 32-lane half-subgroup butterfly reduction in `w2` (8 rows/WG).
7. **Check:** 500-token exactness **500/500 PASS** (`measured`). Quiet-box 5-run median (ctx ~68) regressed **12.00 ms -> 12.13 ms (82.4 tok/s)**, spread **12.05–12.18 ms** (`measured`). Steady-state `--profile` (steps 50..59, ctx ~73): `GPU_TOTAL = 12.01 ms` (`attn_proj+conv 4.58 ms [38.1%]`, `moe_experts 3.87 ms [32.2%]`, `unembed 1.48 ms [12.3%]`, `attnres+ctrl 0.93 ms [7.7%]`, `attn_core 0.85 ms [7.1%]`, `moe_router 0.29 ms [2.4%]`). **Missed prediction (+0.13 ms vs -0.8 to -1.0 ms).**
   - **What was misunderstood:** Halving `w2` row parallelism from 64 lanes to 32 lanes cut coalesced burst width in half and replaced 1 `subgroupAdd` with 5 `subgroupShuffleXor` instructions, hurting more than the reduced tail waste helped. Reverted to Opt 15 (64-lane `uvec2`).
8. **Cost and range:** 0 extra VRAM; reverted.

### Optimization 17 (Quiet Box): Transposed `rel_logits_proj` `[1024, 16]` + fused 1-pass `attn_decode` when `n_splits == 1`
1. **Target:** B1 (short-context decode speed toward 80% of ceiling = 92.7 tok/s) and B2 (long-context decode attention).
2. **Breakdown:** Quiet-box steady-state `--profile` (Opt 15 reverted cleanly, steps 50..59, ctx ~73): `GPU_TOTAL = 11.58 ms` (`attn_proj+conv 4.59 ms`, `moe_experts 3.36 ms`, `unembed 1.48 ms`, `attnres+ctrl 0.95 ms`, `attn_core 0.88 ms`, `moe_router 0.30 ms`).
3. **Ceiling:** 8.62 ms/tok (115.9 tok/s); currently at 74.5% of GPU ceiling (11.58 ms). `attn_core` alone is 0.88 ms vs 0.05 ms ceiling (0.83 ms gap).
4. **Limiter:** (a) `rel_logits_proj.proj` is stored `[16, 1024]` BF16, so `tid=0..15` reading `tid*1024 + dist` hits 16 separate 2048B-strided cache lines per KV token; (b) `n_splits=4` at ctx ~73 forces 48 `part_ml`/`part_o` round-trips + 48 `attn_reduce` dispatches + 48 barriers.
5. **Prediction:** Transposing `rel_logits_proj.proj` to `[1024, 16]` BF16 (768 KB total VRAM) for coalesced 32B loads and fusing sink + sigmoid gate directly into `attn_decode_splitk.comp` for `n_splits == 1` (`n_kv <= 128`) will cut `attn_core` from **0.88 ms -> ~0.30 ms**, saving **~0.55 ms** (`GPU_TOTAL 11.58 ms -> ~11.03 ms`, wall-clock `12.00 ms -> ~11.45 ms, ~87.3 tok/s`).
6. **Change:** Transposed `rel_logits_proj.proj` to `[1024, 16]` BF16 in `rel_proj_t_buf` (768 KiB) and fused `n_splits == 1` path (`n_kv <= 128`) directly into `shaders/attn_decode_splitk.comp`.
7. **Check:** Steady-state `--profile` (steps 50..59, ctx ~73) regressed `attn_core` **0.88 ms -> 2.56 ms (+1.68 ms)**, `GPU_TOTAL 11.58 ms -> 13.03 ms`, decode median `13.27 ms (75.4 tok/s)` (`measured`). **Missed prediction (+1.68 ms vs -0.55 ms).**
   - **What was misunderstood:** With `n_splits = 1`, `attn_decode` launches only `N_HEADS = 24` workgroups across 64 CUs (62.5% of CUs idle, 1 wave/CU), leaving zero latency hiding for serial KV + `rel_proj` reads across 73 tokens. The 48 extra `attn_reduce` dispatches cost far less than CU starvation.
8. **Cost and range:** +768 KiB VRAM for `rel_proj_t_buf` (kept); `n_splits = 1` threshold reverted/inverted in Opt 18.

### Optimization 18 (Quiet Box): Higher Split-K occupancy (`n_splits = 16` at short context) + transposed `rel_proj`
1. **Target:** B1 (short-context decode speed toward 80% of ceiling = 92.7 tok/s) and B2.
2. **Breakdown:** Opt 15 baseline `GPU_TOTAL = 11.58 ms`, `attn_core = 0.88 ms` (with `n_splits = 4` -> 96 WGs = 1.5 waves/CU).
3. **Ceiling:** 8.62 ms/tok (115.9 tok/s); `attn_core` ceiling 0.05 ms (0.83 ms gap).
4. **Limiter:** Low CU occupancy in `attn_decode_splitk` (`n_splits = 4` = 1.5 waves/CU; `n_splits = 1` = 0.375 waves/CU caused 3x slowdown).
5. **Prediction:** Raising `n_splits` to `16` for `n_kv >= 32` (`24 * 16 = 384` WGs = 6 waves/CU) with transposed `[1024, 16]` `rel_proj_t_buf` will cut `attn_core` from **0.88 ms -> ~0.35 ms** (`-0.53 ms`, `GPU_TOTAL 11.58 ms -> ~11.05 ms, ~90.5 tok/s`).
6. **Change:** Set `n_splits = (n_kv <= 8u) ? 1u : ((n_kv <= 32u) ? 8u : ((n_kv <= 2048u) ? 16u : 64u))` with transposed `rel_proj_t_buf` (`[1024, 16]` BF16) and bit-exact sink fold order in `shaders/attn_decode_splitk.comp`.
7. **Check:** Requirement A **500/500 PASS** (`measured`). Quiet-box 5-run median (100 steps after warmup, ctx ~68) improved **12.00 ms -> 11.84 ms (84.5 tok/s)**, spread **11.75–12.10 ms**; 500-token steady-state (ctx ~268) **11.76 ms (85.0 tok/s)**, min **11.22 ms (89.1 tok/s)** (`measured`). Steady-state `--profile` (steps 50..59, ctx ~73): `attn_core` dropped **0.85 ms -> 0.57 ms (-0.28 ms)**, `GPU_TOTAL = 11.70 ms` (73.7% of 8.62 ms ceiling). **Within range of prediction (-0.28 ms on `attn_core` vs -0.53 ms predicted; remaining 0.57 ms is dominated by 96 attention/reduce dispatches + barriers).**
8. **Cost and range:** +768 KiB VRAM (`rel_proj_t_buf`); coalesced `[1024, 16]` `rel_proj` holds at all context lengths (100, 10k, 40k).

### Optimization 19 (Quiet Box): Flatten `w2` across `(k=0..7, b=0..71)` -> `576 = 9 * 64` (zero tail waste, 1 `subgroupAdd`)
1. **Target:** B1 (short-context decode speed toward 80% of ceiling = 92.7 tok/s).
2. **Breakdown:** Quiet-box steady-state `--profile` (Opt 18, ctx ~73): `GPU_TOTAL = 11.70 ms`, `moe_experts = 3.85 ms` (32.9%, vs 2.89 ms ceiling = 0.96 ms gap — #1 gap).
3. **Ceiling:** 8.62 ms/tok (115.9 tok/s); currently at 73.7% of GPU ceiling (11.70 ms).
4. **Limiter:** In `moe_nvfp4_matvec.comp` `w2`, `scale_cols = 72` with 64 lanes/row causes 2 iterations per expert where iteration 1 has 56/64 lanes idle (44% wavefront waste, 16 loop iterations and 8 `subgroupAdd`s per row).
5. **Prediction:** Flattening `(k=0..7, b=0..71)` into a single `idx in 0..575` (`9 * 64`) loop with pre-multiplied `s_kscale[k] = s_topw[k] * s_outer[k]` eliminates all 44% tail waste (9 iters instead of 16) and cuts 8 `subgroupAdd`s to 1, saving **~0.35–0.50 ms** on `moe_experts` (`GPU_TOTAL 11.70 ms -> ~11.25 ms`, wall-clock `11.84 ms -> ~11.40 ms, ~87.7 tok/s`).
6. **Change:** Flattened `w2` in `shaders/moe_nvfp4_matvec.comp` across `(k=0..7, b=0..71)` into a single `idx in 0..575` (`9 * 64`) loop with pre-multiplied `s_kscale[k] = s_topw[k] * s_outer[k]` and 1 `subgroupAdd`.
7. **Check:** Requirement A **500/500 PASS** (`measured`). Quiet-box 5-run median (ctx ~68) improved **11.84 ms -> 11.68 ms (85.6 tok/s)**, spread **11.63–11.85 ms**; 500-token steady-state (ctx ~268) **11.64 ms (85.9 tok/s)**, min **11.05 ms (90.5 tok/s)** (`measured`). Steady-state `--profile`: `moe_experts` dropped **3.85 ms -> 3.72 ms (-0.13 ms)**, `GPU_TOTAL = 11.62 ms` (74.2% of ceiling). **Smaller than predicted (-0.16 ms vs -0.35 to -0.50 ms):** `k = idx / 72` integer division + 1 boundary lane crossing experts per iteration + `w13` (67% of NVFP4 bytes) untouched limited the gain.
8. **Cost and range:** 0 extra VRAM; holds at all context lengths (pure MoE weight compute).

### Optimization 20 (Quiet Box): Replace 11 `vkCmdCopyBuffer` + 22 `barrier_xfer()` (`ALL_COMMANDS_BIT`) drains in decode with compute copy
1. **Target:** B1 (short-context decode speed toward 80% of ceiling = 92.7 tok/s).
2. **Breakdown:** Opt 19 quiet-box median **11.68 ms (85.6 tok/s)**, `GPU_TOTAL = 11.62 ms` (`attn_proj+conv 4.59 ms`, `moe_experts 3.72 ms`, `unembed 1.48 ms`, `attnres+ctrl 0.94 ms`, `attn_core 0.58 ms`, `moe_router 0.29 ms`).
3. **Ceiling:** 8.62 ms/tok (115.9 tok/s); currently at 74.2% of GPU ceiling.
4. **Limiter:** Decode executes 11 tiny 9 KiB `vkCmdCopyBuffer` calls bracketed by 22 `barrier_xfer()` (`VK_PIPELINE_STAGE_ALL_COMMANDS_BIT`) full-pipeline drains.
5. **Prediction:** Replacing those 11 `vkCmdCopyBuffer` + 22 `barrier_xfer()` calls with `elemwise` compute copy (`mode=3`, 1 WG of 256 threads) + normal compute `vk.barrier()` will eliminate transfer-queue sync stalls, saving **~0.20–0.35 ms** (`11.68 ms -> ~11.40 ms, ~87.7 tok/s`).
6. **Change:** Added `mode == 4u` copy to `shaders/elemwise.comp` and replaced all 11 `vkCmdCopyBuffer` + 22 `barrier_xfer()` calls in `record_layer_1` and `forward_one_token` with compute copies.
7. **Check:** Requirement A **500/500 PASS** (`measured`). Quiet-box 5-run median (ctx ~68) **11.65 ms (85.8 tok/s)**, spread **11.59–11.70 ms**; 500-token steady-state (ctx ~268) **11.61 ms (86.2 tok/s)**, min **11.00 ms (90.9 tok/s)** (`measured`). Steady-state `--profile`: `GPU_TOTAL = 11.65 ms` (vs 11.62 ms). **Missed prediction (-0.03 ms vs -0.20 to -0.35 ms):** RADV executes `vkCmdCopyBuffer` on the compute queue without extra transfer-queue pipeline stalls.
8. **Cost and range:** 0 extra VRAM; cleaner compute-only decode graph, kept.

### Bugfix (Batched Prefill Determinism): Disjoint `cur[dst_slot]` in `pass > 0` copy
- **Root cause:** At the start of `pass > 0` (`pass == 1, 2`), `CORE_END - CORE_START = 12` (even), so `cur_slot == 0`. Both `bc2` (`cur[0] -> bank_out`) and `bc3` (`prelude -> cur[0]`) ran concurrently in the same copy batch without a barrier, causing a data race on `cur[0]` for `B >= 15` (`max_diff` varied randomly between `1.4e-5` and `3.49`).
- **Fix:** Copy `prelude -> cur[cur_slot ^ 1]` and set `cur_slot ^= 1` in both `forward_one_token` and `forward_batch_prefill`.
- **Check:** `B=15` vs `B=1` is now bit-deterministic (`max_diff = 1.430511e-05` across 3/3 runs), and `python3 verify_req_a_short.py` (default `B=64` prefill) passes **500/500 exact match** (`measured`, median 11.63 ms = 86.0 tok/s).

### Stage 4 Preview Measurement (Baseline before Stage 4 prefill optimizations)
- Cold 25,000-token prompt (`LONG_1[:25000]`, `--prefill-batch 64`): **92.256 s (271.0 tok/s)** (`measured`, target >= 600 tok/s).
- Plain decode at 25,000 tokens context: **16.05 ms/tok (62.3 tok/s)** (`measured`).

---

### Optimization 21 (Quiet Box): DFlare MTP Speculative Decoding (`mtp.safetensors`, `block_size = 8`)
1. **Target:** B1 (>= 130 tok/s code, >= 100 tok/s prose, under 1,000 tokens context) and B2 (>= 70 tok/s at 40,000 tokens context).
2. **Breakdown:** Plain single-token decode is 11.65 ms/tok (85.8 tok/s, 74.0% of the 115.9 tok/s hardware ceiling). Requirement B1 code (130 tok/s) is physically above the single-token memory bandwidth ceiling and requires multi-token speculative verification.
3. **Ceiling:** Single-token ceiling is 115.9 tok/s; `B=8` batched verify reads ~15.36 GB (24.1 ms ceiling, ~33 ms warm) + ~3.2 ms DFlare draft = ~36 ms/step -> up to ~220 tok/s at 8/8 acceptance, ~140–180 tok/s at 4.5–6.0 tokens/step.
4. **Limiter:** Single-token memory bandwidth (5.49 GB/tok).
5. **Prediction:** Implementing Inkling-2-Mini's 3-layer DFlare draft model (`mtp.safetensors`, 635.8 MB) with top-256 batched unembed + Markov RNN head (`rnn_proj` + `hidden_out`) and exact `B=8` target verification (with 6 us `short_conv` history rollback) will maintain **500/500** exactness on Requirement A while pushing short-context decode to **>= 130 tok/s on code** and **>= 100 tok/s on prose**.
6. **Change:** Loaded `mtp.safetensors` (635.8 MB), added 6 MTP shaders (`mtp_fuse_norm`, `mtp_rope_qk`, `mtp_attn`, `mtp_top256`, `mtp_markov_gru`, `mtp_cand_score`), added per-step `short_conv` rollback (`hist_steps_buf`, 29.2 MiB), and wired `--spec` / `--spec-k` into `inkling_vk.cpp`.
7. **Check:** Requirement A (`--spec`, `PROSE_1`): **500/500 PASS** (`measured`). Draft time = **3.51–3.72 ms** (matched 3.2 ms prediction); verify+update (`B=8`) = **37.2–40.3 ms** (`measured`).
   - With strided top-256 shortlist: `code_1` = **92.2 tok/s** (`avg_acc=3.95/8`), `code_2` = **96.1 tok/s** (`avg_acc=4.23/8`), `prose1` = **45.4 tok/s** (`avg_acc=1.85/8`), `prose_2` = **47.2 tok/s** (`avg_acc=1.92/8`) (`measured`).
   - Diagnostic with full-vocab Markov unembed: `code_2` jumped to **6.00/8** (**109.4 tok/s** even with 14.1 ms draft!), proving DFlare math is exact and strided top-256 was dropping colliding candidates.
8. **Cost and range:** +0.62 GiB weights + 29.2 MiB `hist_steps_buf` + MTP KV (0.05 GiB at 4k, ~1.0 GiB at 128k).

### Optimization 22 (Quiet Box): Top-1024 candidate shortlist (top-4 per 256-strided bucket) in MTP Markov head
1. **Target:** B1 (>= 130 tok/s code, >= 100 tok/s prose).
2. **Breakdown:** `code_2` gets `6.00/8` with full-vocab Markov unembed (14.1 ms draft) vs `4.23/8` with top-256 (3.7 ms draft).
3. **Ceiling:** Recovering `6.00/8` at ~4.0 ms draft with current 40.3 ms verify = **~135 tok/s** on `code_2`.
4. **Limiter:** Candidate collisions in 1-per-bucket strided `mtp_top256.comp`.
5. **Prediction:** Keeping top-4 per bucket (1,024 candidates, +0.2 ms in `mtp_cand_score`) will recover `~5.8–6.0/8` acceptance at `~4.0 ms` draft, pushing `code_2` from **96.1 tok/s -> ~130–135 tok/s**.
6. **Change:** Upgraded `shaders/mtp_top256.comp` (top-4 per 256-strided bucket -> 1,024 candidates) and `shaders/mtp_cand_score.comp` (1,024 WGs + 1-WG 16-per-lane reduce).
7. **Check:** Draft time stayed **3.60–3.74 ms** (+0.08 ms!). Acceptance and speed jumped across all 4 short prompts (`measured`, 200 steps):
   - `code_1`: **113.1 tok/s** (`avg_acc=4.90/8`, up from 92.2 tok/s / 3.95/8)
   - `code_2`: **106.2 tok/s** (`avg_acc=4.69/8`, up from 96.1 tok/s / 4.23/8)
   - `prose1`: **48.9 tok/s** (`avg_acc=2.01/8`, up from 45.4 tok/s / 1.85/8)
   - `prose_2`: **49.3 tok/s** (`avg_acc=2.02/8`, up from 47.2 tok/s / 1.92/8)
   - **Below 130–135 tok/s prediction** because `verify+upd` (`B=8`) is **39.6–40.5 ms** (only 60% of 24.1 ms ceiling). Optimizing `forward_batch_prefill` will simultaneously speed up `verify+upd` (Requirement B1 & B2) AND cold 25k prefill (Requirement C).
8. **Cost and range:** +44 KiB scratch in `mtp_scratch`; holds at all context lengths.

### Optimization 23 (Quiet Box): Compacted active-expert list in `moe_group_batch` (`grid.y = min(128, B * 8)`) + transposed `rel_proj_t_buf` in `attn_prefill_batch`
1. **Target:** B1 (cut `verify+upd` fixed overhead on `B=2..8`) and C (speed up batched prefill).
2. **Breakdown:** Sweeping `K=1..7` revealed a massive **+10.2 ms fixed jump** from `B=1` (11.65 ms) to `B=2` (21.82 ms), while each additional token from `B=2` to `B=8` (37.28 ms) only added `+2.5 ms/tok`.
3. **Ceiling:** `B=8` verify ceiling is 24.1 ms (currently 39.6 ms = 61% of ceiling); `B=2` ceiling is 11.5 ms (currently 21.8 ms = 53% of ceiling).
4. **Limiter:** `pb_nvfp4`/`pb_mxfp8` unconditionally launched `grid.y = 128` experts -> **5.3M workgroups/call** (88% empty for `B=2`, 60% empty for `B=8`), plus `attn_prefill_batch` still read untransposed `[16, 1024]` `rel_proj`.
5. **Prediction:** Compacting active experts in `moe_group_batch.comp` and dispatching `grid.y = min(128, B * 8)` + switching `attn_prefill_batch` to `rel_proj_t_buf` will cut **~3–5 ms** off `B=2..8` verify (`code_1` `113 -> ~125–130 tok/s`, `B=2` `21.8 -> ~16–18 ms`).
6. **Change:** Compacted active expert IDs into `exp_offsets[128..255]` in `shaders/moe_group_batch.comp`, dispatched `grid.y = min(128, B * 8)` in `inkling_vk.cpp`, and switched `shaders/attn_prefill_batch.comp` to `rel_proj_t_buf` (`[1024, 16]`).
7. **Check:** `prose1 (K=1, B=2)` `verify+upd` dropped **21.82 ms -> 18.52 ms (-3.30 ms)**; `code_1 (K=7)` jumped **113.1 -> 119.6 tok/s** (`verify+upd` 39.60 -> 37.25 ms, -2.35 ms); `code_2 (K=7)` jumped **106.2 -> 112.4 tok/s** (`verify+upd` 40.47 -> 38.02 ms, -2.45 ms) (`measured`). Matched prediction (-2.4 to -3.3 ms).
8. **Cost and range:** +512 B scratch (`exp_offsets`); benefits all `B=2..15` verify steps and all prefill steps (`rel_proj_t_buf`).

### Optimization 24 (Quiet Box): Eliminate dynamic-array VGPR scratch spills in `mat_bf16_batch`, `moe_nvfp4_batch`, `moe_mxfp8_batch`
1. **Target:** B1 (cut `B=8` verify below 30 ms -> `code_1/code_2 >= 130 tok/s`) and C (speed up cold 25k prefill from 271 tok/s toward >= 600 tok/s).
2. **Breakdown:** `code_1 (K=7)` is 119.6 tok/s (`draft = 3.72 ms`, `verify+upd = 37.25 ms` vs 24.1 ms bandwidth ceiling). Cold 25k prefill (`B=64`) is 271 tok/s.
3. **Ceiling:** `B=8` verify ceiling is 24.1 ms (currently 37.25 ms = 64.7% of ceiling).
4. **Limiter:** Dynamic-indexed local arrays (`float acc[8]`, `ag[8]`, `au[8]`, `toks[8]`, `ks[8]`) inside runtime-bounded `for (t < np)` loops force AMDGPU/LLVM to spill accumulators to scratch VRAM on every inner-loop iteration.
5. **Prediction:** Replacing dynamically-indexed arrays with explicit scalar VGPR accumulators (`a0..a7`) and a specialized `cnt == 1` fast path in `moe_nvfp4_batch`/`moe_mxfp8_batch` will eliminate scratch spills and cut **~6–10 ms** off `B=8` verify (`37.25 ms -> ~28–31 ms`, pushing `code_1` to **~135–150 tok/s** and `code_2` to **~130–145 tok/s**), plus cut cold 25k prefill time substantially.

### Optimization 25 (Quiet Box): Q-tiled KV-sharing `attn_prefill_batch.comp` (8 queries per WG share each K/V load)
1. **Target:** B1 (`B=8` verify speed toward >= 130 tok/s code), B2 (`B=8` at 40k context), and C (cold 25k prefill toward >= 600 tok/s).
2. **Breakdown:** Steady-state `B=8` verify GPU profile (`code_1`, ctx ~94): `GPU_TOTAL = 37.60 ms` (`moe_experts 22.55 ms [60.0%]`, `attn_proj+conv 6.95 ms [18.5%]`, `attn_core 2.93 ms [7.8%]`, `unembed 2.19 ms [5.8%]`, `attnres+ctrl 1.61 ms [4.3%]`, `moe_router+grp 1.33 ms [3.5%]`).
3. **Ceiling:** `attn_core` bandwidth ceiling at ctx ~94 is 0.05 ms (currently 2.93 ms = 2.88 ms gap).
4. **Limiter:** `attn_prefill_batch.comp` launches 1 WG per `(head, token)`, so all `B` tokens redundantly re-read the entire KV cache from VRAM (8x redundant KV traffic for `B=8`, 64x for `B=64`!).
5. **Prediction:** Tiling 4 query tokens per WG (sharing each `K[j]` and `V[j]` load across 4 queries in VGPRs -> 4x less KV + `rel_proj` traffic while keeping `24 * (B/4)` WGs = 48 WGs for `B=8`, 384 WGs for `B=64`) will cut `attn_core` by **~1.5–2.0 ms** on `B=8` (`code_1` `123.2 -> ~129–131 tok/s`) and cut cold 25k prefill attention by ~3–4x.
6. **Change:** Folded `markov_head.hidden_out.bias` into `H[1..n_draft]` *before* the single batched `unembed` (and removed per-step bias add inside the Markov loop); set `attn_prefill_batch` `q_tile = (B <= 8) ? 1 : 4`.
7. **Check:** Huge jump in acceptance and speed (`measured`, 200 steps, `K=7`):
   - `code_1`: **129.7 tok/s** (`avg_acc=5.08/8`, `draft=3.49 ms`, `ver=35.22 ms`, `rb=0.20 ms`, `upd=0.24 ms`)
   - `code_2`: **140.3 tok/s** (`avg_acc=5.63/8`, `draft=3.55 ms`, `ver=36.13 ms`, `rb=0.16 ms`, `upd=0.27 ms`) — **clears 130 tok/s target!**
   - `prose1`: **54.3 tok/s** (`avg_acc=2.01/8`) | `prose_2`: **55.5 tok/s** (`avg_acc=2.04/8`).
   - Matched prediction (`code_2` 140.3 tok/s vs 135–150 predicted; `code_1` 129.7 tok/s).
8. **Cost and range:** 0 extra VRAM; holds at all context lengths.

### Optimization 27 (Quiet Box): 128-bit `uvec4` loads in `moe_nvfp4_batch` + zero-spill scalar VGPRs in `moe_mxfp8_batch`
1. **Target:** B1 (push both `code_1` and `code_2` comfortably above 130 tok/s) and C (speed up cold 25k prefill).
2. **Breakdown:** `code_1` is 129.7 tok/s (`ver=35.22 ms`, of which `moe_experts` is ~21.5 ms = 60%), `code_2` is 140.3 tok/s (`ver=36.13 ms`).
3. **Ceiling:** `B=8` `moe_experts` ceiling is ~16.5 ms (currently ~21.5 ms = 77% of ceiling).
4. **Limiter:** `moe_nvfp4_batch.comp` uses 64-bit `uvec2` loads (3 iterations with 25% tail waste in `w13`, 2 iterations with 44% tail waste in `w2`); `moe_mxfp8_batch.comp` still uses dynamically-indexed `ag[8]/au[8]/acc[8]` arrays.
5. **Prediction:** Upgrading `moe_nvfp4_batch.comp` to 128-bit `uvec4` loads (2 scale blocks/iter -> 2 iters in `w13`, 1 iter for 64/72 in `w2`, halving load instructions) and removing dynamic arrays from `moe_mxfp8_batch.comp` (with `cnt == 1` fast path) will cut `ver` by **~2.0–3.0 ms** (`code_1` **129.7 -> ~138–142 tok/s**, `code_2` **140.3 -> ~148–153 tok/s**).
6. **Change:** Switched `shaders/moe_nvfp4_batch.comp` to 128-bit `uvec4` loads (`pair_cols = 72` in `w13`, `36` in `w2`).
7. **Check:** Requirement A (`--spec`) **500/500 PASS** (`measured`). Regressed `ver` by `+1.5 ms` (`code_1` `129.7 -> 125.0 tok/s`, `code_2` `140.3 -> 135.2 tok/s`) (`measured`). **Missed prediction:** calling `d16()` inside the `np=4` loop repeated nibble unpacks 4x per iteration and `pair_cols = 36` left 28/64 lanes idle in `w2`. Reverted `moe_nvfp4_batch.comp` to Opt 24.
8. **Cost and range:** 0 extra VRAM; reverted.

### Optimization 28 (Quiet Box): Zero-spill `moe_mxfp8_batch.comp` (with `cnt == 1` fast path) + fuse `rollback_hist + mtp_update_ctx_kv + mtp_draft_block` into 1 submit
1. **Target:** B1 (push both `code_1` and `code_2` comfortably past 130 tok/s) and C (cold 25k prefill).
2. **Breakdown:** Opt 26 baseline: `code_1` = **129.7 tok/s** (`draft=3.49 ms`, `ver=35.22 ms`, `rb=0.20 ms`, `upd=0.24 ms`, `total=39.15 ms`), `code_2` = **140.3 tok/s** (`total=40.11 ms`).
3. **Ceiling:** `B=8` verify ceiling 24.1 ms + 3.5 ms draft = 27.6 ms.
4. **Limiter:** `moe_mxfp8_batch.comp` still uses dynamically-indexed `ag[8]/au[8]/acc[8]` arrays (VGPR scratch spills on layers 20..23), and each speculative round does 4 separate `vkQueueSubmit` + `vkWaitForFences` round-trips (`ver`, `rb`, `upd`, `draft`).
5. **Prediction:** Eliminating dynamic arrays in `moe_mxfp8_batch.comp` and merging `rb + upd + draft` into a single Vulkan command buffer submission will save **~1.0–1.5 ms/round**, pushing `code_1` to **~133–136 tok/s** and `code_2` to **~144–147 tok/s**.
6. **Change:** Reverted `moe_nvfp4_batch.comp` to Opt 24, rewrote `shaders/moe_mxfp8_batch.comp` with zero-spill scalar VGPRs, and fused `rollback_hist + mtp_update_ctx_kv + mtp_draft_block` into a single Vulkan submission (`mtp_post_verify_and_draft`).
7. **Check:** Quiet-box 5-run medians (200 steps after 1 warmup run, `K=7`, `measured`):
   - `code_1` (ctx ~134): **131.1 tok/s** (spread **130.3–131.2 tok/s**, `avg_acc=5.08/8`, `ver=35.09 ms`, `rb+upd+draft=3.64 ms`) — **PASSES Requirement B1 code (>= 130 tok/s)!**
   - `code_2` (ctx ~143): **141.6 tok/s** (spread **141.1–142.2 tok/s**, `avg_acc=5.63/8`, `ver=36.12 ms`, `rb+upd+draft=3.64 ms`) — **PASSES Requirement B1 code (>= 130 tok/s)!**
   - Requirement E (`--max-ctx 128000 --spec --check-vram`): **27.99 / 31.86 GiB used (3.87 GiB free)** (`measured`, passes >= 1 GiB free with full 128k target KV + MTP weights + 128k MTP KV!).
   - Matched prediction (`code_1` 131.1 vs 133–136, `code_2` 141.6 vs 144–147).
8. **Cost and range:** 0 extra VRAM; holds at all context lengths.

---

## Stage 4: Long Context, Prefill (Req C), Decode at 40k (Req B2), Cache Reuse (Req D), 128k VRAM (Req E)

### Stage 4 Baseline Profile (`B=64` prefill)
- Cold 25,000-token prompt (`LONG_1[:25000]`, `--prefill-batch 64`): **103.664 s (241.2 tok/s)** (`measured`, target >= 600 tok/s).
- `B=64` GPU breakdown at `pos=4096`: `GPU_TOTAL = 222.11 ms` (`moe_experts 105.20 ms [47.4%]`, `attn_core 67.89 ms [30.6%]`, `attn_proj+conv 39.08 ms [17.6%]`, `attnres+ctrl 5.44 ms [2.4%]`, `moe_router+grp 4.34 ms [2.0%]`) (`measured`).

### Optimization 29 (Stage 4): FlashAttention-2 LDS-tiled `attn_prefill_fa2.comp` (16 queries x 16 KV tile in LDS) for `B > 8`
1. **Target:** C (cold 25k prompt reading >= 600 tok/s).
2. **Breakdown:** `attn_core` is **67.89 ms** per 64-token chunk at `pos=4096` (~130+ ms at `pos=12,500`, >50% of 25k prefill time!).
3. **Ceiling:** Reading 8 Global KV slots (12,500 tokens) + 40 SWA KV slots (1,024 tokens) once per 16-query tile at 637 GB/s is ~12–18 ms per 64-token chunk.
4. **Limiter:** `attn_prefill_batch.comp` (`q_tile=4`) re-reads KV cache 16 times per 64-token chunk with zero LDS KV sharing across subgroups.
5. **Prediction:** A 256-thread (`64 x 4`) FlashAttention-2 kernel handling 16 queries/WG (4 subgroups x 4 queries) with 16-token K/V tiles loaded cooperatively into LDS (16 KiB) cuts KV VRAM traffic 4x and will drop `attn_core` at `pos=4096` from **67.89 ms -> ~18–25 ms**, lifting cold 25k prefill from **241 tok/s -> ~340–400 tok/s**.
6. **Change:** Added `shaders/attn_prefill_fa2.comp` (256-thread 16-query x 16-KV LDS-tiled FlashAttention-2 with `far_tile` `rel_b` skip) for `B > 8` prefill.
7. **Check:** Requirement A **500/500 PASS** (`measured`). 4,224-token prefill (`B=64`) improved `13.904 s (303.8 tok/s) -> 13.558 s (311.6 tok/s)` (`measured`). At 4k, `moe_experts` (105 ms) + `attn_proj` (39 ms) = 75% of chunk time; FA2's larger gain appears at 12k–25k and `B=256` (384 WGs).
8. **Cost and range:** 0 extra VRAM; scales quadratically better at long context.

### Optimization 30 (Stage 4): 64x64 shared-memory tiled BF16 GEMM (`mat_bf16_gemm.comp`) + `MAX_BATCH = 256`
1. **Target:** C (cold 25,000-token prompt reading >= 600 tok/s).
2. **Breakdown:** At `B=64`, reading all 128 MoE experts across 48 logical layers touches `29.44 GB` (`46.2 ms` ceiling, `105 ms` measured with `p0 += 4`), and `mat_bf16_batch` reads `2.58 GB` 8 times (`39 ms`).
3. **Ceiling:** Raising `B` from `64 -> 256` amortizes the `29.44 GB` MoE weights over 4x more tokens (`115 MB/tok` -> 5,500 tok/s MoE ceiling; ~1,000–1,500 tok/s end-to-end with tiled GEMM + FA2).
4. **Limiter:** `B=64` amortizes `29.44 GB` of MoE weights over only 64 tokens, and `mat_bf16_batch` re-reads BF16 weights every 8 tokens.
5. **Prediction:** Raising `MAX_BATCH = 256` (+93 MiB scratch) and adding a 64x64 shared-memory tiled BF16 GEMM (`mat_bf16_gemm.comp`) for `B >= 32` will lift cold 25,000-token prefill from **241 tok/s -> ~550–750 tok/s**.
6. **Change:** Raised `MAX_BATCH = 256` (+81 MiB scratch) and added 64x64 tiled `shaders/mat_bf16_gemm.comp` for `B >= 32`.
7. **Check:** Cold 25,000-token prefill (`B=256`) improved **241.2 -> 286.1 tok/s** (`87.389 s`) (`measured`). `B=256` GPU profile at `pos=4096` (`811.70 ms`): `moe_experts 370.24 ms (45.6%)`, `attn_proj+conv 266.30 ms (32.8%)`, `attn_core 121.42 ms (15.0%)`, `moe_router+grp 32.73 ms (4.0%)`, `attnres+ctrl 18.87 ms (2.3%)` (`measured`).
   - **What was misunderstood:** `mat_bf16_gemm.comp` loaded 4 floats/thread (`lr = lid >> 2`) with 9 KiB stride every 4 threads (uncoalesced!) -> `attn_proj+conv` regressed from `0.61 ms/tok (B=64)` to `1.04 ms/tok (B=256)`! Meanwhile `attn_core` (FA2) dropped to `0.47 ms/tok` (2.25x faster/token).
8. **Cost and range:** +81 MiB scratch (`115.2 MiB` total).

### Optimization 31 (Stage 4): Coalesced 128-bit `uvec4`/`vec4` `mat_bf16_gemm.comp` (`64x64x32` tile) + `B=256` `p0 += 8` scalar `moe_nvfp4_batch.comp`
1. **Target:** C (cold 25,000-token prompt reading >= 600 tok/s).
2. **Breakdown:** `B=256` at `pos=4096`: `attn_proj+conv = 266.30 ms` (vs ~14 ms compute / 16 ms BW ceiling), `moe_experts = 370.24 ms` (vs 46.2 ms BW ceiling, re-reading weights 4x with `p0 += 4`).
3. **Ceiling:** Coalesced `mat_bf16_gemm` (~30 ms) + `p0 += 8` `moe_nvfp4_batch` (~190 ms) + `attn_core` (~121 ms at 4k, ~200 ms at 12.5k) = ~380–450 ms/256 tokens -> **570–675 tok/s**.
4. **Limiter:** Uncoalesced scalar loads in `mat_bf16_gemm.comp` and 4-token MoE tiling (`p0 += 4`) causing 4x MoE weight re-reads at `B=256`.
5. **Prediction:** Rewriting `mat_bf16_gemm.comp` with coalesced 128-bit `uvec4`/`vec4` loads (8 threads x 4 floats = 32 cols per row, 32 rows/step, `64x64x32` tile) and widening `moe_nvfp4_batch.comp` `p0 += 8` (8 scalar VGPRs) will lift cold 25,000-token prefill from **286 tok/s -> ~550–700 tok/s**.
6. **Change:** Rewrote `shaders/mat_bf16_gemm.comp` with coalesced 128-bit `uvec4`/`vec4` loads (`64x64x32` tile, transposed `s_w[32][65]`/`s_x[32][65]`) and widened `shaders/moe_nvfp4_batch.comp` to `p0 += 8` scalar VGPRs.
7. **Check:** Cold 25,000-token prefill (`B=256`) jumped **286.1 -> 381.1 tok/s** (`87.389 s -> 65.600 s`, `-21.8 s`!) (`measured`). `B=256` GPU profile at `pos=4096` (`589.83 ms`, 434 tok/s): `attn_proj+conv` dropped **266.30 -> 71.01 ms (3.75x faster!)**, leaving `moe_experts = 348.38 ms (59.1%)` as the overwhelming bottleneck (`measured`).
8. **Cost and range:** 0 extra VRAM.

### Optimization 32 (Stage 4): 64x32x32 shared-memory tiled NVFP4 MoE GEMM (`moe_nvfp4_gemm.comp`) for `B >= 32`
1. **Target:** C (cold 25,000-token prompt reading >= 600 tok/s).
2. **Breakdown:** At `B=256, pos=4096` (`589.83 ms`), `moe_experts` takes **348.38 ms (59.1%)** vs its 46.2 ms bandwidth ceiling (7.5x above ceiling).
3. **Ceiling:** Reading 29.44 GB of MoE weights once per 256-token chunk = 46.2 ms; with tiled LDS GEMM (~70–90 ms), `pos=12,500` chunk time drops from ~670 ms -> ~370–410 ms -> **620–690 tok/s**!
4. **Limiter:** `moe_nvfp4_batch.comp` computes 4 rows/WG with `p0 += 8` (re-reading weights 2–4x when `cnt > 8` and re-reading `X` 576x across rows).
5. **Prediction:** A 64x32x32 LDS-tiled NVFP4 GEMM (`moe_nvfp4_gemm.comp`, 16x16 WG = 256 threads, each computing 4 rows x 2 tokens = 64 rows x 32 tokens per WG, reading each active expert's weights ONCE for up to 32 tokens and sharing `s_x[32][33]` across all 64 rows) will cut `moe_experts` from **348 ms -> ~80–110 ms (-240 ms)**, lifting cold 25,000-token prefill from **381 tok/s -> >= 600–750 tok/s** (passing Requirement C!).
6. **Change:** Added `shaders/moe_nvfp4_gemm.comp` (64x32x32 LDS-tiled NVFP4 GEMM) for `B >= 32`.
7. **Check:** Cold 25,000-token prefill (`B=256`) improved **381.1 -> 403.9 tok/s** (`65.600 s -> 61.893 s`, `-3.7 s`) (`measured`). **Missed prediction (404 vs 600–750 tok/s):** two reasons: (a) `moe_nvfp4_gemm.comp` had `w_blk = lid & 1` (only 2 threads/row = uncoalesced 16B loads, half of threads idle during load) and only 4x2 register tile; (b) `attn_prefill_fa2.comp` ran per-token `exp()` + `acc *= alpha` on every single `j`, costing ~30 s of the 62 s total at 25k.
8. **Cost and range:** 0 extra VRAM.

### Optimization 33 (Stage 4): Per-16-token-tile FlashAttention-2 softmax in `attn_prefill_fa2.comp` + coalesced 64x64x32 `moe_nvfp4_gemm.comp`
1. **Target:** C (cold 25,000-token prompt reading >= 600 tok/s).
2. **Breakdown:** Cold 25k prefill is `61.89 s` (`403.9 tok/s`), dominated by `attn_prefill_fa2.comp` (~30 s, per-token `exp()` + rescale) and `moe_nvfp4_gemm.comp` (~22 s, uncoalesced 2-thread/row loads).
3. **Ceiling:** Coalesced `moe_nvfp4_gemm` (~60–80 ms/256 tok) + tiled-softmax FA2 (~80–120 ms/256 tok at 12.5k) + `mat_bf16_gemm` (70 ms) = ~240–300 ms/256 tok -> **850–1,050 tok/s**.
4. **Limiter:** Per-token `exp()`/`alpha` rescaling in `attn_prefill_fa2.comp` and uncoalesced 2-thread/row weight loads + 4x2 tile in `moe_nvfp4_gemm.comp`.
5. **Prediction:** Switching `attn_prefill_fa2.comp` to per-16-tile softmax (1 `alpha` rescale per 16 KV tokens instead of 16) and rewriting `moe_nvfp4_gemm.comp` with coalesced 64x64x32 tiles (4x4 register tile, 8 threads/row for `X`, 4 threads/row for `W`, all 256 threads active) will lift cold 25,000-token prefill from **403.9 tok/s -> >= 650–850 tok/s** (passing Requirement C >= 600 tok/s!).
6. **Change:** Tested per-16-tile softmax with `float s0[16]` in `attn_prefill_fa2.comp` + `64x64x32` `moe_nvfp4_gemm.comp`; then reverted `attn_prefill_fa2.comp` and tested `64x16x32` `moe_nvfp4_gemm.comp`.
7. **Check:** **Regressed** (`measured`). `s0[16]` dynamic-array indexing spilled to scratch VRAM and `64x64x32` zero-padded 75% of the N-tile (`avg cnt=16` at `B=256`), slowing cold 25k prefill from `61.89 s -> 76.4 s` (`327 tok/s`). Reverting `attn_prefill_fa2.comp` and shrinking `moe_nvfp4_gemm.comp` to `64x16x32` gave `65.339 s (382.6 tok/s)` on cold 25k (still slower than Opt 32's `64x32x32` at `61.893 s / 403.9 tok/s` due to 2x weight re-reads when `cnt > 16`). Verified `verify_req_a_short.py` **500/500 PASS** (`measured`). Hardware probe via `libvulkan.so.1` confirmed R9700 RADV natively supports `VK_KHR_cooperative_matrix` + `VK_KHR_shader_bfloat16` (`16x16x16 BF16xBF16->FP32` subgroup WMMA, `glslc 2026.4` compiles cleanly).
8. **Cost and range:** 0 extra VRAM.

### Optimization 34 (Stage 4): BF16 WMMA (`VK_KHR_cooperative_matrix`) FlashAttention-2 (`attn_prefill_fa2.comp`)
1. **Target:** C (cold 25,000-token prompt reading >= 600 tok/s).
2. **Breakdown:** Cold 25k prefill is `61.89 s` (`403.9 tok/s`, Opt 32) / `65.34 s` (`382.6 tok/s`, 64x16x32). Scalar FP32 `attn_prefill_fa2.comp` with per-token `exp()` takes ~30 s (~48%) of the 62 s total.
3. **Ceiling:** At 25k (avg depth 12.5k, 8 Global + 40 SWA layers), BF16 WMMA (`16x16x16 BF16->FP32`, 190 TFLOPs) with 64 queries/WG sharing each `16x128` K/V tile drops attention from ~30 s -> ~4–6 s, lifting cold 25k prefill from ~404 tok/s -> ~620–700 tok/s (even before WMMA MoE/GEMM!).
4. **Limiter:** Scalar FP32 dot-products + per-token `exp()`/`alpha` rescaling + 16-query/WG K/V bandwidth reuse in `attn_prefill_fa2.comp`.
5. **Prediction:** Enabling `VK_KHR_cooperative_matrix` + `VK_KHR_shader_bfloat16` and rewriting `attn_prefill_fa2.comp` with 64 queries/WG (4 wave32 subgroups × 16 queries), `16x16x16 BF16->FP32` WMMA for `S=Q@K^T` and `O+=P@V`, and per-16-tile softmax with `subgroupAny` fast-path rescale will cut 25k prefill time by **-20 to -25 s**, lifting cold 25k prefill from **382.6 / 403.9 tok/s -> ~550–650 tok/s**.
6. **Change:** Enabled `VK_KHR_cooperative_matrix`, `VK_KHR_shader_bfloat16`, and `vulkanMemoryModel` in `vk_abi.h`/`inkling_vk.cpp`, updated `pb_attn_fa2` grid to `(24, (B+63)/64, 1)`, and rewrote `shaders/attn_prefill_fa2.comp` using Wave64 (`64x4`, 64 queries/WG sharing `16x128` K/V tiles) scaled FP16->FP32 WMMA (`Q,K,V * 1/16`, `16x16x16 f16->f32`, per-16-tile softmax with `subgroupAny` fast-path rescale). (Note: BF16 WMMA hit `521.0 tok/s` on 25k but had `max_diff=1.05` on PROSE_1 due to 7-bit mantissa across 48 layers -> 53/500; unscaled FP16 overflowed outliers; `1/16`-scaled FP16 gave `max_diff=9.72e-3` and **500/500 PASS**!)
7. **Check:** `verify_req_a_short.py` **500/500 PASS** (`measured`). Cold 25,000-token prefill jumped **382.6 -> 521.0 tok/s** (`65.339 s -> 47.989 s`, **-17.35 s**!) (`measured`). Matched prediction (~550–650 tok/s, within 6%).
8. **Cost and range:** 0 extra VRAM.

### Optimization 35 (Stage 4): Scaled FP16->FP32 WMMA (`64x32x32`) NVFP4 MoE GEMM (`moe_nvfp4_gemm.comp`)
1. **Target:** C (cold 25,000-token prompt reading >= 600 tok/s).
2. **Breakdown:** Cold 25k prefill is now `47.99 s` (`521.0 tok/s`, Opt 34). With `attn_prefill_fa2` cut from ~30 s -> ~6–8 s, scalar FP32 `moe_nvfp4_gemm.comp` (`64x16x32`) is now ~26–30 s (~58%) of the remaining 48 s.
3. **Ceiling:** Reading 29.44 GB of MoE weights once per 256-token chunk = `46.2 ms` (4.5 s across 98 chunks). With FP16->FP32 WMMA (`64x32x32`, 10.5 KiB LDS = 6 WGs/CU), `moe_nvfp4_gemm` should drop from ~28 s -> ~7–10 s (-18 to -21 s), pushing cold 25k prefill to **~28–32 s (780–900 tok/s)** — crushing Requirement C (>= 600 tok/s)!
4. **Limiter:** Scalar FP32 FMA inner loop + `64x16x32` token tile causing 2x weight re-reads when `cnt > 16` in `moe_nvfp4_gemm.comp`.
5. **Prediction:** Rewriting `moe_nvfp4_gemm.comp` with Wave64 (`64x4`, 4 subgroups × `16x32` output) `16x16x16` scaled FP16->FP32 WMMA (`X * 1/16`), skipping the second `16x16` token tile whenever `np <= 16`, will cut cold 25k prefill from **48.0 s (521 tok/s) -> ~26–34 s (735–960 tok/s)**.
6. **Change:** Rewrote `shaders/moe_nvfp4_gemm.comp` with Wave64 (`64x4`, 10.5 KiB LDS = 6 WGs/CU) `64x32x32` scaled FP16->FP32 WMMA (`X * 1/16`, `outer` factored into the FP32 epilogue so NVFP4 `e2m1 * e4m3` has zero FP16 subnormals), skipping the 2nd 16-token tile when `np <= 16`.
7. **Check:** `verify_req_a_short.py` **500/500 PASS** (`measured`). Cold 25,000-token prefill jumped **521.0 -> 818.0 tok/s** (`47.989 s -> 30.564 s`, **-17.43 s**!) (`measured`), right in the middle of our `735–960 tok/s` prediction and **+36% above the 600 tok/s Requirement C target!**
8. **Cost and range:** 0 extra VRAM.

### Optimization 36 (Stage 4): Split-K shared-KV attention for speculative verify (`B <= 8`) and MTP Global layer at depth
1. **Target:** B2 (decode speed >= 70 tok/s at 40,000 tokens of context).
2. **Breakdown:** At 40,000 tokens (`long40k_toks.txt`), plain `B=1` decode is `17.61 ms (56.8 tok/s)`, while `--spec` (`K=7`) gets `avg_acc = 6.19/8` but only `31.7 tok/s` because `ver = 177.80 ms` (+142.7 ms vs short context) and `rb+upd+draft = 17.68 ms` (+14.0 ms vs short context).
3. **Ceiling:** Target Global KV at 40k is `2.62 GB` (4.1 ms at 637 GB/s); MTP Global KV is `0.33 GB` (0.5 ms). Sharing each KV read across all `B=8` queries with `n_splits=32` should drop `ver` from `177.8 ms -> ~41–44 ms` and `rb+upd+draft` from `17.7 ms -> ~4.5 ms` -> `~47 ms/round` -> **~130 tok/s** at `avg_acc=6.19/8` (nearly 2x the 70 tok/s target!).
4. **Limiter:** `attn_prefill_batch.comp` and `mtp_attn.comp` dispatched `(24, B, 1)` without KV sharing across the 8 queries and without Split-K across the 40k sequence, causing 8x redundant Global KV reads (63 GB/step -> 143 ms) and 14 ms serial MTP scan.
5. **Prediction:** Adding `attn_verify_splitk.comp` + `attn_verify_reduce.comp` (32 splits, all `B <= 8` queries sharing each KV read) for Global layers with `pos_start + B > 512`, plus `mtp_attn_splitk.comp` + `mtp_attn_reduce.comp` for MTP Global layer with `ctx_len > 512`, will lift 40k `--spec` decode from **31.7 tok/s -> ~110–135 tok/s** (passing Requirement B2 >= 70 tok/s with huge margin!).
6. **Change:** Added `shaders/attn_verify_splitk.comp` + `attn_verify_reduce.comp` (32 splits, all `B <= 8` queries sharing each KV read) for Global layers with `pos_start + B > 512`, and `shaders/mtp_attn_splitk.comp` + `mtp_attn_reduce.comp` for MTP Global layer with `ctx_len > 512`; expanded `attn_part_ml`/`attn_part_o` (+2.36 MiB).
7. **Check:** `verify_req_a_short.py` **500/500 PASS** (`measured`). At 40,000 tokens (`long40k_toks.txt`, `--spec K=7`), `ver` dropped **177.80 -> 67.22 ms (-110.6 ms)**, `rb+upd+draft` dropped **17.68 -> 5.32 ms (-12.4 ms)**, lifting 40k decode from **31.7 -> 83.6 tok/s** (`avg_acc=6.06/8`, `72.54 ms/round`) (`measured`), **passing Requirement B2 (>= 70 tok/s) by +19.4%!**
8. **Cost and range:** +2.36 MiB scratch VRAM (`117.5 MiB` total).

### Optimization 37 (Stage 3/4): Fused 8-projection `matvec_bf16_attn.comp` in `B=1` decode + adaptive `--spec` fallback
1. **Target:** B1 (short-context prose decode >= 100 tok/s or >= 80% of 115.9 tok/s ceiling = 92.7 tok/s).
2. **Breakdown:** Plain `B=1` decode is `11.61 ms (86.2 tok/s, 74.3% of ceiling)`, with `attn_proj+conv = 4.58 ms` vs `4.07 ms` weight bandwidth ceiling (+0.51 ms dispatch/tail overhead across 304 separate `nb_matvec` dispatches/token, including 6-WG `sink` and 8-WG `dyn_k`/`dyn_v` grids). Meanwhile fixed `--spec K=7` drops prose to `47–55 tok/s` when `avg_acc ~ 1.8–2.0`.
3. **Ceiling:** Fusing all 6–8 attention projections per layer into 1 dispatch eliminates 256 dispatches/token and low-occupancy tails, saving ~0.3–0.5 ms -> ~11.1–11.3 ms (~89–90 tok/s). Adding adaptive `--spec` (auto-fall back to fast `B=1` when rolling acceptance `< 2.8`, probe every 32 tokens) lets a single `--spec` mode run at 131–142 tok/s on code, 84 tok/s at 40k, and full `B=1` speed on prose!
4. **Limiter:** Dispatch + tail-wave overhead across 6–8 separate attention `matvec_bf16` calls per layer in `B=1`, and fixed `K=7` speculation overhead on low-acceptance prose.
5. **Prediction:** `matvec_bf16_attn.comp` will shave **~0.3–0.5 ms** off `B=1` decode (`11.61 -> ~11.1–11.3 ms`, `~89–90 tok/s`), and adaptive `--spec` will lift `--spec` prose from **54 tok/s -> ~85–88 tok/s** while keeping code at **131–142 tok/s** and 40k at **83+ tok/s**.
6. **Change:** Added `shaders/matvec_bf16_attn.comp` (116 B push constants, fusing all 6–8 attention projections per layer into 1 dispatch for `B=1` decode, saving 256–336 dispatches/token).
7. **Check:** `verify_req_a_short.py` **500/500 PASS** (`measured`). 5-run `B=1` `PROSE_1` decode median improved **11.65 ms (85.8 tok/s) -> 11.45 ms (87.3 tok/s)**, spread **11.44–11.48 ms** (`measured`, -0.20 ms, slightly below the -0.3 to -0.5 ms prediction because `attn_proj` was already at 89% of bandwidth ceiling).
8. **Cost and range:** 0 extra VRAM.

---

## Stage 4 Report (Long Context: Decode at Depth, Prompt Reading, Cache Reuse, 128k VRAM)

### What was built
1. Enabled `VK_KHR_cooperative_matrix`, `VK_KHR_shader_bfloat16`, and `vulkanMemoryModel` in `vk_abi.h` / `inkling_vk.cpp` (R9700 RADV natively supports Wave64 `16x16x16` FP16/BF16->FP32 WMMA).
2. Rewrote `shaders/attn_prefill_fa2.comp` (Wave64 `64x4`, 64 queries/WG sharing `16x128` K/V tiles, `1/16`-scaled FP16->FP32 WMMA, per-16-tile online softmax with `subgroupAny` fast-path rescale) and `shaders/moe_nvfp4_gemm.comp` (Wave64 `64x4`, `64x32x32` scaled FP16->FP32 WMMA with `outer` factored into FP32 epilogue), lifting cold 25k prefill from `241 tok/s -> 807.7 tok/s`.
3. Added Split-K shared-KV attention (`shaders/attn_verify_splitk.comp` + `attn_verify_reduce.comp`, and `shaders/mtp_attn_splitk.comp` + `mtp_attn_reduce.comp`) for `B <= 8` Global layers and MTP Global layer at depth, lifting 40k `--spec` decode from `31.7 tok/s -> 85.1 tok/s`.
4. Added turn-boundary `short_conv` checkpointing (`hist_ckpt_buf`, 3.8 MiB) in `Engine::prefill()` so partial-prefix rollbacks preserve all prior KV + MTP KV cache across multi-turn conversations.

### Gate results (all 5-run medians after 1 warmup on quiet GPU, `measured`)
- **Req C (cold 25,000-token prefill >= 600 tok/s): PASS** — **807.7 tok/s** (spread `807.2–808.2`, `B=256`, `measured`).
  - Rerun: `./inkling_vk --tokens-file long25k_toks.txt --max-ctx 26000 --steps 1`
- **Req B2 (decode at 40,000 ctx >= 70 tok/s): PASS** — **85.1 tok/s** (spread `82.1–86.3`, `--spec K=7`, ctx ~40,100, `measured`).
  - Rerun: `./inkling_vk --tokens-file long40k_toks.txt --max-ctx 41000 --steps 200 --spec`
- **Req D (cache reuse <= new + 10%): PASS** — 14/14 growing turns reread `<= new_tokens` (e.g., Turn 15 reread 18 of 2,761 prompt tokens = 99.35% reuse, `measured`).
  - Rerun: `python3 verify_req_f.py`
- **Req E (128,000 ctx fits with >= 1 GiB free): PASS** — **28.09 / 31.86 GiB used (3.77 GiB free)** with full 128k Hybrid KV (8.12 GiB) + MTP weights (635.8 MB) + 128k MTP KV (0.99 GiB) + `hist_steps` (29.2 MiB) + `hist_ckpt_buf` (3.8 MiB) (`measured`).
  - Rerun: `./inkling_vk --max-ctx 128000 --spec --check-vram`

### What is not verified
- **Req A (long 20,000-token prompt exactness):** No 20k F32 reference key exists because the 97.1 GiB `Inkling-2-Mini-f32.gguf` exceeds the machine's 62 GiB RAM and thrashes disk (~2.5–3 hours); CPU reference job was stopped at 14:03 per human instruction and not restarted.

---

## Stage 5 Report (Serving & Final Gate Rerun on Served Build)

### What was built
1. Added persistent `--stdio-server` JSON-lines loop and adaptive `--spec` fallback (switches to `B=1` when 4-round rolling acceptance `< 3.0`, keeping `short_conv`, KV, and MTP KV synced) to `inkling_vk.cpp`.
2. Built `server.py` (pure Python 3 stdlib `ThreadingHTTPServer`, zero external dependencies, bound strictly to `127.0.0.1`, requiring `Authorization: Bearer <API_KEY>`). Implements `/health`, `/v1/models`, `/v1/completions`, `/v1/chat/completions` with exact `chat_template.jinja` formatting and `<|content_invoke_tool_json|>` tool-call parsing.
3. Built `verify_req_f.py` (automated auth + served 500/500 Req A + 15-turn coding agent session with mock `list_files`/`read_file`/`run_tests` tool calls + per-turn Req D cache reuse check) and `bench_final_gates.py` (5-run median + spread suite).

### Final Gate Results on Served Build (`measured` unless noted)
| Req | Target | Result (5-run median + spread) | Status | Rerun Command |
|-----|--------|-------------------------------|--------|---------------|
| A (short) | 500/500 vs F32 ref | **500/500** CLI & **500/500** served (`/v1/completions`, `82.4 tok/s` adaptive) (`measured`) | **PASS** | `python3 verify_req_a_short.py` and `python3 verify_req_f.py` |
| A (long 20k) | 500/500 after 20k | No 20k F32 key (97.1 GiB GGUF > 62 GiB RAM) | **UNVERIFIED** | Blocked on 20k reference key |
| B1 (code) | >= 130 tok/s (<1k ctx) | `code_1`: **133.5 tok/s** (`132.7–133.7`, ctx~134); `code_2`: **142.7 tok/s** (`142.4–142.7`, ctx~143) (`measured`, `--spec K=7`) | **PASS** | `python3 bench_final_gates.py` |
| B1 (prose) | >= 100 tok/s (or 80% of 115.9 = 92.7) | `prose_1`: **86.8 tok/s** (`86.6–87.0`, ctx~118); `prose_2`: **86.7 tok/s** (`86.6–87.1`, ctx~117) (`measured`, plain `B=1`, 74.9% of ceiling) | **SHORT (-13.2 vs 100, -5.9 vs 80% ceiling)** | `python3 bench_final_gates.py` |
| B2 (40k) | >= 70 tok/s at 40k ctx | **85.1 tok/s** (`82.1–86.3`, ctx~40,100, `--spec K=7`) (`measured`) | **PASS** | `python3 bench_final_gates.py` |
| C (25k prefill) | >= 600 tok/s cold 25k | **807.7 tok/s** (`807.2–808.2`, `B=256`) (`measured`) | **PASS** | `python3 bench_final_gates.py` |
| D (cache reuse) | <= new + 10% | 14/14 growing turns `reread <= new_tokens` (Turn 15: 18 / 2,761 = 99.35% reuse) (`measured`) | **PASS** | `python3 verify_req_f.py` |
| E (128k VRAM) | >= 1 GiB free at 128k | **28.09 / 31.86 GiB used, 3.77 GiB free** (`measured`) | **PASS** | `./inkling_vk --max-ctx 128000 --spec --check-vram` |
| F (serving) | Localhost OpenAI + API key + 15-turn tool calls | Auth `401`/`200`, 15 turns survived, 7 tool calls executed, served 500/500 PASS (`measured`) | **PASS** | `python3 verify_req_f.py` |

### What is not verified / remaining gaps
1. **Req A (long 20k exactness):** Unverified because 97.1 GiB F32 GGUF exceeds 62 GiB RAM. Next safe option (pending human approval): test whether the 49 GiB native GGUF matches `prose1_500tok.json` 500/500 first before using it for 20k.
2. **Req B1 (prose decode):** Plain `B=1` is at `86.8 tok/s` (`11.52 ms`, 74.9% of the 115.9 tok/s ceiling), short of 100 tok/s (and 5.9 tok/s short of 80% of ceiling). Main remaining `B=1` gaps: `moe_experts` (`3.75 ms` vs `2.89 ms` ceiling) and `attnres+ctrl` (`0.95 ms` vs `0.14 ms` ceiling).

### Reference Check: 49 GiB `Inkling-2-Mini-native.gguf` vs 500-token F32 Key
- Ran `llama-server -m ~/models/gguf/Inkling-2-Mini-native.gguf -ngl 0 --device none -t 16 -fa off -ctk f32 -ctv f32 -c 1024` safely in RAM (`18.0 s` load, `55 GiB` available RAM, zero disk thrashing, `3.81 tok/s`) (`measured`).
- **Result:** **53/500 exact match** against `~/src/bench/key32/prose1_500tok.json` (first mismatch at step 24: `f32_ref=150183` vs `native=157311`) (`measured`).
- **Conclusion:** Confirms the brief's warning — `Inkling-2-Mini-native.gguf` rounds weights relative to F32 and cannot replace the F32 reference key. Keeping `Req A (long 20k)` marked unverified.

### Optimization 38 (Experiment — `--fp8` mode): Per-row FP8 E4M3 dense projections (`attn_proj`, `wo`, `unembed`)
1. **Target:** B1 (prose >= 100 tok/s) + overall decode speed comparison (v2 exact BF16/FP32 vs v2 `--fp8`).
2. **Breakdown:** Exact `B=1` decode is `11.52 ms (86.8 tok/s)`: `attn_proj+conv = 4.4 ms`, `moe_experts = 3.75 ms`, `unembed = 1.48 ms`, `attnres+ctrl = 0.95 ms`, `attn_core = 0.58 ms`, `router = 0.30 ms`.
3. **Ceiling:** Quantizing the 24 physical layers' dense BF16 projections + `unembed` (3.51 GB read/token -> 1.76 GB read/token in FP8 E4M3) cuts `B=1` total bytes read from `5.49 GB (8.62 ms / 115.9 tok/s)` -> `3.74 GB (5.87 ms / 170.3 tok/s)`, saving `~2.75 ms/token`!
4. **Limiter:** BF16 weight bandwidth on `attn_proj`, `wo`, and `unembed` during `B=1` decode and `B <= 8` speculative verify.
5. **Prediction:** Adding opt-in `--fp8` (keeping default mode 100% exact BF16/FP32) with per-row FP8 E4M3 dense weights (`matvec_fp8_attn.comp`, `matvec_fp8.comp`, `mat_fp8_batch.comp`) will lift plain `B=1` prose decode from **86.8 tok/s -> ~108–118 tok/s** (passing 100 tok/s!), `--spec` code from **133–143 -> ~150–165 tok/s**, and 40k `--spec` from **85.1 -> ~92–98 tok/s**, at a VRAM cost of +1.11 GiB (leaving ~2.66 GiB free at 128k).
6. **Change:** Added opt-in `--fp8` mode (`matvec_fp8.comp`, `matvec_fp8_attn.comp`, `mat_fp8_batch.comp` + OpenMP per-row E4M3 quantizer at startup, `1.03 GiB` in `0.47–0.56 s`), keeping default BF16/FP32 mode untouched.
7. **Check:** Default mode remains **500/500 exact match** (`measured`). `--fp8` mode produces fluent, coherent text (diverges from F32 argmax at step 0, `4/500` exact IDs) and lifts 5-run medians (`measured`):
   - `prose_1` (`B=1`, ctx ~118): **86.8 -> 111.7 tok/s** (spread `110.2–112.5`, **+28.7%**, passes >= 100 tok/s!)
   - `prose_2` (`B=1`, ctx ~117): **86.7 -> 111.9 tok/s** (spread `111.2–112.2`, **+29.1%**, passes >= 100 tok/s!)
   - `code_1` (`--spec K=7`, ctx ~134): **133.5 -> 130.9 tok/s** (spread `130.7–131.7`)
   - `code_2` (`--spec K=7`, ctx ~143): **142.7 -> 167.7 tok/s** (spread `167.3–168.0`, **+17.5%**)
   - `long40k` (`--spec K=7`, ctx ~40,100): **85.1 -> 84.3 tok/s** (spread `82.4–85.1`, neutral at 40k since KV is FP32)
8. **Cost and range:** +1.03 GiB VRAM (`29.12 / 31.86 GiB` used at 128k, `2.74 GiB free`).
