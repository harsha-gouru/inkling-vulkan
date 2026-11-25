# Inkling-2-Mini Model Map (Stage 0)

Every tensor, shape, storage format, and step-by-step operation for one token of `thinkingmachines/Inkling-2-Mini` (`Inkling2LoopForConditionalGeneration`), verified directly against `~/models/Inkling-2-Mini/` (`measured`) and `~/src/llama.cpp/src/models/inkling2*.cpp` + `conversion/inkling2.py`.

---

## 1. High-Level Architecture

Inkling-2-Mini is a **looped sparse Mixture-of-Experts (MoE) transformer** with **Attention Residuals (AttnRes)**, **short causal convolutions**, **hybrid sliding-window / global attention**, **relative-position bias (no RoPE)**, and **attention sinks**.

| Property | Value | Tag |
|---|---|---|
| Hidden size (`d_model`) | 2,304 | `measured` (`config.json`) |
| Physical layers | 24 (`0..23`) | `measured` |
| Loop structure | Prelude `0..5` (1x) -> Core `6..17` (3 passes) -> Epilogue `18..23` (1x) = **48 logical layers** | `measured` |
| Attention heads / KV heads / head dim | 24 Q heads / 8 KV heads (3:1 GQA) / `head_dim = 128` | `measured` |
| Local (SWA) vs Global layers | 20 SWA layers (window 1,024); 4 Global layers (`5, 11, 17, 23`) | `measured` |
| Logical cache slots | **48 slots**: 40 SWA slots + 8 Global slots (`5`, `11`x3, `17`x3, `23`) | `inferred` |
| Positional encoding | **No RoPE**. Learned relative-position bias (`d_rel = 16`, `rel_extent = 1024`) + log-scaling on Global layers | `measured` |
| Short causal convolutions | Kernel size `K = 4` (3 past steps cached) on K, V, post-Attn, post-MoE; Global K/V add dynamic per-head taps | `measured` |
| AttnRes block size | 6 layers per block (`0..5`, `6..11`, `12..17`, `18..23`), RMS-normalized values | `measured` |
| Loop cross-pass controllers | Layers `6..17`, low-rank `gate_rank = 64` gain/forget controllers on passes 1 and 2 | `measured` |
| MoE | 128 routed experts, top-8 per token, 0 shared experts, `intermediate_size = 1152`, SiTU activation | `measured` |
| Vocabulary | Padded `201,024`, unpadded active `200,058` (`eos = 200006`, `pad = 199999`) | `measured` |
| Max context length | 1,048,576 tokens (`model_max_length`) | `measured` |
| Weight files on disk | `model-00001-of-00001.safetensors` (19.526 GB, 1,244 tensors) + `mtp.safetensors` (0.636 GB, 54 tensors) | `measured` |
| LLM backbone only | 822 tensors, **18.889 GB** (Vision tower: 422 tensors, 0.637 GB; unused for text) | `measured` |
| Active weight bytes / decode token | **5.494 GB** (5.116 GiB) across 48 logical layers, 8/128 experts, and 200,058 unembed rows | `inferred` |

---

## 2. One-Token Flow Sketch

```text
token_id
  |
  v
[embed.weight lookup] -> [RMSNorm (embed_norm)] = x
  |
  v
=== Prelude: physical layers 0..5 (logical slots 0..5) ===
  (layer 0 pushes x to blocks[0], starts fresh stream; layers 0..4 SWA, layer 5 Global)
  |
  +---> save (prelude = x, prelude_blocks = [blocks[0]])
  |
  v
=== Core Loop: 3 passes over physical layers 6..17 ===
  Pass 0 (logical slots  6..17): starts from prelude; no cross-pass messages
  Pass 1 (logical slots 24..35): bank = [pass0 block_2, pass0 final x]; reset to prelude;
                                 each core layer mixes bank -> 4 messages -> Q/K/V/FFN controllers
  Pass 2 (logical slots 36..47): bank = [pass1 block_2, pass1 final x]; reset to prelude;
                                 each core layer mixes bank -> 4 messages -> Q/K/V/FFN controllers
  |
  v
=== Epilogue: physical layers 18..23 (logical slots 18..23) ===
  (layers 18..22 SWA, layer 23 Global; each pass/layer has its OWN KV & short-conv cache slot)
  |
  v
[AttnRes mix(blocks, x, output_query)] -> [RMSNorm (norm)]
  -> [unembed.weight (F32 accum)] -> [30 * tanh(logits / 30)] -> slice [:200058] -> logits
```

---

## 3. Complete Tensor Inventory

### 3.1 LLM Backbone (`model-00001-of-00001.safetensors`, `model.llm.*` — 822 tensors, 18,889,358,496 bytes `measured`)

#### Embeddings, Output Head, and Global AttnRes / Cross-Pass Queries (54 tensors)

| Count | Tensor Name Pattern | Layers / Scope | Dtype | Shape | Bytes Each | Total Bytes |
|---:|---|---|---|---:|---:|---:|
| 1 | `model.llm.embed.weight` | Input | `BF16` | `[201024, 2304]` | 926,318,592 | 926,318,592 |
| 1 | `model.llm.embed_norm.weight` | Input | `BF16` | `[2304]` | 4,608 | 4,608 |
| 19 | `model.llm.attn_res.attn_queries.{Q}` | `Q` in `0..22` except `4,10,16,22` (feeds SWA layer `L=Q+1`) | `F32` | `[2304]` | 9,216 | 175,104 |
| 4 | `model.llm.attn_res.attn_queries.{Q}` | `Q` in `{4, 10, 16, 22}` (feeds Global layer `L=Q+1` = `5,11,17,23`) | `F32` | `[3, 2304]` | 27,648 | 110,592 |
| 24 | `model.llm.attn_res.mlp_queries.{L}` | `L = 0..23` | `F32` | `[2304]` | 9,216 | 221,184 |
| 1 | `model.llm.attn_res.output_query` | Output | `F32` | `[2304]` | 9,216 | 9,216 |
| 1 | `model.llm.middle_cycle_cross_pass_queries` | Core `6..17` (Q and MLP rows) | `F32` | `[24, 2304]` | 221,184 | 221,184 |
| 1 | `model.llm.middle_cycle_cross_pass_kv_queries` | Core `6..17` (K and V rows) | `F32` | `[24, 2304]` | 221,184 | 221,184 |
| 1 | `model.llm.norm.weight` | Output | `BF16` | `[2304]` | 4,608 | 4,608 |
| 1 | `model.llm.unembed.weight` | Output (F32 accum) | `BF16` | `[201024, 2304]` | 926,318,592 | 926,318,592 |

#### Per-Layer Attention & Norms (`model.llm.layers.{L}.*` — 352 tensors)

| Count | Tensor Name Pattern | Layers | Dtype | Shape | Bytes Each | Total Bytes |
|---:|---|---|---|---:|---:|---:|
| 20 | `layers.{L}.attn_norm.weight` | SWA (`0..4, 6..10, 12..16, 18..22`) | `BF16` | `[2304]` | 4,608 | 92,160 |
| 4 | `layers.{L}.q_input_norm.weight` | Global (`5, 11, 17, 23`) | `BF16` | `[2304]` | 4,608 | 18,432 |
| 4 | `layers.{L}.k_input_norm.weight` | Global (`5, 11, 17, 23`) | `BF16` | `[2304]` | 4,608 | 18,432 |
| 4 | `layers.{L}.v_input_norm.weight` | Global (`5, 11, 17, 23`) | `BF16` | `[2304]` | 4,608 | 18,432 |
| 24 | `layers.{L}.attn.wq_du.weight` | `0..23` | `BF16` | `[3072, 2304]` | 14,155,776 | 339,738,624 |
| 24 | `layers.{L}.attn.wk_dv.weight` | `0..23` | `BF16` | `[1024, 2304]` | 4,718,592 | 113,246,208 |
| 24 | `layers.{L}.attn.wv_dv.weight` | `0..23` | `BF16` | `[1024, 2304]` | 4,718,592 | 113,246,208 |
| 24 | `layers.{L}.attn.wo_ud.weight` | `0..23` | `BF16` | `[2304, 3072]` | 14,155,776 | 339,738,624 |
| 24 | `layers.{L}.attn.gate_attn.weight` | `0..23` | `BF16` | `[3072, 2304]` | 14,155,776 | 339,738,624 |
| 24 | `layers.{L}.attn.wr_du.weight` | `0..23` | `BF16` | `[384, 2304]` | 1,769,472 | 42,467,328 |
| 24 | `layers.{L}.attn.rel_logits_proj.proj` | `0..23` | `BF16` | `[16, 1024]` | 32,768 | 786,432 |
| 24 | `layers.{L}.attn.q_l2_clip.scale` | `0..23` | `BF16` | `[24, 128]` | 6,144 | 147,456 |
| 24 | `layers.{L}.attn.rel_l2_clip.scale` | `0..23` | `BF16` | `[24, 16]` | 768 | 18,432 |
| 24 | `layers.{L}.attn.attn_sink_proj.weight` | `0..23` | `BF16` | `[24, 2304]` | 110,592 | 2,654,208 |
| 20 | `layers.{L}.attn.k_sconv.weight` | SWA | `BF16` | `[1024, 1, 4]` | 8,192 | 163,840 |
| 20 | `layers.{L}.attn.v_sconv.weight` | SWA | `BF16` | `[1024, 1, 4]` | 8,192 | 163,840 |
| 4 | `layers.{L}.attn.k_sconv.static_w` | Global (`5, 11, 17, 23`) | `BF16` | `[4, 1024]` | 8,192 | 32,768 |
| 4 | `layers.{L}.attn.v_sconv.static_w` | Global (`5, 11, 17, 23`) | `BF16` | `[4, 1024]` | 8,192 | 32,768 |
| 4 | `layers.{L}.attn.k_sconv.weight_proj` | Global (`5, 11, 17, 23`) | `BF16` | `[32, 2304]` | 147,456 | 589,824 |
| 4 | `layers.{L}.attn.v_sconv.weight_proj` | Global (`5, 11, 17, 23`) | `BF16` | `[32, 2304]` | 147,456 | 589,824 |
| 24 | `layers.{L}.attn_sconv.weight` | `0..23` (post-attention) | `BF16` | `[2304, 1, 4]` | 18,432 | 442,368 |

#### Loop Controllers (`model.llm.layers.{L}.controllers.*`, core layers `6..17` — 144 tensors)

For each of the 4 controllers `C` in `{attn (Q), k (K), v (V), mlp (FFN)}`:

| Count | Tensor Name Pattern | Layers | Dtype | Shape | Bytes Each | Total Bytes |
|---:|---|---|---|---:|---:|---:|
| 48 | `layers.{L}.controllers.{C}.down_weight` | `6..17` (4 per layer) | `BF16` | `[64, 4608]` | 589,824 | 28,311,552 |
| 48 | `layers.{L}.controllers.{C}.gain_weight` | `6..17` (4 per layer) | `BF16` | `[2304, 64]` | 294,912 | 14,155,776 |
| 48 | `layers.{L}.controllers.{C}.forget_weight` | `6..17` (4 per layer) | `BF16` | `[2304, 64]` | 294,912 | 14,155,776 |

#### Per-Layer MoE FFN (`model.llm.layers.{L}.*` — 272 tensors)

| Count | Tensor Name Pattern | Layers | Dtype | Shape | Bytes Each | Total Bytes |
|---:|---|---|---|---:|---:|---:|
| 24 | `layers.{L}.mlp_norm.weight` | `0..23` | `BF16` | `[2304]` | 4,608 | 110,592 |
| 24 | `layers.{L}.mlp.gate.weight` | `0..23` (router) | `BF16` | `[128, 2304]` | 589,824 | 14,155,776 |
| 24 | `layers.{L}.mlp.gate.bias` | `0..23` (selection bias) | `F32` | `[128]` | 512 | 12,288 |
| 24 | `layers.{L}.mlp_sconv.weight` | `0..23` (post-MoE) | `BF16` | `[2304, 1, 4]` | 18,432 | 442,368 |
| 20 | `layers.{L}.mlp.experts.w13_weight` | `0..19` (NVFP4 packed) | `U8` | `[128, 2304, 1152]` | 339,738,624 | 6,794,772,480 |
| 20 | `layers.{L}.mlp.experts.w13_weight_scale` | `0..19` (NVFP4 block scale) | `F8_E4M3` | `[128, 2304, 144]` | 42,467,328 | 849,346,560 |
| 20 | `layers.{L}.mlp.experts.w13_weight_amax` | `0..19` (NVFP4 expert amax) | `F32` | `[128]` | 512 | 10,240 |
| 20 | `layers.{L}.mlp.experts.w13_weight_outer_scale_denominator` | `0..19` (value `2688.0`) | `F32` | `[1]` | 4 | 80 |
| 20 | `layers.{L}.mlp.experts.w2_weight` | `0..19` (NVFP4 packed) | `U8` | `[128, 2304, 576]` | 169,869,312 | 3,397,386,240 |
| 20 | `layers.{L}.mlp.experts.w2_weight_scale` | `0..19` (NVFP4 block scale) | `F8_E4M3` | `[128, 2304, 72]` | 21,233,664 | 424,673,280 |
| 20 | `layers.{L}.mlp.experts.w2_weight_amax` | `0..19` (NVFP4 expert amax) | `F32` | `[128]` | 512 | 10,240 |
| 20 | `layers.{L}.mlp.experts.w2_weight_outer_scale_denominator` | `0..19` (value `2688.0`) | `F32` | `[1]` | 4 | 80 |
| 4 | `layers.{L}.mlp.experts.w13_weight` | `20..23` (MXFP8) | `F8_E4M3` | `[128, 2304, 2304]` | 679,477,248 | 2,717,908,992 |
| 4 | `layers.{L}.mlp.experts.w13_weight_scale` | `20..23` (UE8M0 scale) | `U8` | `[128, 2304, 72]` | 21,233,664 | 84,934,656 |
| 4 | `layers.{L}.mlp.experts.w2_weight` | `20..23` (MXFP8) | `F8_E4M3` | `[128, 2304, 1152]` | 339,738,624 | 1,358,954,496 |
| 4 | `layers.{L}.mlp.experts.w2_weight_scale` | `20..23` (UE8M0 scale) | `U8` | `[128, 2304, 36]` | 10,616,832 | 42,467,328 |

*(LLM tensor count check: 54 + 352 + 144 + 272 = **822 tensors**, 18,889,358,496 bytes `measured`.)*

### 3.2 Other Tensors in the Checkpoint (Not Used for Base Text Generation)

- **Vision Tower (`model.visual.*` inside `model-00001-of-00001.safetensors`)**: 422 `BF16` tensors, 636,512,256 bytes (`measured`). 23-layer ViT (`hidden_size = 1024`, 16 heads, `mlp_ratio = 4.0`, `patch_size = 20`) + 2-layer projector (`proj_in [4096, 4096]`, `proj_out [2304, 4096]`). Ignored by text-only inference.
- **Multi-Token Prediction Head (`mtp.safetensors`)**: 54 `BF16` tensors, 635,795,744 bytes data / 635,801,416 bytes file (`measured`). 3-layer dense transformer (`hidden_size = 2304`, `intermediate_size = 9216`, RoPE `theta = 1e6`, SWA on layers 0 & 2) + `fuse_weight [4, 4]` tapping logical layers `[5, 23, 41, 48]` + `markov_head` (`markov_rank = 256`, `block_size = 8`). Unused for base exactness (Stages 1–2); candidate for speculative decoding in Stage 3+.

---

## 4. Storage Formats and Exact Dequantization Rules

1. **BF16 (`BF16`) and F32 (`F32`)**: Standard IEEE-754 little-endian `bfloat16` (upper 16 bits of FP32) and `float32`. All linear weights are stored `[out_features, in_features]`.
2. **NVFP4 (`layers 0..19` MoE experts `w13_weight`, `w2_weight`)**:
   - Two 4-bit FP4 E2M1 nibbles packed per `uint8` byte along the last (`in_features`) dimension:
     - Low nibble (`byte & 0x0F`) -> even column `2k`; high nibble (`byte >> 4`) -> odd column `2k + 1`.
     - E2M1 lookup table (indices `0..15`): `[0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6]`.
   - Per-block scale `*_weight_scale`: `F8_E4M3` (FP8 E4M3FN, bias 7, `0x7F`/`0xFF` = NaN, no infinities), one scale per **16 consecutive columns** (`block = [1, 16]`).
   - Per-expert outer scale: `amax[expert] / outer_scale_denominator[0]`, where `outer_scale_denominator[0] = 2688.0` (`6.0 * 448.0`, `measured`).
   - Exact dequantized weight:
     $$W[e, r, c] = \text{E2M1}(\text{nibble}_{e,r,c}) \times \text{E4M3FN}(\text{scale}[e, r, \lfloor c/16 \rfloor]) \times \frac{\text{amax}[e]}{2688.0}$$
3. **MXFP8 (`layers 20..23` MoE experts `w13_weight`, `w2_weight`)**:
   - Weights are `F8_E4M3` (FP8 E4M3FN).
   - Scales `*_weight_scale` are `U8` UE8M0 exponents (`2^(byte - 127)`, byte `255` = NaN), one scale per **32 consecutive columns** (`block = [1, 32]`).
   - **Important checkpoint quirk (`measured`)**: `hf_quant_config.json` lists `weight_block_shape: [32, 32]`, but the actual stored scale shapes (`[128, 2304, 72]` and `[128, 2304, 36]`) have block shape **`[1, 32]`**. Use the actual `[1, 32]` block shape.
   - Exact dequantized weight:
     $$W[e, r, c] = \text{E4M3FN}(w[e, r, c]) \times 2^{\text{scale}[e, r, \lfloor c/32 \rfloor] - 127}$$
4. **Interleaved `w13_weight` rows (all layers `0..23`)**:
   - Logical shape is `[128, 2304, 2304]` (`2 * intermediate_size = 2304` rows).
   - **Even rows** (`0, 2, 4, ..., 2302`) are the **gate** (`W1`) projection (`1152` rows).
   - **Odd rows** (`1, 3, 5, ..., 2303`) are the **up** (`W3`) projection (`1152` rows).
5. **Short-conv weight ordering**:
   - SWA `k_sconv.weight` / `v_sconv.weight` (`[1024, 1, 4]`), `attn_sconv.weight` (`[2304, 1, 4]`), and `mlp_sconv.weight` (`[2304, 1, 4]`): chronological order `[t-3, t-2, t-1, t]` (index 3 multiplies current token `t`).
   - Global `k_sconv.static_w` / `v_sconv.static_w` (`[4, 1024]`): **reversed lag order** `[lag=0 (t), lag=1 (t-1), lag=2 (t-2), lag=3 (t-3)]`.

---

## 5. Exact Order of Operations for One Token

Let token `token_id` arrive at 0-based sequence position `pos`. All internal activations and norms use `eps = 1e-6`.

### Helper Primitives
- **RMSNorm (no gamma)**: $\text{RMSNorm}(x) = x / \sqrt{\frac{1}{d}\sum x_i^2 + 10^{-6}}$. With weight $\gamma$: $\gamma \odot \text{RMSNorm}(x)$.
- **L2 Clip**: $\text{Clip}(x, s, T)$: if per-channel scale $s$ is present, $x \leftarrow x \odot s$; then $n = \|x\|_2 + 10^{-6}$; return $x \cdot \min(1, T / n)$ per head vector.
- **AttnRes `mix(blocks, cur, Q)`**: Form list $S = [\text{blocks}[0], \dots, \text{blocks}[m-1], \text{cur}]$ (each length 2,304). RMS-normalize each vector in $S$ (no gamma): $\hat{S}_k = \text{RMSNorm}(S_k)$. For each query row $q \in Q$: scores $a_k = (\hat{S}_k \cdot q) / \sqrt{2304}$, weights $w = \text{softmax}(a)$, output $= \sum_k w_k \hat{S}_k$.
- **Short Conv `convolve(x_t, W_static, dyn_t, ring_buf)`**:
  - History has the 3 previous inputs `[x_{t-3}, x_{t-2}, x_{t-1}]` (zero-initialized at sequence start). Push `x_t` to form 4-tap window `[x_{t-3}, x_{t-2}, x_{t-1}, x_t]`, and save `[x_{t-2}, x_{t-1}, x_t]` back into the layer's slot state.
  - Static conv: $y = \sum_{j=0}^{3} W_{\text{chron}}[j] \odot x_{t-3+j}$.
  - If dynamic taps `dyn_t` (`[8 heads, 4 lags]`, Global K/V only): add $\sum_{\text{lag}=0}^{3} \text{dyn}_t[h, \text{lag}] \cdot x_{t-\text{lag}}[h]$ per KV head $h$.
  - Residual + scale: return $(x_t + y) / \sqrt{5}$ (static) or $(x_t + y) / 3$ (static + dynamic, since $\sqrt{2 \cdot 4 + 1} = 3$).
- **Loop Controller `control(x, msg, C)`** (passes 1 and 2 on core layers `6..17`):
  - $u = [\text{RMSNorm}(x), \text{RMSNorm}(\text{msg})]$ (length 4,608).
  - $h = W_{\text{down}} u$ (length 64).
  - $g = \tanh(W_{\text{gain}} h)$, $f = \tanh(W_{\text{forget}} h)$ (each length 2,304).
  - Return $(1 - f) \odot x + g \odot \text{msg}$.

### Step 0: Token Embedding & Log-Scale Factor
1. Look up row `token_id` of `model.llm.embed.weight` -> `cur` (2,304).
2. `cur = RMSNorm(cur) * embed_norm.weight`.
3. Compute scalar log-scale factor for Global layers:
   $$\tau = 1 + 0.1 \cdot \ln\left(\max\left(1,\; \frac{\text{pos} + 1}{8192}\right)\right)$$
4. Initialize empty `blocks = []`.

### Step 1: Layer Schedule (48 Logical Layers Across 24 Physical Layers)
Each execution step `(il, ip)` uses physical weights `ip` and its **own independent cache slot `il`** (separate KV cache and short-conv ring buffer):
1. **Prelude**: `ip = 0..5`, `il = 0..5`, no loop messages.
   - After `ip = 5`: save `prelude = cur` and `prelude_blocks = copy(blocks)` (contains `[block_0]`, pushed at `ip = 0`).
2. **Core Pass 0**: `ip = 6..17`, `il = 6..17`, no loop messages.
3. **Core Pass 1**:
   - Build `bank = blocks[len(prelude_blocks)+1 :]` (i.e. `[block_2]` from `ip = 12`) and `bank_out = cur` (output of `ip = 17`).
   - Reset `cur = prelude`, `blocks = copy(prelude_blocks)`.
   - For `ip = 6..17`, `il = 24 + (ip - 6)` (slots `24..35`): compute 4 cross-pass messages via `mix(bank, bank_out, msg_query[ip])` where `msg_query[ip]` stacks `[queries[2i], kv_queries[2i], kv_queries[2i+1], queries[2i+1]]` (`i = ip - 6`), and run layer `(il, ip)` with controllers.
4. **Core Pass 2**:
   - Rebuild `bank` and `bank_out` from Pass 1's `blocks` and `cur`; reset `cur = prelude`, `blocks = copy(prelude_blocks)`.
   - For `ip = 6..17`, `il = 36 + (ip - 6)` (slots `36..47`): same as Pass 1 with new bank.
5. **Epilogue**: `ip = 18..23`, `il = 18..23`, no loop messages.

### Step 2: Inside One Layer `(il, ip)`
1. Save `inpSA = cur`.
2. **Attention input (AttnRes + optional Controller + RMSNorm)**:
   - If `ip == 0`: `in_q = in_k = in_v = cur`.
   - Else (`ip >= 1`): call `mix(blocks, cur, attn_queries.{ip-1})`.
     - SWA layer: 1 mixed vector shared as `in_q = in_k = in_v`.
     - Global layer (`5, 11, 17, 23`): 3 mixed vectors -> `in_q`, `in_k`, `in_v`.
   - If loop pass 1 or 2: apply `control(in_q, msg_q, attn)`, `control(in_k, msg_k, k)`, `control(in_v, msg_v, v)`. (Note: on SWA layers in passes 1–2, `in_q`, `in_k`, `in_v` now differ.)
   - Normalize:
     - SWA: `RMSNorm(in_*) * attn_norm.weight`.
     - Global: `q_input_norm`, `k_input_norm`, `v_input_norm` respectively.
3. **Projections, Short Convs, Clipping, and Value Norm**:
   - $Q = W_q \cdot \text{in}_q$ (3,072 -> `[24, 128]`), then `Clip(Q, q_l2_clip.scale, 22.627417)`.
   - $K = \text{convolve}(W_k \cdot \text{in}_k,\; k\_\text{sconv},\; \text{dyn}_k)$ (1,024 -> `[8, 128]`), where $\text{dyn}_k = k\_\text{sconv.weight\_proj} \cdot \text{in}_k$ on Global layers only (`None` on SWA). Then `Clip(K, None, 22.627417)`.
   - $V = \text{convolve}(W_v \cdot \text{in}_v,\; v\_\text{sconv},\; \text{dyn}_v)$ (1,024 -> `[8, 128]`), then gamma-free `RMSNorm(V)` per head (`eps = 1e-6`).
   - $R = W_r \cdot \text{in}_q$ (384 -> `[24, 16]`), then `Clip(R, rel_l2_clip.scale, 8.0)`.
   - On Global layers (`5, 11, 17, 23`): multiply $Q \leftarrow \tau \cdot Q$ and $R \leftarrow \tau \cdot R$.
   - Store $K, V$ into slot `il`'s KV cache at `pos`.
4. **Attention with Relative-Position Bias & Per-Query Sink**:
   - Per-head sink logit: $t_h = (W_{\text{sink}} \cdot \text{in}_q)_h$ (`[24]`).
   - Relative bias lookup table: `table[h, d] = R[h] @ rel_logits_proj.proj[:, d]` for distance $d \in [0, 1023]$, and `table[h, 1024] = 0` for $d \ge 1024$.
   - Causal window: SWA attends to keys with $0 \le \text{pos} - \text{pos}_j < 1024$; Global attends to all keys $0 \le \text{pos}_j \le \text{pos}$.
   - For each Q head $h$ (mapped to KV head $\lfloor h / 3 \rfloor$) and valid key $j$ at distance $d = \text{pos} - \text{pos}_j$:
     $$\text{score}_{h,j} = \frac{Q_h \cdot K_{\lfloor h/3 \rfloor, j}}{\sqrt{128}} + \text{table}[h, \min(d, 1024)]$$
   - Softmax with sink logit $t_h$:
     $$p_{h,j} = \frac{\exp(\text{score}_{h,j})}{\exp(t_h) + \sum_{j'} \exp(\text{score}_{h,j'})}$$
     *(Equivalently, subtract $t_h$ from every score and include a virtual zero-valued key with score `0`.)*
   - Weighted sum of $V$: $O_h = \sum_j p_{h,j} V_{\lfloor h/3 \rfloor, j}$ -> concat to length 3,072.
5. **Attention Gate, Output Projection, Post-Conv, and Block Boundary**:
   - Gate: $O \leftarrow O \odot \sigma(W_{\text{gate\_attn}} \cdot \text{in}_q)$.
   - Output projection + post-conv: `attn_out = convolve(W_o @ O, attn_sconv.weight, None)`.
   - **Block boundary rule**:
     - If `ip % 6 == 0` (`ip` in `{0, 6, 12, 18}`): append `inpSA` to `blocks`, and set `ffn_inp = attn_out` (starts a new residual stream).
     - Else: `ffn_inp = inpSA + attn_out`.
6. **MoE FFN**:
   - `x_m = mix(blocks, ffn_inp, mlp_queries.{ip})`.
   - If loop pass 1 or 2: `x_m = control(x_m, msg_ffn, mlp)`.
   - `x_m = RMSNorm(x_m) * mlp_norm.weight`.
   - **Router (F32)**:
     - `probs = sigmoid(W_gate @ x_m)` (length 128).
     - `sel = probs + gate.bias` (bias affects **selection only**, not the weight value).
     - Pick top-8 expert indices `E` by `sel`.
     - Take unbiased weights `w_e = probs[e]` for `e in E`, normalize `w_e <- w_e / (sum(w_e) + 1e-20)`, then scale `w_e <- 8.0 * w_e` (`route_scale = 8.0`).
   - **Experts (`e in E`, 8 of 128)**:
     - Dequantize `w13[e]` (even rows = `W_gate`, odd rows = `W_up`) and `w2[e]`.
     - `g = W_gate @ x_m` (1,152), `u = W_up @ x_m` (1,152).
     - **SiTU activation** (`beta = 4.0`, `linear_beta = 25.0`):
       $$\text{SiTU}(g, u) = \left(4 \tanh\!\left(\frac{g}{4}\right) \cdot \sigma(g)\right) \odot \left(25 \tanh\!\left(\frac{u}{25}\right)\right)$$
     - `y_e = W2[e] @ SiTU(g, u)` (2,304).
   - Combine: `moe_raw = sum_{e in E} w_e * y_e`.
   - Post-MoE short conv: `mlp_out = convolve(moe_raw, mlp_sconv.weight, None)`.
   - Residual add: `cur = ffn_inp + mlp_out`.

### Step 3: Output Head
1. `cur = mix(blocks, cur, output_query)`.
2. `cur = RMSNorm(cur) * norm.weight`.
3. `logits = unembed.weight @ cur` (computed with F32 accumulation).
4. Softcap: `logits = 30.0 * tanh(logits / 30.0)`.
5. Truncate padded vocabulary: return `logits[:200058]`.

---

## 6. Persistent State Footprint (`inferred`)

For context length $N$ (single sequence):
- **Short-conv state** (48 slots x 3 past steps x `(1024 + 1024 + 2304 + 2304) = 6,656` channels):
  - `48 * 3 * 6,656 = 958,464` elements (~3.66 MiB in F32, ~1.83 MiB in FP16/BF16) — independent of context length $N$.
- **KV cache** (`8 KV heads * 128 head_dim = 1,024` elements per K or V per token per slot):
  - **40 SWA slots** (capped at 1,024 tokens): `40 * 1,024 * 2 * 1,024 = 83,886,080` elements (**160 MiB** in FP16/BF16, **320 MiB** in F32) — fixed for any $N \ge 1,024$.
  - **8 Global slots** (scales with $N$): `8 * N * 2 * 1,024 = 16,384 * N` elements.
    - At $N = 128,000$: **4.00 GiB** in FP16/BF16 (**8.00 GiB** in F32).
- **128k VRAM sanity check (`inferred`)**:
  - 18.89 GB (17.59 GiB) LLM weights + 4.16 GiB FP16 KV cache + ~0.5–1.5 GiB activations/scratch = **~22.3–23.3 GiB** on a 32 GiB card -> leaves ~8–9 GiB free (well above Requirement E's 1 GiB free target).
