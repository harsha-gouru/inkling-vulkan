import os, shutil, sys, wgpu
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
from build_batched_spv import FAST_COMMON

# 1. Save tapped hidden state Src[0..C-1, 2304] -> Taps[0..C-1, tap_slot, 2304]
# Dispatch: (9, max_C, 1) where max_C is 1, 8, 16, or 64
D_SAVE_TAP = """
@group(0) @binding(0) var<storage, read> Src: array<f32>;
@group(0) @binding(1) var<storage, read_write> Taps: array<f32>;
@group(0) @binding(2) var<storage, read> Meta: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>; // x=tap_slot (0..3)
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let bi = wg.y;
    if (bi >= Meta[0]) { return; }
    let c = wg.x * 256u + lid.x;
    if (c >= 2304u) { return; }
    Taps[(bi * 4u + params.x) * 2304u + c] = Src[bi * 2304u + c];
}
"""

# 2. Save conv inputs for B=8 verify into Ring[48, 8, C] (supports strided QRSG or contiguous buffers)
# Dispatch: ((8 * C + 255) / 256, 1, 1)
D_SAVE_RING = """
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Ring: array<f32>;
@group(0) @binding(2) var<uniform> params: vec4<u32>; // x=l_slot, y=C, z=src_stride, w=src_off
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x; let C = params.y;
    if (idx >= 8u * C) { return; }
    let t = idx / C;
    let c = idx - t * C;
    Ring[(params.x * 8u + t) * C + c] = X[t * params.z + params.w + c];
}
"""

# 3. GPU-side snapshot of Hist[seq_slot, 48, 3, C] -> Snap[48, 3, C]
# Dispatch: ((3 * C + 255) / 256, 48, 1)
D_SNAP_HIST = """
@group(0) @binding(0) var<storage, read> Hist: array<f32>;
@group(0) @binding(1) var<storage, read_write> Snap: array<f32>;
@group(0) @binding(2) var<storage, read> Meta: array<u32>; // Meta[1] = seq_slot
@group(0) @binding(3) var<uniform> params: vec4<u32>;      // x=C (1024 or 2304)
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let l_slot = wg.y;
    let idx = wg.x * 256u + lid.x;
    let sz = 3u * params.x;
    if (idx >= sz) { return; }
    let seq_slot = Meta[1];
    Snap[l_slot * sz + idx] = Hist[(seq_slot * 48u + l_slot) * sz + idx];
}
"""

# 4. Rollback Hist[seq_slot, 48, 3, C] from Snap[48, 3, C] + Ring[48, 8, C] using n_accept = Meta[4]
# Skipped if Meta[5] != 0u (used when running inject-only after prefill_chunk or B=1 step)
# Dispatch: ((C + 255) / 256, 48, 1)
D_RB_HIST = """
@group(0) @binding(0) var<storage, read> Snap: array<f32>;
@group(0) @binding(1) var<storage, read> Ring: array<f32>;
@group(0) @binding(2) var<storage, read_write> Hist: array<f32>;
@group(0) @binding(3) var<storage, read> Meta: array<u32>; // [1]=seq_slot, [4]=n_accept(1..8), [5]=skip_rb
@group(0) @binding(4) var<uniform> params: vec4<u32>;      // x=C
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    if (Meta[5] != 0u) { return; }
    let l_slot = wg.y;
    let c = wg.x * 256u + lid.x;
    let C = params.x;
    if (c >= C) { return; }
    let K = i32(Meta[4]);
    let seq_slot = Meta[1];
    let hb_snap = l_slot * 3u * C;
    let hb_dst  = (seq_slot * 48u + l_slot) * 3u * C;
    let rb = l_slot * 8u * C;
    for (var j: i32 = 0; j < 3; j++) {
        let src_t = K - 3 + j;
        var v: f32;
        if (src_t >= 0) {
            v = Ring[rb + u32(src_t) * C + c];
        } else {
            v = Snap[hb_snap + u32(src_t + 3) * C + c];
        }
        Hist[hb_dst + u32(j) * C + c] = v;
    }
}
"""

# 5. DSpark context fuse + dflare_norm: Taps[t, 0..3, 2304] -> Out[t, 2304] for t < Meta[4]
# Dispatch: (64, 1, 1) x 256 threads
D_FUSE = """
@group(0) @binding(0) var<storage, read> Taps: array<f32>;
@group(0) @binding(1) var<storage, read> FuseW: array<f32>;
@group(0) @binding(2) var<storage, read> NormW: array<f32>;
@group(0) @binding(3) var<storage, read_write> Out: array<f32>;
@group(0) @binding(4) var<storage, read> Meta: array<u32>; // Meta[4] = n_toks (1..64)
var<workgroup> smem: array<f32, 4>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let t = wg.x;
    if (t >= Meta[4]) { return; }
    let tid = lid.x; let sg = tid >> 6u; let lane = tid & 63u;
    let fb = t * 9216u;
    let w0 = FuseW[0]; let w1 = FuseW[1]; let w2 = FuseW[2]; let w3 = FuseW[3];
    var sq: f32 = 0.0;
    for (var i = tid; i < 2304u; i += 256u) {
        let v = w0 * Taps[fb + i] + w1 * Taps[fb + 2304u + i] + w2 * Taps[fb + 4608u + i] + w3 * Taps[fb + 6912u + i];
        Out[t * 2304u + i] = v;
        sq += v * v;
    }
    let sg_s = subgroupAdd(sq);
    if (lane == 0u) { smem[sg] = sg_s; }
    workgroupBarrier();
    let inv = inverseSqrt((smem[0] + smem[1] + smem[2] + smem[3]) / 2304.0 + 1e-6);
    for (var i = tid; i < 2304u; i += 256u) { Out[t * 2304u + i] *= inv * NormW[i]; }
}
"""

# 6. DSpark QK-Norm + NeoX RoPE (to_cache=0 -> OutF32[T, H, 128]; to_cache=1 -> pack2x16float into CacheU32)
# Dispatch: (H, 8, 1) when to_cache=0; (8, 64, 1) when to_cache=1 (reads Meta[4]=n_toks, Meta[1]=seq_slot, Meta[2]=start_pos)
D_QKN_ROPE = """
@group(0) @binding(0) var<storage, read> In: array<f32>;
@group(0) @binding(1) var<storage, read> NormW: array<f32>;
@group(0) @binding(2) var<storage, read_write> OutF32: array<f32>;
@group(0) @binding(3) var<storage, read_write> CacheU32: array<u32>;
@group(0) @binding(4) var<storage, read> Meta: array<u32>; // [1]=seq_slot, [2]=start_pos, [4]=n_toks, [14]=seq_cap
@group(0) @binding(5) var<uniform> params: vec4<u32>;      // x=H(24/8), y=to_cache(0/1), z=cache_idx(0/1), w=is_swa(0/1)
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let h = wg.x; let t = wg.y; let tid = lid.x;
    if (params.y == 1u && t >= Meta[4]) { return; }
    let H = params.x;
    let ib = (t * H + h) * 128u;
    let v0 = In[ib + tid];
    let v1 = In[ib + tid + 64u];
    let sq = subgroupAdd(v0 * v0 + v1 * v1);
    let inv = inverseSqrt(sq / 128.0 + 1e-6);

    let x0 = v0 * inv * NormW[tid];
    let x1 = v1 * inv * NormW[tid + 64u];

    let pos_u = Meta[2] + t;
    let pos = f32(pos_u);
    let freq = pow(1000000.0, -f32(2u * tid) / 128.0);
    let ang = pos * freq;
    let c = cos(ang); let s = sin(ang);
    let r0 = x0 * c - x1 * s;
    let r1 = x0 * s + x1 * c;

    if (params.y == 1u) {
        let is_swa = params.w;
        let n_layers = select(1u, 2u, is_swa == 1u);
        let cap = select(Meta[14], 1024u, is_swa == 1u);
        let pos_phys = select(pos_u, pos_u & 1023u, is_swa == 1u);
        let c_off = ((Meta[1] * n_layers + params.z) * cap + pos_phys) * 512u + h * 64u;
        CacheU32[c_off + tid] = pack2x16float(vec2<f32>(r0, r1));
    } else {
        OutF32[ib + tid] = r0;
        OutF32[ib + tid + 64u] = r1;
    }
}
"""

# 7. Pack V_ctx[t, 8, 128] into FP16 Draft VCache
# Dispatch: (8, 64, 1) x 64 threads
D_STORE_V = """
@group(0) @binding(0) var<storage, read> In: array<f32>;
@group(0) @binding(1) var<storage, read_write> CacheU32: array<u32>;
@group(0) @binding(2) var<storage, read> Meta: array<u32>; // [1]=seq_slot, [2]=start_pos, [4]=n_toks, [14]=seq_cap
@group(0) @binding(3) var<uniform> params: vec4<u32>;      // x=cache_idx(0/1), y=is_swa(0/1)
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let h = wg.x; let t = wg.y; let tid = lid.x;
    if (t >= Meta[4]) { return; }
    let ib = (t * 8u + h) * 128u;
    let pos_u = Meta[2] + t;
    let is_swa = params.y;
    let n_layers = select(1u, 2u, is_swa == 1u);
    let cap = select(Meta[14], 1024u, is_swa == 1u);
    let pos_phys = select(pos_u, pos_u & 1023u, is_swa == 1u);
    let c_off = ((Meta[1] * n_layers + params.x) * cap + pos_phys) * 512u + h * 64u;
    CacheU32[c_off + tid] = pack2x16float(vec2<f32>(In[ib + tid], In[ib + tid + 64u]));
}
"""

# 8. Copy last accepted token f_rnn CtxTmp[Meta[4]-1] -> CtxAnc[Meta[1]]
# Dispatch: (9, 1, 1) x 256 threads
D_COPY_ANC = """
@group(0) @binding(0) var<storage, read> CtxTmp: array<f32>;
@group(0) @binding(1) var<storage, read_write> CtxAnc: array<f32>;
@group(0) @binding(2) var<storage, read> Meta: array<u32>; // [1]=seq_slot, [4]=n_toks
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let c = wg.x * 256u + lid.x;
    if (c >= 2304u) { return; }
    let last_t = Meta[4] - 1u;
    CtxAnc[Meta[1] * 2304u + c] = CtxTmp[last_t * 2304u + c];
}
"""

# 9. Specialized B=1 16-Rows/WG Unembed for draft7 (4x fewer WGs, no Logits write, no tanh)
# Dispatch: (12504, 1, 1) x 256 threads
D_ATTN = FAST_COMMON + """
@group(0) @binding(0) var<storage, read> W: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> PartV: array<f32>;
@group(0) @binding(3) var<storage, read_write> PartI: array<u32>;
@group(0) @binding(4) var<storage, read> RowScale: array<f32>;
var<workgroup> sh_v: array<f32, 16>;
var<workgroup> sh_i: array<u32, 16>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x; let sub = tid >> 6u; let lane = tid & 63u;
    let r0 = (wg.x * 4u + sub) * 4u;
    var a0: f32 = 0.0; var a1: f32 = 0.0; var a2: f32 = 0.0; var a3: f32 = 0.0;
    if (r0 + 3u < 200058u) {
        let b0 = r0 * 144u;
        for (var k = lane; k < 144u; k += 64u) {
            let xk = k * 4u;
            let x0 = X[xk]; let x1 = X[xk+1u]; let x2 = X[xk+2u]; let x3 = X[xk+3u];
            let w0 = W[b0 + k];         a0 += dot4_fp8(w0.x, x0) + dot4_fp8(w0.y, x1) + dot4_fp8(w0.z, x2) + dot4_fp8(w0.w, x3);
            let w1 = W[b0 + 144u + k];  a1 += dot4_fp8(w1.x, x0) + dot4_fp8(w1.y, x1) + dot4_fp8(w1.z, x2) + dot4_fp8(w1.w, x3);
            let w2 = W[b0 + 288u + k];  a2 += dot4_fp8(w2.x, x0) + dot4_fp8(w2.y, x1) + dot4_fp8(w2.z, x2) + dot4_fp8(w2.w, x3);
            let w3 = W[b0 + 432u + k];  a3 += dot4_fp8(w3.x, x0) + dot4_fp8(w3.y, x1) + dot4_fp8(w3.z, x2) + dot4_fp8(w3.w, x3);
        }
    } else {
        for (var j: u32 = 0u; j < 4u; j++) {
            let r = r0 + j;
            if (r < 200058u) {
                let b = r * 144u;
                var acc: f32 = 0.0;
                for (var k = lane; k < 144u; k += 64u) {
                    let xk = k * 4u; let w = W[b + k];
                    acc += dot4_fp8(w.x, X[xk]) + dot4_fp8(w.y, X[xk+1u]) + dot4_fp8(w.z, X[xk+2u]) + dot4_fp8(w.w, X[xk+3u]);
                }
                if (j == 0u) { a0 = acc; } else if (j == 1u) { a1 = acc; } else if (j == 2u) { a2 = acc; } else { a3 = acc; }
            }
        }
    }
    let s0 = subgroupAdd(a0); let s1 = subgroupAdd(a1); let s2 = subgroupAdd(a2); let s3 = subgroupAdd(a3);
    if (lane == 0u) {
        let sb = sub * 4u;
        sh_v[sb + 0u] = select(-1e30, s0 * RowScale[min(r0 + 0u, 200057u)], r0 + 0u < 200058u); sh_i[sb + 0u] = r0 + 0u;
        sh_v[sb + 1u] = select(-1e30, s1 * RowScale[min(r0 + 1u, 200057u)], r0 + 1u < 200058u); sh_i[sb + 1u] = r0 + 1u;
        sh_v[sb + 2u] = select(-1e30, s2 * RowScale[min(r0 + 2u, 200057u)], r0 + 2u < 200058u); sh_i[sb + 2u] = r0 + 2u;
        sh_v[sb + 3u] = select(-1e30, s3 * RowScale[min(r0 + 3u, 200057u)], r0 + 3u < 200058u); sh_i[sb + 3u] = r0 + 3u;
    }
    workgroupBarrier();
    if (tid == 0u) {
        var bv: f32 = -1e30; var bi: u32 = 0u;
        for (var j: u32 = 0u; j < 16u; j++) {
            if (sh_v[j] > bv) { bv = sh_v[j]; bi = sh_i[j]; }
        }
        PartV[wg.x] = bv; PartI[wg.x] = bi;
    }
}
"""

# 10. Fused Residual Add + RMSNorm: Cur[b] += Add[b]; Out[b] = rms_norm(Cur[b], NormW)
# Dispatch: (1, 8, 1) x 256 threads
D_ADD = """
@group(0) @binding(0) var<storage, read_write> Cur: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> Add: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> NormW: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> Out: array<vec4<f32>>;
var<workgroup> smem: array<f32, 4>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let b = wg.y; let tid = lid.x; let sg = tid >> 6u; let lane = tid & 63u;
    let base = b * 576u;
    var s: f32 = 0.0;
    for (var i = tid; i < 576u; i += 256u) {
        let v = Cur[base + i] + Add[base + i];
        Cur[base + i] = v;
        s += dot(v, v);
    }
    let sg_s = subgroupAdd(s);
    if (lane == 0u) { smem[sg] = sg_s; }
    workgroupBarrier();
    let inv = inverseSqrt((smem[0] + smem[1] + smem[2] + smem[3]) / 2304.0 + 1e-6);
    for (var i = tid; i < 576u; i += 256u) {
        Out[base + i] = Cur[base + i] * inv * NormW[i];
    }
}
"""

# 11. Fused 8-Token FP8 Gate + Up GEMM + SwiGLU: Out[t, row] = silu(WGate[row] @ X[t]) * (WUp[row] @ X[t])
# Dispatch: (4608, 1, 1) x 128 threads (2 rows per WG -> 9216 rows)
D_SWIGLU = FAST_COMMON + """
@group(0) @binding(0) var<storage, read> WGate: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> WUp: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x; let sub = tid >> 6u; let lane = tid & 63u;
    let row = wg.x * 2u + sub;
    let base = row * 144u;
    var g0: f32=0.0; var u0: f32=0.0; var g1: f32=0.0; var u1: f32=0.0;
    var g2: f32=0.0; var u2: f32=0.0; var g3: f32=0.0; var u3: f32=0.0;
    var g4: f32=0.0; var u4: f32=0.0; var g5: f32=0.0; var u5: f32=0.0;
    var g6: f32=0.0; var u6: f32=0.0; var g7: f32=0.0; var u7: f32=0.0;
    for (var k = lane; k < 144u; k += 64u) {
        let wg_w = WGate[base + k]; let wu_w = WUp[base + k];
        let xk = k * 4u;
        { let o = 0u*576u+xk; let x0 = X[o]; let x1 = X[o+1u]; let x2 = X[o+2u]; let x3 = X[o+3u];
          g0 += dot4_fp8(wg_w.x,x0)+dot4_fp8(wg_w.y,x1)+dot4_fp8(wg_w.z,x2)+dot4_fp8(wg_w.w,x3);
          u0 += dot4_fp8(wu_w.x,x0)+dot4_fp8(wu_w.y,x1)+dot4_fp8(wu_w.z,x2)+dot4_fp8(wu_w.w,x3); }
        { let o = 1u*576u+xk; let x0 = X[o]; let x1 = X[o+1u]; let x2 = X[o+2u]; let x3 = X[o+3u];
          g1 += dot4_fp8(wg_w.x,x0)+dot4_fp8(wg_w.y,x1)+dot4_fp8(wg_w.z,x2)+dot4_fp8(wg_w.w,x3);
          u1 += dot4_fp8(wu_w.x,x0)+dot4_fp8(wu_w.y,x1)+dot4_fp8(wu_w.z,x2)+dot4_fp8(wu_w.w,x3); }
        { let o = 2u*576u+xk; let x0 = X[o]; let x1 = X[o+1u]; let x2 = X[o+2u]; let x3 = X[o+3u];
          g2 += dot4_fp8(wg_w.x,x0)+dot4_fp8(wg_w.y,x1)+dot4_fp8(wg_w.z,x2)+dot4_fp8(wg_w.w,x3);
          u2 += dot4_fp8(wu_w.x,x0)+dot4_fp8(wu_w.y,x1)+dot4_fp8(wu_w.z,x2)+dot4_fp8(wu_w.w,x3); }
        { let o = 3u*576u+xk; let x0 = X[o]; let x1 = X[o+1u]; let x2 = X[o+2u]; let x3 = X[o+3u];
          g3 += dot4_fp8(wg_w.x,x0)+dot4_fp8(wg_w.y,x1)+dot4_fp8(wg_w.z,x2)+dot4_fp8(wg_w.w,x3);
          u3 += dot4_fp8(wu_w.x,x0)+dot4_fp8(wu_w.y,x1)+dot4_fp8(wu_w.z,x2)+dot4_fp8(wu_w.w,x3); }
        { let o = 4u*576u+xk; let x0 = X[o]; let x1 = X[o+1u]; let x2 = X[o+2u]; let x3 = X[o+3u];
          g4 += dot4_fp8(wg_w.x,x0)+dot4_fp8(wg_w.y,x1)+dot4_fp8(wg_w.z,x2)+dot4_fp8(wg_w.w,x3);
          u4 += dot4_fp8(wu_w.x,x0)+dot4_fp8(wu_w.y,x1)+dot4_fp8(wu_w.z,x2)+dot4_fp8(wu_w.w,x3); }
        { let o = 5u*576u+xk; let x0 = X[o]; let x1 = X[o+1u]; let x2 = X[o+2u]; let x3 = X[o+3u];
          g5 += dot4_fp8(wg_w.x,x0)+dot4_fp8(wg_w.y,x1)+dot4_fp8(wg_w.z,x2)+dot4_fp8(wg_w.w,x3);
          u5 += dot4_fp8(wu_w.x,x0)+dot4_fp8(wu_w.y,x1)+dot4_fp8(wu_w.z,x2)+dot4_fp8(wu_w.w,x3); }
        { let o = 6u*576u+xk; let x0 = X[o]; let x1 = X[o+1u]; let x2 = X[o+2u]; let x3 = X[o+3u];
          g6 += dot4_fp8(wg_w.x,x0)+dot4_fp8(wg_w.y,x1)+dot4_fp8(wg_w.z,x2)+dot4_fp8(wg_w.w,x3);
          u6 += dot4_fp8(wu_w.x,x0)+dot4_fp8(wu_w.y,x1)+dot4_fp8(wu_w.z,x2)+dot4_fp8(wu_w.w,x3); }
        { let o = 7u*576u+xk; let x0 = X[o]; let x1 = X[o+1u]; let x2 = X[o+2u]; let x3 = X[o+3u];
          g7 += dot4_fp8(wg_w.x,x0)+dot4_fp8(wg_w.y,x1)+dot4_fp8(wg_w.z,x2)+dot4_fp8(wg_w.w,x3);
          u7 += dot4_fp8(wu_w.x,x0)+dot4_fp8(wu_w.y,x1)+dot4_fp8(wu_w.z,x2)+dot4_fp8(wu_w.w,x3); }
    }
    let sg_v4 = WGate[9216u * 144u + (row >> 2u)];
    let su_v4 = WUp[9216u * 144u + (row >> 2u)];
    let c_idx = row & 3u;
    let sc_g = bitcast<f32>(select(select(sg_v4.x, sg_v4.y, c_idx == 1u), select(sg_v4.z, sg_v4.w, c_idx == 3u), c_idx >= 2u));
    let sc_u = bitcast<f32>(select(select(su_v4.x, su_v4.y, c_idx == 1u), select(su_v4.z, su_v4.w, c_idx == 3u), c_idx >= 2u));
    var gs = array<f32,8>(subgroupAdd(g0)*sc_g,subgroupAdd(g1)*sc_g,subgroupAdd(g2)*sc_g,subgroupAdd(g3)*sc_g,subgroupAdd(g4)*sc_g,subgroupAdd(g5)*sc_g,subgroupAdd(g6)*sc_g,subgroupAdd(g7)*sc_g);
    var us = array<f32,8>(subgroupAdd(u0)*sc_u,subgroupAdd(u1)*sc_u,subgroupAdd(u2)*sc_u,subgroupAdd(u3)*sc_u,subgroupAdd(u4)*sc_u,subgroupAdd(u5)*sc_u,subgroupAdd(u6)*sc_u,subgroupAdd(u7)*sc_u);
    if (lane == 0u) {
        for (var t: u32 = 0u; t < 8u; t++) {
            let g = gs[t];
            Out[t * 9216u + row] = (g / (1.0 + exp(-g))) * us[t];
        }
    }
}
"""

# 12. Markov RNN Pass 1 (subgroupAdd, reads CtxAnc[Meta[1]])
# Dispatch: (768, 1, 1) x 64 threads
D_MRNN = FAST_COMMON + """
@group(0) @binding(0) var<storage, read> State: array<f32>;
@group(0) @binding(1) var<storage, read> MarkovW1: array<u32>;
@group(0) @binding(2) var<storage, read> Cur: array<f32>;
@group(0) @binding(3) var<storage, read> CtxAnc: array<f32>;
@group(0) @binding(4) var<storage, read> RnnW: array<u32>;
@group(0) @binding(5) var<storage, read> RnnB: array<f32>;
@group(0) @binding(6) var<storage, read> DraftToks: array<u32>;
@group(0) @binding(7) var<storage, read_write> JOut: array<f32>;
@group(0) @binding(8) var<storage, read> Meta: array<u32>; // Meta[1] = seq_slot
@group(0) @binding(9) var<uniform> params: vec4<u32>;      // x=step (1..7)
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wg.x; let tid = lid.x; let step = params.x;
    let prev = DraftToks[step - 1u];
    let w_base = row * 2560u;
    let m1_base = prev * 128u;
    let cur_base = step * 2304u;
    let ctx_base = Meta[1] * 2304u;

    var acc: f32 = 0.0;
    for (var k = tid; k < 128u; k += 64u) {
        acc += bf16x2_dot(RnnW[w_base + k], vec2<f32>(State[2u * k], State[2u * k + 1u]));
    }
    for (var k = tid; k < 128u; k += 64u) {
        let mw = MarkovW1[m1_base + k];
        let x = vec2<f32>(bitcast<f32>((mw & 0xFFFFu) << 16u), bitcast<f32>(mw & 0xFFFF0000u));
        acc += bf16x2_dot(RnnW[w_base + 128u + k], x);
    }
    for (var k = tid; k < 1152u; k += 64u) {
        acc += bf16x2_dot(RnnW[w_base + 256u + k], vec2<f32>(Cur[cur_base + 2u * k], Cur[cur_base + 2u * k + 1u]));
    }
    for (var k = tid; k < 1152u; k += 64u) {
        acc += bf16x2_dot(RnnW[w_base + 1408u + k], vec2<f32>(CtxAnc[ctx_base + 2u * k], CtxAnc[ctx_base + 2u * k + 1u]));
    }
    let s = subgroupAdd(acc);
    if (tid == 0u) { JOut[row] = s + RnnB[row]; }
}
"""

# 13. Markov RNN Pass 2: update State[256] + Comb[2304] = Cur[step] + hidden_out @ tanh(JOut[512:768]) + b
# Dispatch: (9, 1, 1) x 256 threads
D_MUPD = FAST_COMMON + """
@group(0) @binding(0) var<storage, read> JOut: array<f32>;
@group(0) @binding(1) var<storage, read_write> State: array<f32>;
@group(0) @binding(2) var<storage, read> Cur: array<f32>;
@group(0) @binding(3) var<storage, read> HiddenW: array<u32>;
@group(0) @binding(4) var<storage, read> HiddenB: array<f32>;
@group(0) @binding(5) var<storage, read_write> Comb: array<f32>;
@group(0) @binding(6) var<uniform> params: vec4<u32>; // x=step
var<workgroup> lat: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x; let step = params.x;
    if (wg.x == 0u) {
        let g = 1.0 / (1.0 + exp(-JOut[tid]));
        let c = tanh(JOut[256u + tid]);
        State[tid] = c + g * (State[tid] - c);
    }
    lat[tid] = tanh(JOut[512u + tid]);
    workgroupBarrier();
    let row = wg.x * 256u + tid;
    if (row >= 2304u) { return; }
    let base = row * 128u;
    var acc: f32 = HiddenB[row];
    for (var k: u32 = 0u; k < 128u; k++) {
        acc += bf16x2_dot(HiddenW[base + k], vec2<f32>(lat[2u * k], lat[2u * k + 1u]));
    }
    Comb[row] = Cur[step * 2304u + row] + acc;
}
"""

# 14. Argmax over first 50015 entries of PartV/PartI -> DraftToks[step]
# Dispatch: (1, 1, 1) x 256 threads
D_AM_SLOT = """
@group(0) @binding(0) var<storage, read> PartV: array<f32>;
@group(0) @binding(1) var<storage, read> PartI: array<u32>;
@group(0) @binding(2) var<storage, read_write> DraftToks: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>; // x=step
var<workgroup> smem_v: array<f32, 256>;
var<workgroup> smem_i: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    var bv: f32 = -1e30; var bi: u32 = 0u;
    for (var k = tid; k < 12504u; k += 256u) {
        let v = PartV[k];
        if (v > bv) { bv = v; bi = PartI[k]; }
    }
    smem_v[tid] = bv; smem_i[tid] = bi; workgroupBarrier();
    for (var st = 128u; st > 0u; st >>= 1u) {
        if (tid < st && smem_v[tid + st] > smem_v[tid]) { smem_v[tid] = smem_v[tid + st]; smem_i[tid] = smem_i[tid + st]; }
        workgroupBarrier();
    }
    if (tid == 0u) { DraftToks[params.x] = smem_i[0]; }
}
"""

# 15. 4-token Scalar-Register KV-sharing Split-K Attention for verify8 (grid: (24, 2, 16))
V_ATTN_SK = FAST_COMMON + """
@group(0) @binding(0) var<storage, read> QRSG: array<f32>;
@group(0) @binding(1) var<storage, read> RelBias: array<f32>;
@group(0) @binding(2) var<storage, read> KCache0: array<u32>;
@group(0) @binding(3) var<storage, read> KCache1: array<u32>;
@group(0) @binding(4) var<storage, read> VCache0: array<u32>;
@group(0) @binding(5) var<storage, read> VCache1: array<u32>;
@group(0) @binding(6) var<storage, read_write> PartML: array<f32>;
@group(0) @binding(7) var<storage, read_write> PartAcc: array<f32>;
@group(0) @binding(8) var<storage, read> Meta: array<u32>;
@group(0) @binding(9) var<uniform> params: vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let h = wg.x; let b_grp = wg.y; let split = wg.z; let kv_h = h / 3u; let tid = lid.x;
    let b0 = b_grp * 4u;
    let cache_idx = params.x; let is_swa = params.y;
    let seq_slot = Meta[1];
    let pos0 = Meta[2] + b0;
    let pos_max = pos0 + 3u;

    let k_start: u32 = select(0u, select(0u, pos0 - 1023u, pos0 >= 1024u), is_swa == 1u);
    let total_k = pos_max - k_start + 1u;
    let chunk = (total_k + 15u) >> 4u;
    let s_begin = k_start + split * chunk;
    let s_end = min(pos_max + 1u, s_begin + chunk);

    if (s_begin >= s_end) {
        for (var j: u32 = 0u; j < 4u; j++) {
            let bi = b0 + j;
            let ml_idx = (bi * 384u + h * 16u + split) * 2u;
            let ac_base = (bi * 384u + h * 16u + split) * 128u;
            if (tid == 0u) { PartML[ml_idx] = -1e30; PartML[ml_idx + 1u] = 0.0; }
            PartAcc[ac_base + tid] = 0.0; PartAcc[ac_base + tid + 64u] = 0.0;
        }
        return;
    }

    let qb0 = (b0 + 0u) * 8600u + h * 128u; let q0_0 = QRSG[qb0 + tid]; let q1_0 = QRSG[qb0 + tid + 64u];
    let qb1 = (b0 + 1u) * 8600u + h * 128u; let q0_1 = QRSG[qb1 + tid]; let q1_1 = QRSG[qb1 + tid + 64u];
    let qb2 = (b0 + 2u) * 8600u + h * 128u; let q0_2 = QRSG[qb2 + tid]; let q1_2 = QRSG[qb2 + tid + 64u];
    let qb3 = (b0 + 3u) * 8600u + h * 128u; let q0_3 = QRSG[qb3 + tid]; let q1_3 = QRSG[qb3 + tid + 64u];
    let sk0 = QRSG[(b0 + 0u) * 8600u + 3456u + h];
    let sk1 = QRSG[(b0 + 1u) * 8600u + 3456u + h];
    let sk2 = QRSG[(b0 + 2u) * 8600u + 3456u + h];
    let sk3 = QRSG[(b0 + 3u) * 8600u + 3456u + h];
    var r0: f32 = 0.0; var r1: f32 = 0.0; var r2: f32 = 0.0; var r3: f32 = 0.0;
    if (tid < 16u) {
        r0 = QRSG[(b0 + 0u) * 8600u + 3072u + h * 16u + tid];
        r1 = QRSG[(b0 + 1u) * 8600u + 3072u + h * 16u + tid];
        r2 = QRSG[(b0 + 2u) * 8600u + 3072u + h * 16u + tid];
        r3 = QRSG[(b0 + 3u) * 8600u + 3072u + h * 16u + tid];
    }
    let rb_base = (tid & 15u) * 1024u;

    let inv_sqrt_d: f32 = 0.005524271728019903;
    let seq_cap = Meta[14]; let half_b = Meta[15];
    let hi = (seq_slot >= half_b);
    let half_slot = select(seq_slot, seq_slot - half_b, hi);
    let n_layers = select(8u, 40u, is_swa == 1u);
    let cap = select(seq_cap, 1024u, is_swa == 1u);
    let slot_base = (half_slot * n_layers + cache_idx) * cap * 256u + kv_h * 32u + (tid & 31u);
    let sh0 = (tid >> 5u) * 8u; let sh1 = sh0 + 16u;

    var m0: f32 = -1e30; var l0: f32 = 0.0; var a0_0: f32 = 0.0; var a1_0: f32 = 0.0;
    var m1: f32 = -1e30; var l1: f32 = 0.0; var a0_1: f32 = 0.0; var a1_1: f32 = 0.0;
    var m2: f32 = -1e30; var l2: f32 = 0.0; var a0_2: f32 = 0.0; var a1_2: f32 = 0.0;
    var m3: f32 = -1e30; var l3: f32 = 0.0; var a0_3: f32 = 0.0; var a1_3: f32 = 0.0;

    let fast_end: u32 = select(s_begin, min(s_end, pos0 - 1023u), is_swa == 0u && pos0 >= 1024u);
    var k = s_begin;
    for (; k < fast_end; k++) {
        let kv_off = slot_base + k * 256u;
        var kp_u: u32; var vp_u: u32;
        if (!hi) { kp_u = KCache0[kv_off]; vp_u = VCache0[kv_off]; }
        else     { kp_u = KCache1[kv_off]; vp_u = VCache1[kv_off]; }
        let k0 = decode_fp8((kp_u >> sh0) & 0xFFu);
        let k1 = decode_fp8((kp_u >> sh1) & 0xFFu);
        let v0 = decode_fp8((vp_u >> sh0) & 0xFFu);
        let v1 = decode_fp8((vp_u >> sh1) & 0xFFu);

        { let sc = subgroupAdd(q0_0 * k0 + q1_0 * k1) * inv_sqrt_d - sk0; let mn = max(m0, sc); let al = exp(m0 - mn); let p = exp(sc - mn); l0 = l0 * al + p; a0_0 = a0_0 * al + p * v0; a1_0 = a1_0 * al + p * v1; m0 = mn; }
        { let sc = subgroupAdd(q0_1 * k0 + q1_1 * k1) * inv_sqrt_d - sk1; let mn = max(m1, sc); let al = exp(m1 - mn); let p = exp(sc - mn); l1 = l1 * al + p; a0_1 = a0_1 * al + p * v0; a1_1 = a1_1 * al + p * v1; m1 = mn; }
        { let sc = subgroupAdd(q0_2 * k0 + q1_2 * k1) * inv_sqrt_d - sk2; let mn = max(m2, sc); let al = exp(m2 - mn); let p = exp(sc - mn); l2 = l2 * al + p; a0_2 = a0_2 * al + p * v0; a1_2 = a1_2 * al + p * v1; m2 = mn; }
        { let sc = subgroupAdd(q0_3 * k0 + q1_3 * k1) * inv_sqrt_d - sk3; let mn = max(m3, sc); let al = exp(m3 - mn); let p = exp(sc - mn); l3 = l3 * al + p; a0_3 = a0_3 * al + p * v0; a1_3 = a1_3 * al + p * v1; m3 = mn; }
    }
    for (; k < s_end; k++) {
        let k_phys = select(k, k & 1023u, is_swa == 1u);
        let kv_off = slot_base + k_phys * 256u;
        var kp_u: u32; var vp_u: u32;
        if (!hi) { kp_u = KCache0[kv_off]; vp_u = VCache0[kv_off]; }
        else     { kp_u = KCache1[kv_off]; vp_u = VCache1[kv_off]; }
        let k0 = decode_fp8((kp_u >> sh0) & 0xFFu);
        let k1 = decode_fp8((kp_u >> sh1) & 0xFFu);
        let v0 = decode_fp8((vp_u >> sh0) & 0xFFu);
        let v1 = decode_fp8((vp_u >> sh1) & 0xFFu);

        if (k <= pos0) {
            let d = pos0 - k;
            if (is_swa == 0u || d < 1024u) {
                var sc = subgroupAdd(q0_0 * k0 + q1_0 * k1) * inv_sqrt_d - sk0;
                if (d < 1024u) { sc += subgroupAdd(select(0.0, r0 * RelBias[rb_base + d], tid < 16u)); }
                let mn = max(m0, sc); let al = exp(m0 - mn); let p = exp(sc - mn); l0 = l0 * al + p; a0_0 = a0_0 * al + p * v0; a1_0 = a1_0 * al + p * v1; m0 = mn;
            }
        }
        if (k <= pos0 + 1u) {
            let d = pos0 + 1u - k;
            if (is_swa == 0u || d < 1024u) {
                var sc = subgroupAdd(q0_1 * k0 + q1_1 * k1) * inv_sqrt_d - sk1;
                if (d < 1024u) { sc += subgroupAdd(select(0.0, r1 * RelBias[rb_base + d], tid < 16u)); }
                let mn = max(m1, sc); let al = exp(m1 - mn); let p = exp(sc - mn); l1 = l1 * al + p; a0_1 = a0_1 * al + p * v0; a1_1 = a1_1 * al + p * v1; m1 = mn;
            }
        }
        if (k <= pos0 + 2u) {
            let d = pos0 + 2u - k;
            if (is_swa == 0u || d < 1024u) {
                var sc = subgroupAdd(q0_2 * k0 + q1_2 * k1) * inv_sqrt_d - sk2;
                if (d < 1024u) { sc += subgroupAdd(select(0.0, r2 * RelBias[rb_base + d], tid < 16u)); }
                let mn = max(m2, sc); let al = exp(m2 - mn); let p = exp(sc - mn); l2 = l2 * al + p; a0_2 = a0_2 * al + p * v0; a1_2 = a1_2 * al + p * v1; m2 = mn;
            }
        }
        if (k <= pos0 + 3u) {
            let d = pos0 + 3u - k;
            if (is_swa == 0u || d < 1024u) {
                var sc = subgroupAdd(q0_3 * k0 + q1_3 * k1) * inv_sqrt_d - sk3;
                if (d < 1024u) { sc += subgroupAdd(select(0.0, r3 * RelBias[rb_base + d], tid < 16u)); }
                let mn = max(m3, sc); let al = exp(m3 - mn); let p = exp(sc - mn); l3 = l3 * al + p; a0_3 = a0_3 * al + p * v0; a1_3 = a1_3 * al + p * v1; m3 = mn;
            }
        }
    }

    { let bi = b0 + 0u; let ml = (bi * 384u + h * 16u + split) * 2u; let ac = (bi * 384u + h * 16u + split) * 128u; if (tid == 0u) { PartML[ml] = m0; PartML[ml+1u] = l0; } PartAcc[ac+tid] = a0_0 * 0.03125; PartAcc[ac+tid+64u] = a1_0 * 0.03125; }
    { let bi = b0 + 1u; let ml = (bi * 384u + h * 16u + split) * 2u; let ac = (bi * 384u + h * 16u + split) * 128u; if (tid == 0u) { PartML[ml] = m1; PartML[ml+1u] = l1; } PartAcc[ac+tid] = a0_1 * 0.03125; PartAcc[ac+tid+64u] = a1_1 * 0.03125; }
    { let bi = b0 + 2u; let ml = (bi * 384u + h * 16u + split) * 2u; let ac = (bi * 384u + h * 16u + split) * 128u; if (tid == 0u) { PartML[ml] = m2; PartML[ml+1u] = l2; } PartAcc[ac+tid] = a0_2 * 0.03125; PartAcc[ac+tid+64u] = a1_2 * 0.03125; }
    { let bi = b0 + 3u; let ml = (bi * 384u + h * 16u + split) * 2u; let ac = (bi * 384u + h * 16u + split) * 128u; if (tid == 0u) { PartML[ml] = m3; PartML[ml+1u] = l3; } PartAcc[ac+tid] = a0_3 * 0.03125; PartAcc[ac+tid+64u] = a1_3 * 0.03125; }
}
"""

# 16. 4-token Scalar-Register KV-sharing Split-K Attention for draft7 (grid: (24, 2, 16))
D_ATTN_SK = """
@group(0) @binding(0) var<storage, read> Q: array<f32>;         // [8, 24, 128]
@group(0) @binding(1) var<storage, read> KBlk: array<f32>;      // [8, 8, 128]
@group(0) @binding(2) var<storage, read> VBlk: array<f32>;      // [8, 8, 128]
@group(0) @binding(3) var<storage, read> KCacheU32: array<u32>;
@group(0) @binding(4) var<storage, read> VCacheU32: array<u32>;
@group(0) @binding(5) var<storage, read_write> PartML: array<f32>;
@group(0) @binding(6) var<storage, read_write> PartAcc: array<f32>;
@group(0) @binding(7) var<storage, read> Meta: array<u32>;      // [1]=seq_slot, [2]=ctx_len, [14]=seq_cap
@group(0) @binding(8) var<uniform> params: vec4<u32>;           // x=cache_idx(0/1), y=is_swa(0/1)

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let h = wg.x; let b_grp = wg.y; let split = wg.z; let kv_h = h / 3u; let tid = lid.x;
    let b0 = b_grp * 4u;
    let c_idx = params.x; let is_swa = params.y;
    let n = Meta[2];
    let k_start: u32 = select(0u, select(0u, n - 1024u, n >= 1024u), is_swa == 1u);
    let total_k = n - k_start;
    let chunk = (total_k + 15u) >> 4u;
    let s_begin = k_start + split * chunk;
    let s_end = min(n, s_begin + chunk);

    let qb0 = ((b0 + 0u) * 24u + h) * 128u; let q0_0 = Q[qb0 + tid]; let q1_0 = Q[qb0 + tid + 64u];
    let qb1 = ((b0 + 1u) * 24u + h) * 128u; let q0_1 = Q[qb1 + tid]; let q1_1 = Q[qb1 + tid + 64u];
    let qb2 = ((b0 + 2u) * 24u + h) * 128u; let q0_2 = Q[qb2 + tid]; let q1_2 = Q[qb2 + tid + 64u];
    let qb3 = ((b0 + 3u) * 24u + h) * 128u; let q0_3 = Q[qb3 + tid]; let q1_3 = Q[qb3 + tid + 64u];

    let inv_sqrt_d: f32 = 0.08838834764831845;
    let n_layers = select(1u, 2u, is_swa == 1u);
    let cap = select(Meta[14], 1024u, is_swa == 1u);
    let slot_base = (Meta[1] * n_layers + c_idx) * cap * 512u + kv_h * 64u + tid;

    var m0: f32 = -1e30; var l0: f32 = 0.0; var a0_0: f32 = 0.0; var a1_0: f32 = 0.0;
    var m1: f32 = -1e30; var l1: f32 = 0.0; var a0_1: f32 = 0.0; var a1_1: f32 = 0.0;
    var m2: f32 = -1e30; var l2: f32 = 0.0; var a0_2: f32 = 0.0; var a1_2: f32 = 0.0;
    var m3: f32 = -1e30; var l3: f32 = 0.0; var a0_3: f32 = 0.0; var a1_3: f32 = 0.0;

    for (var k = s_begin; k < s_end; k++) {
        let k_phys = select(k, k & 1023u, is_swa == 1u);
        let ko = slot_base + k_phys * 512u;
        let kp = unpack2x16float(KCacheU32[ko]);
        let vp = unpack2x16float(VCacheU32[ko]);
        { let sc = subgroupAdd(q0_0 * kp.x + q1_0 * kp.y) * inv_sqrt_d; let mn = max(m0, sc); let al = exp(m0 - mn); let p = exp(sc - mn); l0 = l0 * al + p; a0_0 = a0_0 * al + p * vp.x; a1_0 = a1_0 * al + p * vp.y; m0 = mn; }
        { let sc = subgroupAdd(q0_1 * kp.x + q1_1 * kp.y) * inv_sqrt_d; let mn = max(m1, sc); let al = exp(m1 - mn); let p = exp(sc - mn); l1 = l1 * al + p; a0_1 = a0_1 * al + p * vp.x; a1_1 = a1_1 * al + p * vp.y; m1 = mn; }
        { let sc = subgroupAdd(q0_2 * kp.x + q1_2 * kp.y) * inv_sqrt_d; let mn = max(m2, sc); let al = exp(m2 - mn); let p = exp(sc - mn); l2 = l2 * al + p; a0_2 = a0_2 * al + p * vp.x; a1_2 = a1_2 * al + p * vp.y; m2 = mn; }
        { let sc = subgroupAdd(q0_3 * kp.x + q1_3 * kp.y) * inv_sqrt_d; let mn = max(m3, sc); let al = exp(m3 - mn); let p = exp(sc - mn); l3 = l3 * al + p; a0_3 = a0_3 * al + p * vp.x; a1_3 = a1_3 * al + p * vp.y; m3 = mn; }
    }

    if (split == 0u) {
        for (var k: u32 = 0u; k < 8u; k++) {
            let bo = (k * 8u + kv_h) * 128u;
            let kb0 = KBlk[bo + tid]; let kb1 = KBlk[bo + tid + 64u];
            let vb0 = VBlk[bo + tid]; let vb1 = VBlk[bo + tid + 64u];
            { let sc = subgroupAdd(q0_0 * kb0 + q1_0 * kb1) * inv_sqrt_d; let mn = max(m0, sc); let al = exp(m0 - mn); let p = exp(sc - mn); l0 = l0 * al + p; a0_0 = a0_0 * al + p * vb0; a1_0 = a1_0 * al + p * vb1; m0 = mn; }
            { let sc = subgroupAdd(q0_1 * kb0 + q1_1 * kb1) * inv_sqrt_d; let mn = max(m1, sc); let al = exp(m1 - mn); let p = exp(sc - mn); l1 = l1 * al + p; a0_1 = a0_1 * al + p * vb0; a1_1 = a1_1 * al + p * vb1; m1 = mn; }
            { let sc = subgroupAdd(q0_2 * kb0 + q1_2 * kb1) * inv_sqrt_d; let mn = max(m2, sc); let al = exp(m2 - mn); let p = exp(sc - mn); l2 = l2 * al + p; a0_2 = a0_2 * al + p * vb0; a1_2 = a1_2 * al + p * vb1; m2 = mn; }
            { let sc = subgroupAdd(q0_3 * kb0 + q1_3 * kb1) * inv_sqrt_d; let mn = max(m3, sc); let al = exp(m3 - mn); let p = exp(sc - mn); l3 = l3 * al + p; a0_3 = a0_3 * al + p * vb0; a1_3 = a1_3 * al + p * vb1; m3 = mn; }
        }
    }

    { let t = b0 + 0u; let ml = (t * 384u + h * 16u + split) * 2u; let ac = (t * 384u + h * 16u + split) * 128u; if (tid == 0u) { PartML[ml] = m0; PartML[ml+1u] = l0; } PartAcc[ac+tid] = a0_0; PartAcc[ac+tid+64u] = a1_0; }
    { let t = b0 + 1u; let ml = (t * 384u + h * 16u + split) * 2u; let ac = (t * 384u + h * 16u + split) * 128u; if (tid == 0u) { PartML[ml] = m1; PartML[ml+1u] = l1; } PartAcc[ac+tid] = a0_1; PartAcc[ac+tid+64u] = a1_1; }
    { let t = b0 + 2u; let ml = (t * 384u + h * 16u + split) * 2u; let ac = (t * 384u + h * 16u + split) * 128u; if (tid == 0u) { PartML[ml] = m2; PartML[ml+1u] = l2; } PartAcc[ac+tid] = a0_2; PartAcc[ac+tid+64u] = a1_2; }
    { let t = b0 + 3u; let ml = (t * 384u + h * 16u + split) * 2u; let ac = (t * 384u + h * 16u + split) * 128u; if (tid == 0u) { PartML[ml] = m3; PartML[ml+1u] = l3; } PartAcc[ac+tid] = a0_3; PartAcc[ac+tid+64u] = a1_3; }
}
"""

# 17. Draft Split-K Attention Combine (no gate_attn, writes Out[8, 24, 128])
# Dispatch: (24, 8, 1) x 64 threads
D_ATTN_CB = """
@group(0) @binding(0) var<storage, read> PartML: array<f32>;
@group(0) @binding(1) var<storage, read> PartAcc: array<f32>;
@group(0) @binding(2) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let h = wg.x; let t = wg.y; let tid = lid.x;
    let base = t * 384u + h * 16u;
    var m_g: f32 = -1e30;
    for (var s: u32 = 0u; s < 16u; s++) { m_g = max(m_g, PartML[(base + s) * 2u]); }
    var l_g: f32 = 0.0; var acc0: f32 = 0.0; var acc1: f32 = 0.0;
    for (var s: u32 = 0u; s < 16u; s++) {
        let ms = PartML[(base + s) * 2u];
        let ls = PartML[(base + s) * 2u + 1u];
        if (ls > 0.0) {
            let w = exp(ms - m_g);
            l_g += ls * w;
            let ab = (base + s) * 128u;
            acc0 += PartAcc[ab + tid] * w;
            acc1 += PartAcc[ab + tid + 64u] * w;
        }
    }
    let inv_l = 1.0 / l_g;
    let ob = (t * 24u + h) * 128u;
    Out[ob + tid]       = acc0 * inv_l;
    Out[ob + tid + 64u] = acc1 * inv_l;
}
"""

DSPARK_SHADERS = [
    ("d_save_tap",  D_SAVE_TAP),
    ("d_save_ring", D_SAVE_RING),
    ("d_snap_hist", D_SNAP_HIST),
    ("d_rb_hist",   D_RB_HIST),
    ("d_fuse",      D_FUSE),
    ("d_qkn_rope",  D_QKN_ROPE),
    ("d_store_v",   D_STORE_V),
    ("d_copy_anc",  D_COPY_ANC),
    ("d_attn",      D_ATTN),
    ("d_add",       D_ADD),
    ("d_swiglu",    D_SWIGLU),
    ("d_mrnn",      D_MRNN),
    ("d_mupd",      D_MUPD),
    ("d_am_slot",   D_AM_SLOT),
    ("v_attn_sk",   V_ATTN_SK),
    ("d_attn_sk",   D_ATTN_SK),
    ("d_attn_cb",   D_ATTN_CB),
]

if __name__ == "__main__":
    out_dir = os.environ["DUMP_SPV_DIR"]
    os.makedirs(out_dir, exist_ok=True)
    ad = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    dev = ad.request_device_sync(required_features=["subgroup"])
    for name, code in DSPARK_SHADERS:
        m = dev.create_shader_module(code=code)
        dev.create_compute_pipeline(layout="auto", compute={"module": m, "entry_point": "main"})
    for i, (name, _) in enumerate(DSPARK_SHADERS):
        src_f = f"{out_dir}/shader_{i+2:02d}.spv"
        dst_f = f"{out_dir}/{name}.spv"
        shutil.copyfile(src_f, dst_f)
        raw = open(dst_f, "rb").read()
        assert raw[:4] == b"\x03\x02\x23\x07", f"Bad SPIR-V in {dst_f}"
    print(f"Compiled & verified all {len(DSPARK_SHADERS)} DSpark SPIR-V shaders into {out_dir}!")
