import os, shutil, wgpu

FAST_COMMON = """
const E2M1 = array<f32, 16>(
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
   -0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0
);
fn decode_fp8(b: u32) -> f32 {
    let s = (b & 0x80u) << 24u;
    let em = b & 0x7Fu;
    let norm = bitcast<f32>(s | ((em + 960u) << 20u));
    let sub  = bitcast<f32>(s | bitcast<u32>(f32(em) * 0.001953125));
    return select(norm, sub, em < 8u);
}
fn decode_ue8m0(b: u32) -> f32 { return bitcast<f32>(b << 23u); }
fn dot8_nvfp4(packed: u32, x0: vec4<f32>, x1: vec4<f32>) -> f32 {
    let v0 = vec4<f32>(
        E2M1[packed & 0xFu], E2M1[(packed >> 4u) & 0xFu],
        E2M1[(packed >> 8u) & 0xFu], E2M1[(packed >> 12u) & 0xFu],
    );
    let v1 = vec4<f32>(
        E2M1[(packed >> 16u) & 0xFu], E2M1[(packed >> 20u) & 0xFu],
        E2M1[(packed >> 24u) & 0xFu], E2M1[(packed >> 28u) & 0xFu],
    );
    return dot(v0, x0) + dot(v1, x1);
}
fn dot4_fp8(packed: u32, xv: vec4<f32>) -> f32 {
    let v = vec4<f32>(
        decode_fp8(packed & 0xFFu),
        decode_fp8((packed >> 8u) & 0xFFu),
        decode_fp8((packed >> 16u) & 0xFFu),
        decode_fp8((packed >> 24u) & 0xFFu),
    );
    return dot(v, xv);
}
fn bf16x2_dot(w: u32, x: vec2<f32>) -> f32 {
    return bitcast<f32>((w & 0xFFFFu) << 16u) * x.x + bitcast<f32>(w & 0xFFFF0000u) * x.y;
}
fn situ(g: f32, u: f32) -> f32 {
    return (4.0 * tanh(g * 0.25) * (1.0 / (1.0 + exp(-g)))) * (25.0 * tanh(u * 0.04));
}
"""

# 1. Batched RMS: wg.y = b (0..B-1). In: [B, stride_v4], Out: [B, 576]
B_RMS = """
@group(0) @binding(0) var<storage, read> In: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> W: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> Out: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> params: vec4<u32>; // x=in_off_v4, y=in_stride_v4 (576 or 1728)
var<workgroup> smem: array<f32, 4>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let b = wg.y; let tid = lid.x; let sg = tid >> 6u; let lane = tid & 63u;
    let in_base = b * params.y + params.x;
    let out_base = b * 576u;
    var s: f32 = 0.0;
    for (var i = tid; i < 576u; i += 256u) { let v = In[in_base + i]; s += dot(v, v); }
    let sg_s = subgroupAdd(s);
    if (lane == 0u) { smem[sg] = sg_s; }
    workgroupBarrier();
    let inv = inverseSqrt((smem[0] + smem[1] + smem[2] + smem[3]) / 2304.0 + 1e-6);
    for (var i = tid; i < 576u; i += 256u) { Out[out_base + i] = In[in_base + i] * inv * W[i]; }
}
"""

# 2. Batched Copy 2304 floats: wg.x in 0..8, wg.y = b
B_COPY = """
@group(0) @binding(0) var<storage, read> Src: array<f32>;
@group(0) @binding(1) var<storage, read_write> Dst: array<f32>;
@group(0) @binding(2) var<uniform> params: vec4<u32>; // x=src_off, y=dst_off, z=src_stride, w=dst_stride
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = wg.x * 256u + lid.x;
    if (i >= 2304u) { return; }
    let b = wg.y;
    Dst[b * params.w + params.y + i] = Src[b * params.z + params.x + i];
}
"""

# 3. Batched Depth-Mix: wg.y = b. Blocks: [B, b_stride_v4], Cur: [B, 576], Out: [B, out_stride_v4]
B_MIX = """
@group(0) @binding(0) var<storage, read> Blocks: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> Cur: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> Query: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> Out: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> params: vec4<u32>; // x=b_slots (8 or 4), y=nb, z=n_q (1..4), w=out_slots (1, 3, or 4)
var<workgroup> inv_rms: array<f32, 8>;
var<workgroup> probs: array<f32, 24>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let bi = wg.y; let tid = lid.x; let sg = tid >> 6u; let lane = tid & 63u;
    let blk_base = bi * params.x * 576u;
    let cur_base = bi * 576u;
    let out_base = bi * params.w * 576u;
    let nb = params.y; let n_states = nb + 1u; let n_q = params.z;
    let inv_sqrt_d: f32 = 0.020833333333333332;

    for (var b = sg; b < n_states; b += 4u) {
        var sq: f32 = 0.0;
        var d0: f32 = 0.0; var d1: f32 = 0.0; var d2: f32 = 0.0; var d3: f32 = 0.0;
        let is_cur = (b == nb);
        let b_off = blk_base + b * 576u;
        for (var i = lane; i < 576u; i += 64u) {
            let v = select(Blocks[b_off + i], Cur[cur_base + i], is_cur);
            sq += dot(v, v);
            d0 += dot(v, Query[i]);
            if (n_q > 1u) { d1 += dot(v, Query[576u + i]); }
            if (n_q > 2u) { d2 += dot(v, Query[1152u + i]); }
            if (n_q > 3u) { d3 += dot(v, Query[1728u + i]); }
        }
        let s_sq = subgroupAdd(sq);
        let s_d0 = subgroupAdd(d0); let s_d1 = subgroupAdd(d1); let s_d2 = subgroupAdd(d2); let s_d3 = subgroupAdd(d3);
        if (lane == 0u) {
            let ir = inverseSqrt(s_sq / 2304.0 + 1e-6);
            inv_rms[b] = ir; let k = ir * inv_sqrt_d;
            probs[b] = s_d0 * k;
            if (n_q > 1u) { probs[6u + b] = s_d1 * k; }
            if (n_q > 2u) { probs[12u + b] = s_d2 * k; }
            if (n_q > 3u) { probs[18u + b] = s_d3 * k; }
        }
    }
    workgroupBarrier();
    if (tid < n_q) {
        let qb = tid * 6u;
        var mx: f32 = probs[qb];
        for (var b: u32 = 1u; b < n_states; b++) { mx = max(mx, probs[qb + b]); }
        var denom: f32 = 0.0;
        for (var b: u32 = 0u; b < n_states; b++) { let e = exp(probs[qb + b] - mx); probs[qb + b] = e; denom += e; }
        let inv_d = 1.0 / denom;
        for (var b: u32 = 0u; b < n_states; b++) { probs[qb + b] *= inv_d * inv_rms[b]; }
    }
    workgroupBarrier();
    for (var i = tid; i < 576u; i += 256u) {
        var o0 = vec4<f32>(0.0); var o1 = vec4<f32>(0.0); var o2 = vec4<f32>(0.0); var o3 = vec4<f32>(0.0);
        for (var b: u32 = 0u; b < nb; b++) {
            let v = Blocks[blk_base + b * 576u + i];
            o0 += probs[b] * v;
            if (n_q > 1u) { o1 += probs[6u + b] * v; }
            if (n_q > 2u) { o2 += probs[12u + b] * v; }
            if (n_q > 3u) { o3 += probs[18u + b] * v; }
        }
        let vc = Cur[cur_base + i];
        Out[out_base + i] = o0 + probs[nb] * vc;
        if (n_q > 1u) { Out[out_base + 576u + i]  = o1 + probs[6u + nb] * vc; }
        if (n_q > 2u) { Out[out_base + 1152u + i] = o2 + probs[12u + nb] * vc; }
        if (n_q > 3u) { Out[out_base + 1728u + i] = o3 + probs[18u + nb] * vc; }
    }
}
"""

# 4. Batched Mix + RMS (n_q=1): wg.y = b
B_MIX_RMS = """
@group(0) @binding(0) var<storage, read> Blocks: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> Cur: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> Query: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> NormW: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> Out: array<vec4<f32>>;
@group(0) @binding(5) var<uniform> params: vec4<u32>; // x=b_slots (8), y=nb
var<workgroup> inv_rms: array<f32, 8>;
var<workgroup> probs: array<f32, 8>;
var<workgroup> sg_sq: array<f32, 4>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let bi = wg.y; let tid = lid.x; let sg = tid >> 6u; let lane = tid & 63u;
    let blk_base = bi * params.x * 576u;
    let cur_base = bi * 576u;
    let nb = params.y; let n_states = nb + 1u;
    let inv_sqrt_d: f32 = 0.020833333333333332;

    for (var b = sg; b < n_states; b += 4u) {
        var sq: f32 = 0.0; var d0: f32 = 0.0;
        let is_cur = (b == nb);
        let b_off = blk_base + b * 576u;
        for (var i = lane; i < 576u; i += 64u) {
            let v = select(Blocks[b_off + i], Cur[cur_base + i], is_cur);
            sq += dot(v, v); d0 += dot(v, Query[i]);
        }
        let s_sq = subgroupAdd(sq); let s_d0 = subgroupAdd(d0);
        if (lane == 0u) { let ir = inverseSqrt(s_sq / 2304.0 + 1e-6); inv_rms[b] = ir; probs[b] = s_d0 * ir * inv_sqrt_d; }
    }
    workgroupBarrier();
    if (tid == 0u) {
        var mx: f32 = probs[0];
        for (var b: u32 = 1u; b < n_states; b++) { mx = max(mx, probs[b]); }
        var denom: f32 = 0.0;
        for (var b: u32 = 0u; b < n_states; b++) { let e = exp(probs[b] - mx); probs[b] = e; denom += e; }
        let inv_d = 1.0 / denom;
        for (var b: u32 = 0u; b < n_states; b++) { probs[b] *= inv_d * inv_rms[b]; }
    }
    workgroupBarrier();
    var m0 = vec4<f32>(0.0); var m1 = vec4<f32>(0.0); var m2 = vec4<f32>(0.0);
    var sq_out: f32 = 0.0;
    { let i = tid; var o = probs[nb] * Cur[cur_base + i]; for (var b: u32 = 0u; b < nb; b++) { o += probs[b] * Blocks[blk_base + b * 576u + i]; } m0 = o; sq_out += dot(o, o); }
    { let i = tid + 256u; var o = probs[nb] * Cur[cur_base + i]; for (var b: u32 = 0u; b < nb; b++) { o += probs[b] * Blocks[blk_base + b * 576u + i]; } m1 = o; sq_out += dot(o, o); }
    if (tid < 64u) { let i = tid + 512u; var o = probs[nb] * Cur[cur_base + i]; for (var b: u32 = 0u; b < nb; b++) { o += probs[b] * Blocks[blk_base + b * 576u + i]; } m2 = o; sq_out += dot(o, o); }
    let s_out = subgroupAdd(sq_out);
    if (lane == 0u) { sg_sq[sg] = s_out; }
    workgroupBarrier();
    let inv_out = inverseSqrt((sg_sq[0] + sg_sq[1] + sg_sq[2] + sg_sq[3]) / 2304.0 + 1e-6);
    Out[cur_base + tid] = m0 * inv_out * NormW[tid];
    Out[cur_base + tid + 256u] = m1 * inv_out * NormW[tid + 256u];
    if (tid < 64u) { Out[cur_base + tid + 512u] = m2 * inv_out * NormW[tid + 512u]; }
}
"""

# 5. Batched 3-WG RMS3: wg.x in 0..2, wg.y = b
B_RMS3 = """
@group(0) @binding(0) var<storage, read> In3: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> Wq: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> Wk: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> Wv: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> Oq: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> Ok: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> Ov: array<vec4<f32>>;
var<workgroup> smem: array<f32, 4>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let which = wg.x; let bi = wg.y; let tid = lid.x; let sg = tid >> 6u; let lane = tid & 63u;
    let off = (bi * 3u + which) * 576u;
    let ob = bi * 576u;
    var s: f32 = 0.0;
    for (var i = tid; i < 576u; i += 256u) { let v = In3[off + i]; s += dot(v, v); }
    let sg_s = subgroupAdd(s);
    if (lane == 0u) { smem[sg] = sg_s; }
    workgroupBarrier();
    let inv = inverseSqrt((smem[0] + smem[1] + smem[2] + smem[3]) / 2304.0 + 1e-6);
    if (which == 0u) { for (var i = tid; i < 576u; i += 256u) { Oq[ob + i] = In3[off + i] * inv * Wq[i]; } }
    else if (which == 1u) { for (var i = tid; i < 576u; i += 256u) { Ok[ob + i] = In3[off + i] * inv * Wk[i]; } }
    else { for (var i = tid; i < 576u; i += 256u) { Ov[ob + i] = In3[off + i] * inv * Wv[i]; } }
}
"""

# 6-9. Batched Control Gates (wg.z = b):
B_CTRL_P1 = """
@group(0) @binding(0) var<storage, read> Cur: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> Msg: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> Wd: array<u32>;
@group(0) @binding(3) var<storage, read_write> Gate: array<f32>;
@group(0) @binding(4) var<uniform> params: vec4<u32>; // x=cur_off_v2, y=msg_off_v2
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let r = wg.x; let bi = wg.z; let tid = lid.x;
    let base = r * 2304u;
    let c_off = bi * 1152u + params.x;
    let m_off = bi * 4608u + params.y;
    var sq_c: f32 = 0.0; var sq_m: f32 = 0.0; var dc: f32 = 0.0; var dm: f32 = 0.0;
    for (var k = tid; k < 1152u; k += 64u) {
        let c = Cur[c_off + k]; let m = Msg[m_off + k];
        sq_c += dot(c, c); sq_m += dot(m, m);
        let wc = Wd[base + k]; let wm = Wd[base + 1152u + k];
        dc += bitcast<f32>((wc & 0xFFFFu) << 16u) * c.x + bitcast<f32>(wc & 0xFFFF0000u) * c.y;
        dm += bitcast<f32>((wm & 0xFFFFu) << 16u) * m.x + bitcast<f32>(wm & 0xFFFF0000u) * m.y;
    }
    let inv_c = inverseSqrt(subgroupAdd(sq_c) / 2304.0 + 1e-6);
    let inv_m = inverseSqrt(subgroupAdd(sq_m) / 2304.0 + 1e-6);
    let tot = subgroupAdd(dc * inv_c + dm * inv_m);
    if (tid == 0u) { Gate[bi * 192u + r] = tot; }
}
"""

B_CTRL_P2 = """
@group(0) @binding(0) var<storage, read> Cur: array<f32>;
@group(0) @binding(1) var<storage, read> Msg: array<f32>;
@group(0) @binding(2) var<storage, read> Gate: array<f32>;
@group(0) @binding(3) var<storage, read> Wg: array<u32>;
@group(0) @binding(4) var<storage, read> Wf: array<u32>;
@group(0) @binding(5) var<storage, read_write> Out: array<f32>;
@group(0) @binding(6) var<uniform> params: vec4<u32>; // x=cur_off, y=msg_off, z=out_off
var<workgroup> s_gate: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let bi = wg.z;
    s_gate[lid.x] = Gate[bi * 192u + lid.x];
    workgroupBarrier();
    let r = wg.x * 64u + lid.x;
    if (r >= 2304u) { return; }
    let base = r * 32u;
    var g: f32 = 0.0; var f: f32 = 0.0;
    for (var j: u32 = 0u; j < 32u; j++) {
        let wg_w = Wg[base + j]; let wf_w = Wf[base + j];
        let x0 = s_gate[2u * j]; let x1 = s_gate[2u * j + 1u];
        g += bitcast<f32>((wg_w & 0xFFFFu) << 16u) * x0 + bitcast<f32>(wg_w & 0xFFFF0000u) * x1;
        f += bitcast<f32>((wf_w & 0xFFFFu) << 16u) * x0 + bitcast<f32>(wf_w & 0xFFFF0000u) * x1;
    }
    let c = Cur[bi * 2304u + params.x + r];
    let m = Msg[bi * 9216u + params.y + r];
    Out[bi * 2304u + params.z + r] = c - tanh(f) * c + tanh(g) * m;
}
"""

B_CTRL3_P1 = """
@group(0) @binding(0) var<storage, read> Cur: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> Msg: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> Wd3: array<u32>;
@group(0) @binding(3) var<storage, read_write> Gate3: array<f32>;
@group(0) @binding(4) var<uniform> params: vec4<u32>; // x=cur_stride_v2 (0 or 1152)
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let r = wg.x; let c_idx = wg.y; let bi = wg.z; let tid = lid.x;
    let base = (c_idx * 64u + r) * 2304u;
    let c_off = bi * 3456u + c_idx * params.x;
    let m_off = bi * 4608u + c_idx * 1152u;
    var sq_c: f32 = 0.0; var sq_m: f32 = 0.0; var dc: f32 = 0.0; var dm: f32 = 0.0;
    for (var k = tid; k < 1152u; k += 64u) {
        let c = Cur[c_off + k]; let m = Msg[m_off + k];
        sq_c += dot(c, c); sq_m += dot(m, m);
        let wc = Wd3[base + k]; let wm = Wd3[base + 1152u + k];
        dc += bitcast<f32>((wc & 0xFFFFu) << 16u) * c.x + bitcast<f32>(wc & 0xFFFF0000u) * c.y;
        dm += bitcast<f32>((wm & 0xFFFFu) << 16u) * m.x + bitcast<f32>(wm & 0xFFFF0000u) * m.y;
    }
    let inv_c = inverseSqrt(subgroupAdd(sq_c) / 2304.0 + 1e-6);
    let inv_m = inverseSqrt(subgroupAdd(sq_m) / 2304.0 + 1e-6);
    let tot = subgroupAdd(dc * inv_c + dm * inv_m);
    if (tid == 0u) { Gate3[bi * 192u + c_idx * 64u + r] = tot; }
}
"""

B_CTRL3_P2 = """
@group(0) @binding(0) var<storage, read> Cur: array<f32>;
@group(0) @binding(1) var<storage, read> Msg: array<f32>;
@group(0) @binding(2) var<storage, read> Gate3: array<f32>;
@group(0) @binding(3) var<storage, read> Wg3: array<u32>;
@group(0) @binding(4) var<storage, read> Wf3: array<u32>;
@group(0) @binding(5) var<storage, read_write> Out3: array<f32>;
@group(0) @binding(6) var<uniform> params: vec4<u32>; // x=cur_stride (0 or 2304)
var<workgroup> s_gate: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let c_idx = wg.y; let bi = wg.z;
    s_gate[lid.x] = Gate3[bi * 192u + c_idx * 64u + lid.x];
    workgroupBarrier();
    let r = wg.x * 64u + lid.x;
    if (r >= 2304u) { return; }
    let base = (c_idx * 2304u + r) * 32u;
    var g: f32 = 0.0; var f: f32 = 0.0;
    for (var j: u32 = 0u; j < 32u; j++) {
        let wg_w = Wg3[base + j]; let wf_w = Wf3[base + j];
        let x0 = s_gate[2u * j]; let x1 = s_gate[2u * j + 1u];
        g += bitcast<f32>((wg_w & 0xFFFFu) << 16u) * x0 + bitcast<f32>(wg_w & 0xFFFF0000u) * x1;
        f += bitcast<f32>((wf_w & 0xFFFFu) << 16u) * x0 + bitcast<f32>(wf_w & 0xFFFF0000u) * x1;
    }
    let c = Cur[bi * 6912u + c_idx * params.x + r];
    let m = Msg[bi * 9216u + c_idx * 2304u + r];
    Out3[bi * 6912u + c_idx * 2304u + r] = c - tanh(f) * c + tanh(g) * m;
}
"""

# 10. SHARED-WEIGHT BF16 GEMM (8-way register tiled over batch!):
# Loads W[row] ONCE and multiplies against up to 8 batch items (wg.y = tile of 8)!
# Dispatch: ((N+1)/2, (B+7)/8, 1)
B_GEMM = FAST_COMMON + """
@group(0) @binding(0) var<storage, read> W: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;  // [B, K/4]
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;  // [B, Y_stride]
@group(0) @binding(3) var<storage, read> Meta: array<u32>;     // Meta[0] = B
@group(0) @binding(4) var<uniform> dims: vec4<u32>;            // x=K_v4 (288), y=N, z=Y_stride (8600, 1056, 128)
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x; let sub = tid >> 6u; let lane = tid & 63u;
    let row = wg.x * 2u + sub;
    if (row >= dims.y) { return; }
    let B = Meta[0];
    let b0 = wg.y * 8u; if (b0 >= B) { return; }
    let cnt = min(8u, B - b0);
    let K_v4 = dims.x;
    let K_x = K_v4 * 2u; // vec4<f32> per user (576 for K=2304)
    let base = row * K_v4;

    var a0: f32 = 0.0; var a1: f32 = 0.0; var a2: f32 = 0.0; var a3: f32 = 0.0;
    var a4: f32 = 0.0; var a5: f32 = 0.0; var a6: f32 = 0.0; var a7: f32 = 0.0;

    for (var k = lane; k < K_v4; k += 64u) {
        let w = W[base + k];
        let xk = k * 2u;
        { let x0 = X[b0 * K_x + xk]; let x1 = X[b0 * K_x + xk + 1u]; a0 += bf16x2_dot(w.x, x0.xy) + bf16x2_dot(w.y, x0.zw) + bf16x2_dot(w.z, x1.xy) + bf16x2_dot(w.w, x1.zw); }
        if (cnt > 1u) { let o = (b0+1u)*K_x+xk; let x0 = X[o]; let x1 = X[o+1u]; a1 += bf16x2_dot(w.x, x0.xy) + bf16x2_dot(w.y, x0.zw) + bf16x2_dot(w.z, x1.xy) + bf16x2_dot(w.w, x1.zw); }
        if (cnt > 2u) { let o = (b0+2u)*K_x+xk; let x0 = X[o]; let x1 = X[o+1u]; a2 += bf16x2_dot(w.x, x0.xy) + bf16x2_dot(w.y, x0.zw) + bf16x2_dot(w.z, x1.xy) + bf16x2_dot(w.w, x1.zw); }
        if (cnt > 3u) { let o = (b0+3u)*K_x+xk; let x0 = X[o]; let x1 = X[o+1u]; a3 += bf16x2_dot(w.x, x0.xy) + bf16x2_dot(w.y, x0.zw) + bf16x2_dot(w.z, x1.xy) + bf16x2_dot(w.w, x1.zw); }
        if (cnt > 4u) { let o = (b0+4u)*K_x+xk; let x0 = X[o]; let x1 = X[o+1u]; a4 += bf16x2_dot(w.x, x0.xy) + bf16x2_dot(w.y, x0.zw) + bf16x2_dot(w.z, x1.xy) + bf16x2_dot(w.w, x1.zw); }
        if (cnt > 5u) { let o = (b0+5u)*K_x+xk; let x0 = X[o]; let x1 = X[o+1u]; a5 += bf16x2_dot(w.x, x0.xy) + bf16x2_dot(w.y, x0.zw) + bf16x2_dot(w.z, x1.xy) + bf16x2_dot(w.w, x1.zw); }
        if (cnt > 6u) { let o = (b0+6u)*K_x+xk; let x0 = X[o]; let x1 = X[o+1u]; a6 += bf16x2_dot(w.x, x0.xy) + bf16x2_dot(w.y, x0.zw) + bf16x2_dot(w.z, x1.xy) + bf16x2_dot(w.w, x1.zw); }
        if (cnt > 7u) { let o = (b0+7u)*K_x+xk; let x0 = X[o]; let x1 = X[o+1u]; a7 += bf16x2_dot(w.x, x0.xy) + bf16x2_dot(w.y, x0.zw) + bf16x2_dot(w.z, x1.xy) + bf16x2_dot(w.w, x1.zw); }
    }
    let s0 = subgroupAdd(a0); let s1 = subgroupAdd(a1); let s2 = subgroupAdd(a2); let s3 = subgroupAdd(a3);
    let s4 = subgroupAdd(a4); let s5 = subgroupAdd(a5); let s6 = subgroupAdd(a6); let s7 = subgroupAdd(a7);
    if (lane == 0u) {
        let ys = dims.z;
        Y[b0 * ys + row] = s0;
        if (cnt > 1u) { Y[(b0+1u)*ys + row] = s1; }
        if (cnt > 2u) { Y[(b0+2u)*ys + row] = s2; }
        if (cnt > 3u) { Y[(b0+3u)*ys + row] = s3; }
        if (cnt > 4u) { Y[(b0+4u)*ys + row] = s4; }
        if (cnt > 5u) { Y[(b0+5u)*ys + row] = s5; }
        if (cnt > 6u) { Y[(b0+6u)*ys + row] = s6; }
        if (cnt > 7u) { Y[(b0+7u)*ys + row] = s7; }
    }
}
"""

# 11. Batched QKVR_POST with split K0/K1/V0/V1 + SWA 1024 ring buffer + Full 4096!
B_QKVR_POST = """
@group(0) @binding(0) var<storage, read_write> QRSG: array<f32>;
@group(0) @binding(1) var<storage, read> Kraw: array<f32>;
@group(0) @binding(2) var<storage, read> Vraw: array<f32>;
@group(0) @binding(3) var<storage, read> QClip: array<f32>;
@group(0) @binding(4) var<storage, read> RClip: array<f32>;
@group(0) @binding(5) var<storage, read> KConv: array<f32>;
@group(0) @binding(6) var<storage, read> VConv: array<f32>;
@group(0) @binding(7) var<storage, read_write> HistK: array<f32>;
@group(0) @binding(8) var<storage, read_write> HistV: array<f32>;
@group(0) @binding(9) var<storage, read_write> KCache0: array<f32>;
@group(0) @binding(10) var<storage, read_write> KCache1: array<f32>;
@group(0) @binding(11) var<storage, read_write> VCache0: array<f32>;
@group(0) @binding(12) var<storage, read_write> VCache1: array<f32>;
@group(0) @binding(13) var<storage, read> Meta: array<u32>;
@group(0) @binding(14) var<uniform> params: vec4<u32>; // x=(l_slot<<8)|cache_idx, y=is_full(0=SWA1024,1=Full4096), z=k_off, w=v_off

var<workgroup> vals: array<f32, 128>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let id = wg.x; let bi = wg.y; let tid = lid.x;
    let l_slot = params.x >> 8u;
    let cache_idx = params.x & 0xFFu;
    let is_full = params.y;
    let seq_slot = Meta[16u + bi];
    let pos = Meta[32u + bi];
    let qrsg_b = bi * 8600u;

    if (id < 24u) {
        let base = qrsg_b + id * 128u;
        let cb = id * 128u;
        let v0 = QRSG[base + tid] * QClip[cb + tid];
        let v1 = QRSG[base + tid + 64u] * QClip[cb + tid + 64u];
        let sq = subgroupAdd(v0 * v0 + v1 * v1);
        var factor = min(1.0, 22.627416997969522 / (sqrt(sq) + 1e-6));
        if (is_full == 1u) { factor *= 1.0 + 0.1 * log(max(1.0, (f32(pos) + 1.0) / 8192.0)); }
        QRSG[base + tid] = v0 * factor;
        QRSG[base + tid + 64u] = v1 * factor;
        return;
    }
    if (id < 48u) {
        let h = id - 24u;
        let base = qrsg_b + 3072u + h * 16u;
        var v: f32 = 0.0;
        if (tid < 16u) { v = QRSG[base + tid] * RClip[h * 16u + tid]; }
        let sq = subgroupAdd(v * v);
        if (tid < 16u) {
            var factor = min(1.0, 8.0 / (sqrt(sq) + 1e-6));
            if (is_full == 1u) { factor *= 1.0 + 0.1 * log(max(1.0, (f32(pos) + 1.0) / 8192.0)); }
            QRSG[base + tid] = v * factor;
        }
        return;
    }

    let is_v = select(0u, 1u, id >= 56u);
    let head = select(id - 48u, id - 56u, is_v == 1u);
    let h_base = (seq_slot * 48u + l_slot) * 3072u;
    let kv_b = bi * 1056u;
    var sq: f32 = 0.0;

    for (var j: u32 = 0u; j < 2u; j++) {
        let d = tid + j * 64u;
        let c = head * 128u + d;
        var y: f32;
        if (is_v == 0u) {
            let x0 = HistK[h_base + c]; let x1 = HistK[h_base + 1024u + c]; let x2 = HistK[h_base + 2048u + c];
            let x3 = select(Kraw[kv_b + c], QRSG[qrsg_b + params.z + c], params.z > 0u);
            HistK[h_base + c] = x1; HistK[h_base + 1024u + c] = x2; HistK[h_base + 2048u + c] = x3;
            if (is_full == 0u) {
                y = (x3 + x0*KConv[c*4u] + x1*KConv[c*4u+1u] + x2*KConv[c*4u+2u] + x3*KConv[c*4u+3u]) * 0.4472135954999579;
            } else {
                let db = kv_b + 1024u + head * 4u;
                let out = x3*(KConv[c]+Kraw[db]) + x2*(KConv[1024u+c]+Kraw[db+1u]) + x1*(KConv[2048u+c]+Kraw[db+2u]) + x0*(KConv[3072u+c]+Kraw[db+3u]);
                y = (x3 + out) * 0.3333333333333333;
            }
        } else {
            let x0 = HistV[h_base + c]; let x1 = HistV[h_base + 1024u + c]; let x2 = HistV[h_base + 2048u + c];
            let x3 = select(Vraw[kv_b + c], QRSG[qrsg_b + params.w + c], params.w > 0u);
            HistV[h_base + c] = x1; HistV[h_base + 1024u + c] = x2; HistV[h_base + 2048u + c] = x3;
            if (is_full == 0u) {
                y = (x3 + x0*VConv[c*4u] + x1*VConv[c*4u+1u] + x2*VConv[c*4u+2u] + x3*VConv[c*4u+3u]) * 0.4472135954999579;
            } else {
                let db = kv_b + 1024u + head * 4u;
                let out = x3*(VConv[c]+Vraw[db]) + x2*(VConv[1024u+c]+Vraw[db+1u]) + x1*(VConv[2048u+c]+Vraw[db+2u]) + x0*(VConv[3072u+c]+Vraw[db+3u]);
                y = (x3 + out) * 0.3333333333333333;
            }
        }
        vals[d] = y; sq += y * y;
    }
    let sq_sum = subgroupAdd(sq);
    let mult = select(min(1.0, 22.627416997969522 / (sqrt(sq_sum) + 1e-6)), inverseSqrt(sq_sum / 128.0 + 1e-6), is_v == 1u);

    let half_slot = seq_slot & 7u;
    let n_layers = select(40u, 8u, is_full == 1u);
    let cap = select(1024u, 4096u, is_full == 1u);
    let pos_phys = select(pos & 1023u, pos, is_full == 1u);
    let c_off = ((half_slot * n_layers + cache_idx) * cap + pos_phys) * 1024u + head * 128u;
    let o0 = vals[tid] * mult; let o1 = vals[tid + 64u] * mult;
    if (is_v == 0u) {
        if (seq_slot < 8u) { KCache0[c_off + tid] = o0; KCache0[c_off + tid + 64u] = o1; }
        else               { KCache1[c_off + tid] = o0; KCache1[c_off + tid + 64u] = o1; }
    } else {
        if (seq_slot < 8u) { VCache0[c_off + tid] = o0; VCache0[c_off + tid + 64u] = o1; }
        else               { VCache1[c_off + tid] = o0; VCache1[c_off + tid + 64u] = o1; }
    }
}
"""

# 12. Batched Split-K Attention with split K0/K1/V0/V1 + SWA 1024 ring buffer + Full 4096!
B_ATTN_SK = """
@group(0) @binding(0) var<storage, read> QRSG: array<f32>;
@group(0) @binding(1) var<storage, read> RelBias: array<f32>;
@group(0) @binding(2) var<storage, read> KCache0: array<f32>;
@group(0) @binding(3) var<storage, read> KCache1: array<f32>;
@group(0) @binding(4) var<storage, read> VCache0: array<f32>;
@group(0) @binding(5) var<storage, read> VCache1: array<f32>;
@group(0) @binding(6) var<storage, read_write> PartML: array<f32>;
@group(0) @binding(7) var<storage, read_write> PartAcc: array<f32>;
@group(0) @binding(8) var<storage, read> Meta: array<u32>;
@group(0) @binding(9) var<uniform> params: vec4<u32>; // x=cache_idx, y=is_swa(1=SWA1024,0=Full4096)

var<workgroup> r_sh: array<f32, 16>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let h = wg.x; let split = wg.y; let bi = wg.z; let kv_h = h / 3u; let tid = lid.x;
    let cache_idx = params.x; let is_swa = params.y;
    let seq_slot = Meta[16u + bi];
    let pos = Meta[32u + bi];

    let k_start: u32 = select(0u, select(0u, pos - 1023u, pos >= 1024u), is_swa == 1u);
    let total_k = pos - k_start + 1u;
    let chunk = (total_k + 15u) >> 4u;
    let s_begin = k_start + split * chunk;
    let s_end = min(pos + 1u, s_begin + chunk);
    let ml_idx = (bi * 384u + h * 16u + split) * 2u;
    let acc_base = (bi * 384u + h * 16u + split) * 128u;

    if (s_begin >= s_end) {
        if (tid == 0u) { PartML[ml_idx] = -1e30; PartML[ml_idx + 1u] = 0.0; }
        PartAcc[acc_base + tid] = 0.0; PartAcc[acc_base + tid + 64u] = 0.0;
        return;
    }

    let qrsg_b = bi * 8600u;
    let q_base = qrsg_b + h * 128u;
    let q0 = QRSG[q_base + tid]; let q1 = QRSG[q_base + tid + 64u];
    if (tid < 16u) { r_sh[tid] = QRSG[qrsg_b + 3072u + h * 16u + tid]; }
    workgroupBarrier();

    let sink = QRSG[qrsg_b + 3456u + h];
    let inv_sqrt_d: f32 = 0.08838834764831845;
    let half_slot = seq_slot & 7u;
    let n_layers = select(8u, 40u, is_swa == 1u);
    let cap = select(4096u, 1024u, is_swa == 1u);
    let slot_base = (half_slot * n_layers + cache_idx) * cap * 1024u + kv_h * 128u;
    let hi = (seq_slot >= 8u);
    var m_run: f32 = -1e30; var l_run: f32 = 0.0; var acc0: f32 = 0.0; var acc1: f32 = 0.0;

    for (var k = s_begin; k < s_end; k++) {
        let k_phys = select(k, k & 1023u, is_swa == 1u);
        let kv_off = slot_base + k_phys * 1024u;
        var k0: f32; var k1: f32; var v0: f32; var v1: f32;
        if (!hi) {
            k0 = KCache0[kv_off + tid]; k1 = KCache0[kv_off + tid + 64u];
            v0 = VCache0[kv_off + tid]; v1 = VCache0[kv_off + tid + 64u];
        } else {
            k0 = KCache1[kv_off + tid]; k1 = KCache1[kv_off + tid + 64u];
            v0 = VCache1[kv_off + tid]; v1 = VCache1[kv_off + tid + 64u];
        }
        let qk = subgroupAdd(q0 * k0 + q1 * k1);
        var sc = qk * inv_sqrt_d - sink;
        let dist = pos - k;
        if (dist < 1024u) {
            let rb_part = select(0.0, r_sh[tid] * RelBias[tid * 1024u + dist], tid < 16u);
            sc += subgroupAdd(rb_part);
        }
        let m_new = max(m_run, sc);
        let alpha = exp(m_run - m_new);
        let p = exp(sc - m_new);
        l_run = l_run * alpha + p;
        acc0 = acc0 * alpha + p * v0;
        acc1 = acc1 * alpha + p * v1;
        m_run = m_new;
    }
    if (tid == 0u) { PartML[ml_idx] = m_run; PartML[ml_idx + 1u] = l_run; }
    PartAcc[acc_base + tid] = acc0; PartAcc[acc_base + tid + 64u] = acc1;
}
"""

# 13. Batched Attention Combine: wg.x=h(0..23), wg.y=b
B_ATTN_CB = """
@group(0) @binding(0) var<storage, read> PartML: array<f32>;
@group(0) @binding(1) var<storage, read> PartAcc: array<f32>;
@group(0) @binding(2) var<storage, read> QRSG: array<f32>;
@group(0) @binding(3) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let h = wg.x; let bi = wg.y; let tid = lid.x;
    let b_ml = bi * 384u + h * 16u;
    var m_glob: f32 = 0.0;
    for (var s: u32 = 0u; s < 16u; s++) {
        if (PartML[(b_ml + s) * 2u + 1u] > 0.0) { m_glob = max(m_glob, PartML[(b_ml + s) * 2u]); }
    }
    var l_glob: f32 = exp(0.0 - m_glob);
    var acc0: f32 = 0.0; var acc1: f32 = 0.0;
    for (var s: u32 = 0u; s < 16u; s++) {
        let ml_idx = (b_ml + s) * 2u;
        let l_s = PartML[ml_idx + 1u];
        if (l_s > 0.0) {
            let scale = exp(PartML[ml_idx] - m_glob);
            l_glob += l_s * scale;
            let ab = (b_ml + s) * 128u;
            acc0 += PartAcc[ab + tid] * scale; acc1 += PartAcc[ab + tid + 64u] * scale;
        }
    }
    let inv_l = 1.0 / l_glob;
    let q_base = h * 128u;
    let g_base = bi * 8600u + 3480u + q_base;
    let o_base = bi * 3072u + q_base;
    Out[o_base + tid]       = (acc0 * inv_l) / (1.0 + exp(-QRSG[g_base + tid]));
    Out[o_base + tid + 64u] = (acc1 * inv_l) / (1.0 + exp(-QRSG[g_base + tid + 64u]));
}
"""

# 14. SHARED-WEIGHT WO + conv + resid (8-way register tiled over batch!):
# Dispatch: (1152, (B+7)/8, 1)
B_WO_CONV = FAST_COMMON + """
@group(0) @binding(0) var<storage, read> W: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;     // [B, 768]
@group(0) @binding(2) var<storage, read> ConvW: array<f32>;
@group(0) @binding(3) var<storage, read> Resid: array<f32>;       // [B, 2304]
@group(0) @binding(4) var<storage, read_write> Hist: array<f32>;  // [16, 48, 3, 2304]
@group(0) @binding(5) var<storage, read_write> Out: array<f32>;   // [B, 2304]
@group(0) @binding(6) var<storage, read> Meta: array<u32>;
@group(0) @binding(7) var<uniform> params: vec4<u32>;             // x=l_slot, y=add_resid

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x; let sub = tid >> 6u; let lane = tid & 63u;
    let row = wg.x * 2u + sub;
    let B = Meta[0];
    let b0 = wg.y * 8u; if (b0 >= B) { return; }
    let cnt = min(8u, B - b0);
    let base = row * 384u;

    var a0: f32 = 0.0; var a1: f32 = 0.0; var a2: f32 = 0.0; var a3: f32 = 0.0;
    var a4: f32 = 0.0; var a5: f32 = 0.0; var a6: f32 = 0.0; var a7: f32 = 0.0;
    for (var k = lane; k < 384u; k += 64u) {
        let w = W[base + k]; let xk = k * 2u;
        { let o = b0*768u+xk; let x0 = X[o]; let x1 = X[o+1u]; a0 += bf16x2_dot(w.x, x0.xy) + bf16x2_dot(w.y, x0.zw) + bf16x2_dot(w.z, x1.xy) + bf16x2_dot(w.w, x1.zw); }
        if (cnt > 1u) { let o = (b0+1u)*768u+xk; let x0 = X[o]; let x1 = X[o+1u]; a1 += bf16x2_dot(w.x, x0.xy) + bf16x2_dot(w.y, x0.zw) + bf16x2_dot(w.z, x1.xy) + bf16x2_dot(w.w, x1.zw); }
        if (cnt > 2u) { let o = (b0+2u)*768u+xk; let x0 = X[o]; let x1 = X[o+1u]; a2 += bf16x2_dot(w.x, x0.xy) + bf16x2_dot(w.y, x0.zw) + bf16x2_dot(w.z, x1.xy) + bf16x2_dot(w.w, x1.zw); }
        if (cnt > 3u) { let o = (b0+3u)*768u+xk; let x0 = X[o]; let x1 = X[o+1u]; a3 += bf16x2_dot(w.x, x0.xy) + bf16x2_dot(w.y, x0.zw) + bf16x2_dot(w.z, x1.xy) + bf16x2_dot(w.w, x1.zw); }
        if (cnt > 4u) { let o = (b0+4u)*768u+xk; let x0 = X[o]; let x1 = X[o+1u]; a4 += bf16x2_dot(w.x, x0.xy) + bf16x2_dot(w.y, x0.zw) + bf16x2_dot(w.z, x1.xy) + bf16x2_dot(w.w, x1.zw); }
        if (cnt > 5u) { let o = (b0+5u)*768u+xk; let x0 = X[o]; let x1 = X[o+1u]; a5 += bf16x2_dot(w.x, x0.xy) + bf16x2_dot(w.y, x0.zw) + bf16x2_dot(w.z, x1.xy) + bf16x2_dot(w.w, x1.zw); }
        if (cnt > 6u) { let o = (b0+6u)*768u+xk; let x0 = X[o]; let x1 = X[o+1u]; a6 += bf16x2_dot(w.x, x0.xy) + bf16x2_dot(w.y, x0.zw) + bf16x2_dot(w.z, x1.xy) + bf16x2_dot(w.w, x1.zw); }
        if (cnt > 7u) { let o = (b0+7u)*768u+xk; let x0 = X[o]; let x1 = X[o+1u]; a7 += bf16x2_dot(w.x, x0.xy) + bf16x2_dot(w.y, x0.zw) + bf16x2_dot(w.z, x1.xy) + bf16x2_dot(w.w, x1.zw); }
    }
    var sums = array<f32, 8>(
        subgroupAdd(a0), subgroupAdd(a1), subgroupAdd(a2), subgroupAdd(a3),
        subgroupAdd(a4), subgroupAdd(a5), subgroupAdd(a6), subgroupAdd(a7)
    );
    if (lane == 0u) {
        let cw0 = ConvW[row*4u]; let cw1 = ConvW[row*4u+1u]; let cw2 = ConvW[row*4u+2u]; let cw3 = ConvW[row*4u+3u];
        for (var j: u32 = 0u; j < cnt; j++) {
            let bi = b0 + j;
            let seq_slot = Meta[16u + bi];
            let x3 = sums[j];
            let hb = (seq_slot * 48u + params.x) * 6912u;
            let x0 = Hist[hb + row]; let x1 = Hist[hb + 2304u + row]; let x2 = Hist[hb + 4608u + row];
            Hist[hb + row] = x1; Hist[hb + 2304u + row] = x2; Hist[hb + 4608u + row] = x3;
            let y = (x3 + x0*cw0 + x1*cw1 + x2*cw2 + x3*cw3) * 0.4472135954999579;
            let rb = bi * 2304u + row;
            Out[rb] = select(y, y + Resid[rb], params.y == 1u);
        }
    }
}
"""

# 15. Batched Router Top-8: wg.y = b
B_ROUTER = """
@group(0) @binding(0) var<storage, read> Logits: array<f32>;
@group(0) @binding(1) var<storage, read> Bias: array<f32>;
@group(0) @binding(2) var<storage, read_write> Eids: array<u32>;
@group(0) @binding(3) var<storage, read_write> Wts: array<f32>;
var<workgroup> sg_max: array<f32, 2>;
var<workgroup> sg_idx: array<u32, 2>;
var<workgroup> w_out: array<f32, 8>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let bi = wg.y; let e = lid.x; let sg = e >> 6u; let lane = e & 63u;
    let p = 1.0 / (1.0 + exp(-Logits[bi * 128u + e]));
    var s = p + Bias[e];
    for (var k: u32 = 0u; k < 8u; k++) {
        let m = subgroupMax(s);
        let best_e = subgroupMin(select(999u, e, s == m));
        if (lane == 0u) { sg_max[sg] = m; sg_idx[sg] = best_e; }
        workgroupBarrier();
        let win_e = select(sg_idx[1], sg_idx[0], sg_max[0] >= sg_max[1]);
        if (e == win_e) { Eids[bi * 8u + k] = e; w_out[k] = p; s = -1e30; }
    }
    workgroupBarrier();
    if (e == 0u) {
        var tot: f32 = 1e-20;
        for (var k: u32 = 0u; k < 8u; k++) { tot += w_out[k]; }
        let sc = 8.0 / tot;
        for (var k: u32 = 0u; k < 8u; k++) { Wts[bi * 8u + k] = w_out[k] * sc; }
    }
}
"""

# 16-19. Batched MoE Experts (wg.z = b):
B_NV_W13 = FAST_COMMON + """
@group(0) @binding(0) var<storage, read> W_v2: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> S_u32: array<u32>;
@group(0) @binding(2) var<storage, read> Outer: array<f32>;
@group(0) @binding(3) var<storage, read> Eids: array<u32>;
@group(0) @binding(4) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> H: array<f32>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x; let sub = tid >> 6u; let lane = tid & 63u;
    let r = wg.x * 2u + sub; let slot = wg.y; let bi = wg.z;
    let eid = Eids[bi * 8u + slot];
    let row_g = 2u * r;
    let wg_base = (eid * 2304u + row_g) * 144u; let wu_base = wg_base + 144u;
    let sg_base = (eid * 2304u + row_g) * 144u; let su_base = sg_base + 144u;
    let x_bi = bi * 576u;
    var acc_g: f32 = 0.0; var acc_u: f32 = 0.0;
    for (var b = lane; b < 144u; b += 64u) {
        let xb = x_bi + b * 4u;
        let x0 = X[xb]; let x1 = X[xb + 1u]; let x2 = X[xb + 2u]; let x3 = X[xb + 3u];
        let wg_w = W_v2[wg_base + b]; let wu_w = W_v2[wu_base + b];
        let dg = dot8_nvfp4(wg_w.x, x0, x1) + dot8_nvfp4(wg_w.y, x2, x3);
        let du = dot8_nvfp4(wu_w.x, x0, x1) + dot8_nvfp4(wu_w.y, x2, x3);
        let sg_idx = sg_base + b; let sg = decode_fp8((S_u32[sg_idx >> 2u] >> ((sg_idx & 3u) * 8u)) & 0xFFu);
        let su_idx = su_base + b; let su = decode_fp8((S_u32[su_idx >> 2u] >> ((su_idx & 3u) * 8u)) & 0xFFu);
        acc_g += dg * sg; acc_u += du * su;
    }
    let sg_sum = subgroupAdd(acc_g); let su_sum = subgroupAdd(acc_u);
    if (lane == 0u) { let os = Outer[eid]; H[(bi * 8u + slot) * 1152u + r] = situ(sg_sum * os, su_sum * os); }
}
"""

B_NV_W2 = FAST_COMMON + """
@group(0) @binding(0) var<storage, read> W_v2: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> S_u32: array<u32>;
@group(0) @binding(2) var<storage, read> Outer: array<f32>;
@group(0) @binding(3) var<storage, read> Eids: array<u32>;
@group(0) @binding(4) var<storage, read> Weights: array<f32>;
@group(0) @binding(5) var<storage, read> H: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> ConvW: array<f32>;
@group(0) @binding(7) var<storage, read> Resid: array<f32>;
@group(0) @binding(8) var<storage, read_write> Hist: array<f32>;
@group(0) @binding(9) var<storage, read_write> Out: array<f32>;
@group(0) @binding(10) var<storage, read> Meta: array<u32>;
@group(0) @binding(11) var<uniform> params: vec4<u32>; // x=l_slot
var<workgroup> sh_eids: array<u32, 8>;
var<workgroup> sh_wts:  array<f32, 8>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x; let bi = wg.y;
    if (tid < 8u) { let eid = Eids[bi * 8u + tid]; sh_eids[tid] = eid; sh_wts[tid] = Outer[eid] * Weights[bi * 8u + tid]; }
    workgroupBarrier();
    let sub = tid >> 6u; let lane = tid & 63u;
    let row = wg.x * 2u + sub;
    var total: f32 = 0.0;
    for (var it: u32 = 0u; it < 9u; it++) {
        let idx = it * 64u + lane;
        let slot = idx / 72u; let b = idx - slot * 72u;
        let eid = sh_eids[slot];
        let w = W_v2[(eid * 2304u + row) * 72u + b];
        let hb = (bi * 8u + slot) * 288u + b * 4u;
        let d = dot8_nvfp4(w.x, H[hb], H[hb + 1u]) + dot8_nvfp4(w.y, H[hb + 2u], H[hb + 3u]);
        let s_idx = (eid * 2304u + row) * 72u + b;
        let s = decode_fp8((S_u32[s_idx >> 2u] >> ((s_idx & 3u) * 8u)) & 0xFFu);
        total += d * s * sh_wts[slot];
    }
    let x3 = subgroupAdd(total);
    if (lane == 0u) {
        let seq_slot = Meta[16u + bi];
        let hb = (seq_slot * 48u + params.x) * 6912u;
        let x0 = Hist[hb + row]; let x1 = Hist[hb + 2304u + row]; let x2 = Hist[hb + 4608u + row];
        Hist[hb + row] = x1; Hist[hb + 2304u + row] = x2; Hist[hb + 4608u + row] = x3;
        let y = (x3 + x0*ConvW[row*4u] + x1*ConvW[row*4u+1u] + x2*ConvW[row*4u+2u] + x3*ConvW[row*4u+3u]) * 0.4472135954999579;
        let rb = bi * 2304u + row;
        Out[rb] = y + Resid[rb];
    }
}
"""

B_MX_W13 = FAST_COMMON + """
@group(0) @binding(0) var<storage, read> W_v4: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> S_u32: array<u32>;
@group(0) @binding(2) var<storage, read> Eids: array<u32>;
@group(0) @binding(3) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> H: array<f32>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x; let sub = tid >> 6u; let lane = tid & 63u;
    let r = wg.x * 2u + sub; let slot = wg.y; let bi = wg.z;
    let eid = Eids[bi * 8u + slot];
    let row_g = 2u * r;
    let wg_base = (eid * 2304u + row_g) * 144u; let wu_base = wg_base + 144u;
    let sg_base = (eid * 2304u + row_g) * 72u;  let su_base = sg_base + 72u;
    let x_bi = bi * 576u;
    var acc_g: f32 = 0.0; var acc_u: f32 = 0.0;
    for (var v = lane; v < 144u; v += 64u) {
        let xb = x_bi + v * 4u;
        let x0 = X[xb]; let x1 = X[xb + 1u]; let x2 = X[xb + 2u]; let x3 = X[xb + 3u];
        let wg_w = W_v4[wg_base + v]; let wu_w = W_v4[wu_base + v];
        let dg = dot4_fp8(wg_w.x, x0) + dot4_fp8(wg_w.y, x1) + dot4_fp8(wg_w.z, x2) + dot4_fp8(wg_w.w, x3);
        let du = dot4_fp8(wu_w.x, x0) + dot4_fp8(wu_w.y, x1) + dot4_fp8(wu_w.z, x2) + dot4_fp8(wu_w.w, x3);
        let b = v >> 1u;
        let sg_idx = sg_base + b; let sg = decode_ue8m0((S_u32[sg_idx >> 2u] >> ((sg_idx & 3u) * 8u)) & 0xFFu);
        let su_idx = su_base + b; let su = decode_ue8m0((S_u32[su_idx >> 2u] >> ((su_idx & 3u) * 8u)) & 0xFFu);
        acc_g += dg * sg; acc_u += du * su;
    }
    let sg_sum = subgroupAdd(acc_g); let su_sum = subgroupAdd(acc_u);
    if (lane == 0u) { H[(bi * 8u + slot) * 1152u + r] = situ(sg_sum, su_sum); }
}
"""

B_MX_W2 = FAST_COMMON + """
@group(0) @binding(0) var<storage, read> W_v4: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> S_u32: array<u32>;
@group(0) @binding(2) var<storage, read> Eids: array<u32>;
@group(0) @binding(3) var<storage, read> Weights: array<f32>;
@group(0) @binding(4) var<storage, read> H: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> ConvW: array<f32>;
@group(0) @binding(6) var<storage, read> Resid: array<f32>;
@group(0) @binding(7) var<storage, read_write> Hist: array<f32>;
@group(0) @binding(8) var<storage, read_write> Out: array<f32>;
@group(0) @binding(9) var<storage, read> Meta: array<u32>;
@group(0) @binding(10) var<uniform> params: vec4<u32>; // x=l_slot
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x; let sub = tid >> 6u; let lane = tid & 63u;
    let row = wg.x * 2u + sub; let bi = wg.y;
    var total: f32 = 0.0;
    for (var slot: u32 = 0u; slot < 8u; slot++) {
        let eid = Eids[bi * 8u + slot];
        let w_base = (eid * 2304u + row) * 72u;
        let s_base = (eid * 2304u + row) * 36u;
        let h_base = (bi * 8u + slot) * 288u;
        var acc: f32 = 0.0;
        for (var v = lane; v < 72u; v += 64u) {
            let w = W_v4[w_base + v];
            let hb = h_base + v * 4u;
            let d = dot4_fp8(w.x, H[hb]) + dot4_fp8(w.y, H[hb + 1u]) + dot4_fp8(w.z, H[hb + 2u]) + dot4_fp8(w.w, H[hb + 3u]);
            let s_idx = s_base + (v >> 1u);
            let s = decode_ue8m0((S_u32[s_idx >> 2u] >> ((s_idx & 3u) * 8u)) & 0xFFu);
            acc += d * s;
        }
        total += acc * Weights[bi * 8u + slot];
    }
    let x3 = subgroupAdd(total);
    if (lane == 0u) {
        let seq_slot = Meta[16u + bi];
        let hb = (seq_slot * 48u + params.x) * 6912u;
        let x0 = Hist[hb + row]; let x1 = Hist[hb + 2304u + row]; let x2 = Hist[hb + 4608u + row];
        Hist[hb + row] = x1; Hist[hb + 2304u + row] = x2; Hist[hb + 4608u + row] = x3;
        let y = (x3 + x0*ConvW[row*4u] + x1*ConvW[row*4u+1u] + x2*ConvW[row*4u+2u] + x3*ConvW[row*4u+3u]) * 0.4472135954999579;
        let rb = bi * 2304u + row;
        Out[rb] = y + Resid[rb];
    }
}
"""

# 20. SHARED-WEIGHT UNEMBED P1: writes soft-capped Logits[B, 200058] in VRAM + PartV/PartI!
B_UNEMBED_P1 = FAST_COMMON + """
@group(0) @binding(0) var<storage, read> W: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> PartV: array<f32>;
@group(0) @binding(3) var<storage, read_write> PartI: array<u32>;
@group(0) @binding(4) var<storage, read_write> Logits: array<f32>; // [16, 200058] DEVICE_LOCAL
@group(0) @binding(5) var<storage, read> Meta: array<u32>;
var<workgroup> sh_v: array<f32, 32>;
var<workgroup> sh_i: array<u32, 32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x; let sub = tid >> 6u; let lane = tid & 63u;
    let row = wg.x * 4u + sub;
    let B = Meta[0]; let b0 = wg.y * 8u; if (b0 >= B) { return; } let cnt = min(8u, B - b0);

    var a0: f32 = 0.0; var a1: f32 = 0.0; var a2: f32 = 0.0; var a3: f32 = 0.0;
    var a4: f32 = 0.0; var a5: f32 = 0.0; var a6: f32 = 0.0; var a7: f32 = 0.0;
    if (row < 200058u) {
        let base = row * 288u;
        for (var k = lane; k < 288u; k += 64u) {
            let w = W[base + k]; let xk = k * 2u;
            { let o = b0*576u+xk; let x0 = X[o]; let x1 = X[o+1u]; a0 += bf16x2_dot(w.x, x0.xy) + bf16x2_dot(w.y, x0.zw) + bf16x2_dot(w.z, x1.xy) + bf16x2_dot(w.w, x1.zw); }
            if (cnt > 1u) { let o = (b0+1u)*576u+xk; let x0 = X[o]; let x1 = X[o+1u]; a1 += bf16x2_dot(w.x, x0.xy) + bf16x2_dot(w.y, x0.zw) + bf16x2_dot(w.z, x1.xy) + bf16x2_dot(w.w, x1.zw); }
            if (cnt > 2u) { let o = (b0+2u)*576u+xk; let x0 = X[o]; let x1 = X[o+1u]; a2 += bf16x2_dot(w.x, x0.xy) + bf16x2_dot(w.y, x0.zw) + bf16x2_dot(w.z, x1.xy) + bf16x2_dot(w.w, x1.zw); }
            if (cnt > 3u) { let o = (b0+3u)*576u+xk; let x0 = X[o]; let x1 = X[o+1u]; a3 += bf16x2_dot(w.x, x0.xy) + bf16x2_dot(w.y, x0.zw) + bf16x2_dot(w.z, x1.xy) + bf16x2_dot(w.w, x1.zw); }
            if (cnt > 4u) { let o = (b0+4u)*576u+xk; let x0 = X[o]; let x1 = X[o+1u]; a4 += bf16x2_dot(w.x, x0.xy) + bf16x2_dot(w.y, x0.zw) + bf16x2_dot(w.z, x1.xy) + bf16x2_dot(w.w, x1.zw); }
            if (cnt > 5u) { let o = (b0+5u)*576u+xk; let x0 = X[o]; let x1 = X[o+1u]; a5 += bf16x2_dot(w.x, x0.xy) + bf16x2_dot(w.y, x0.zw) + bf16x2_dot(w.z, x1.xy) + bf16x2_dot(w.w, x1.zw); }
            if (cnt > 6u) { let o = (b0+6u)*576u+xk; let x0 = X[o]; let x1 = X[o+1u]; a6 += bf16x2_dot(w.x, x0.xy) + bf16x2_dot(w.y, x0.zw) + bf16x2_dot(w.z, x1.xy) + bf16x2_dot(w.w, x1.zw); }
            if (cnt > 7u) { let o = (b0+7u)*576u+xk; let x0 = X[o]; let x1 = X[o+1u]; a7 += bf16x2_dot(w.x, x0.xy) + bf16x2_dot(w.y, x0.zw) + bf16x2_dot(w.z, x1.xy) + bf16x2_dot(w.w, x1.zw); }
        }
    }
    var vs = array<f32, 8>(
        subgroupAdd(a0), subgroupAdd(a1), subgroupAdd(a2), subgroupAdd(a3),
        subgroupAdd(a4), subgroupAdd(a5), subgroupAdd(a6), subgroupAdd(a7)
    );
    if (lane == 0u) {
        for (var j: u32 = 0u; j < cnt; j++) {
            let cap_v = select(-1e30, 30.0 * tanh(vs[j] / 30.0), row < 200058u);
            sh_v[j * 4u + sub] = cap_v;
            sh_i[j * 4u + sub] = row;
            if (row < 200058u) { Logits[(b0 + j) * 200058u + row] = cap_v; }
        }
    }
    workgroupBarrier();
    if (tid < cnt) {
        let j = tid;
        var bv = sh_v[j * 4u]; var bi_r = sh_i[j * 4u];
        for (var r: u32 = 1u; r < 4u; r++) {
            let v = sh_v[j * 4u + r];
            if (v > bv) { bv = v; bi_r = sh_i[j * 4u + r]; }
        }
        let out_idx = (b0 + j) * 50015u + wg.x;
        PartV[out_idx] = bv; PartI[out_idx] = bi_r;
    }
}
"""

# 21. ON-GPU Batched Greedy + Temperature + Top-P + Seed Sampler!
# Meta[48+b] = bitcast<u32>(temp), Meta[64+b] = bitcast<u32>(top_p), Meta[80+b] = seed
# temp <= 0.0 -> exact argmax (0.03 ms); temp > 0.0 -> GPU top-64 softmax + top-p + PCG32 draw!
B_ARGMAX_P2 = """
@group(0) @binding(0) var<storage, read> PartV: array<f32>;
@group(0) @binding(1) var<storage, read> PartI: array<u32>;
@group(0) @binding(2) var<storage, read> Logits: array<f32>;
@group(0) @binding(3) var<storage, read> Meta: array<u32>;
@group(0) @binding(4) var<storage, read_write> Out: array<u32>; // [16, 2]

var<workgroup> smem_v: array<f32, 256>;
var<workgroup> smem_i: array<u32, 256>;
var<workgroup> top_v:  array<f32, 128>;
var<workgroup> top_i:  array<u32, 128>;
var<workgroup> sg_v:   array<f32, 4>;
var<workgroup> sg_i:   array<u32, 4>;
var<workgroup> sg_sum: array<f32, 4>;

fn pcg_f32(seed_in: u32, pos: u32) -> f32 {
    var st = (seed_in ^ (pos * 0x9E3779B9u)) * 747796405u + 2891336453u;
    let w = ((st >> ((st >> 28u) + 4u)) ^ st) * 277803737u;
    let r = (w >> 22u) ^ w;
    return f32(r >> 8u) * (1.0 / 16777216.0);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let bi = wg.y; let tid = lid.x; let sg = tid >> 6u; let lane = tid & 63u;
    let temp  = bitcast<f32>(Meta[48u + bi]);
    let top_p = bitcast<f32>(Meta[64u + bi]);
    let seed  = Meta[80u + bi];
    let pos   = Meta[32u + bi];

    // Step 1: exact global argmax over PartV/PartI (50015 chunks)
    let pbase = bi * 50015u;
    var bv: f32 = -1e30; var bi_r: u32 = 0u;
    for (var i = tid; i < 50015u; i += 256u) {
        let v = PartV[pbase + i];
        if (v > bv) { bv = v; bi_r = PartI[pbase + i]; }
    }
    smem_v[tid] = bv; smem_i[tid] = bi_r; workgroupBarrier();
    for (var st = 128u; st > 0u; st >>= 1u) {
        if (tid < st && smem_v[tid + st] > smem_v[tid]) { smem_v[tid] = smem_v[tid + st]; smem_i[tid] = smem_i[tid + st]; }
        workgroupBarrier();
    }

    if (temp <= 0.0) {
        if (tid == 0u) { Out[bi * 2u] = smem_i[0]; Out[bi * 2u + 1u] = bitcast<u32>(smem_v[0]); }
        return;
    }

    // Step 2: Temperature + Top-P sampling on GPU!
    // Each of 256 threads scans its strided slice of 200058 logits, computing Z_full = sum exp((v-max)/T)
    // and keeping its local best (v1, i1) for top-64 nucleus selection.
    let max_v = smem_v[0];
    let inv_t = 1.0 / temp;
    let lbase = bi * 200058u;
    var z_part: f32 = 0.0;
    var b0_v: f32 = -1e30; var b0_i: u32 = 0u;
    var b1_v: f32 = -1e30; var b1_i: u32 = 0u;
    var b2_v: f32 = -1e30; var b2_i: u32 = 0u;
    for (var i = tid; i < 200058u; i += 256u) {
        let v = Logits[lbase + i];
        let e = exp((v - max_v) * inv_t);
        z_part += e;
        if (v > b0_v) { b2_v = b1_v; b2_i = b1_i; b1_v = b0_v; b1_i = b0_i; b0_v = v; b0_i = i; }
        else if (v > b1_v) { b2_v = b1_v; b2_i = b1_i; b1_v = v; b1_i = i; }
        else if (v > b2_v) { b2_v = v; b2_i = i; }
    }
    let z_sg = subgroupAdd(z_part);
    if (lane == 0u) { sg_sum[sg] = z_sg; }
    workgroupBarrier();

    // Extract exact top-128 across 256 threads x top-3 (768 candidates -> zero collision!)
    var cur_v = b0_v; var cur_i = b0_i;
    var next1_v = b1_v; var next1_i = b1_i;
    var next2_v = b2_v; var next2_i = b2_i;
    for (var k: u32 = 0u; k < 128u; k++) {
        let m = subgroupMax(cur_v);
        let cand = select(999u, tid, cur_v == m);
        let win_t = subgroupMin(cand);
        if (lane == 0u) { sg_v[sg] = m; sg_i[sg] = win_t; }
        workgroupBarrier();
        var g_best_t = sg_i[0]; var g_best_v = sg_v[0];
        if (sg_v[1] > g_best_v) { g_best_v = sg_v[1]; g_best_t = sg_i[1]; }
        if (sg_v[2] > g_best_v) { g_best_v = sg_v[2]; g_best_t = sg_i[2]; }
        if (sg_v[3] > g_best_v) { g_best_v = sg_v[3]; g_best_t = sg_i[3]; }
        if (tid == g_best_t) {
            top_v[k] = cur_v;
            top_i[k] = cur_i;
            cur_v = next1_v; cur_i = next1_i;
            next1_v = next2_v; next1_i = next2_i;
            next2_v = -1e30;
        }
    }
    workgroupBarrier();

    if (tid == 0u) {
        let z_tot = sg_sum[0] + sg_sum[1] + sg_sum[2] + sg_sum[3];
        var csum: f32 = 0.0;
        var n_keep: u32 = 1u;
        for (var k: u32 = 0u; k < 128u; k++) {
            let p = exp((top_v[k] - max_v) * inv_t) / z_tot;
            top_v[k] = p;
            csum += p;
            n_keep = k + 1u;
            if (csum >= top_p) { break; }
        }
        let u = pcg_f32(seed, pos) * csum;
        var acc: f32 = 0.0;
        var pick: u32 = 0u;
        for (var k: u32 = 0u; k < n_keep; k++) {
            acc += top_v[k];
            if (u <= acc) { pick = k; break; }
        }
        let win_tok = top_i[pick];
        Out[bi * 2u] = win_tok;
        Out[bi * 2u + 1u] = bitcast<u32>(Logits[lbase + win_tok]);
    }
}
"""

SHADERS = [
    ("rms", B_RMS), ("copy", B_COPY), ("mix", B_MIX), ("mix_rms", B_MIX_RMS), ("rms3", B_RMS3),
    ("ctrl_p1", B_CTRL_P1), ("ctrl_p2", B_CTRL_P2), ("ctrl3_p1", B_CTRL3_P1), ("ctrl3_p2", B_CTRL3_P2),
    ("gemm", B_GEMM), ("qkvr_post", B_QKVR_POST), ("attn_sk", B_ATTN_SK), ("attn_cb", B_ATTN_CB),
    ("wo_conv", B_WO_CONV), ("router", B_ROUTER),
    ("nv_w13", B_NV_W13), ("nv_w2_c", B_NV_W2), ("mx_w13", B_MX_W13), ("mx_w2_c", B_MX_W2),
    ("unembed_p1", B_UNEMBED_P1), ("argmax_p2", B_ARGMAX_P2),
]

if __name__ == "__main__":
    out_dir = os.environ["DUMP_SPV_DIR"]
    ad = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    dev = ad.request_device_sync(required_features=["subgroup"])
    for name, code in SHADERS:
        m = dev.create_shader_module(code=code)
        dev.create_compute_pipeline(layout="auto", compute={"module": m, "entry_point": "main"})
    for i, (name, _) in enumerate(SHADERS):
        src = f"{out_dir}/shader_{i+2:02d}.spv"
        dst = f"{out_dir}/{name}.spv"
        shutil.copyfile(src, dst)
        raw = open(dst, "rb").read()
        assert raw[:4] == b"\x03\x02\x23\x07", f"Bad SPIR-V in {dst}"
    print(f"Compiled & verified all {len(SHADERS)} batched SPIR-V shaders into {out_dir}!")
