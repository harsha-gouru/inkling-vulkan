import os, shutil, sys, wgpu
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
from build_batched_spv import (
    FAST_COMMON, B_RMS, B_COPY, B_MIX, B_MIX_RMS, B_RMS3,
    B_CTRL_P1, B_CTRL_P2, B_CTRL3_P1, B_CTRL3_P2, B_GEMM, B_ATTN_SK, B_ATTN_CB, B_ROUTER
)

# ==============================================================================
# Chunked Prefill Shaders (C up to 64 tokens of ONE sequence seq_slot!)
# Meta[0] = C (chunk_len, 1..64)
# Meta[1] = seq_slot (0..15)
# Meta[2] = start_pos (0..4095)
# Meta[16+i] = seq_slot, Meta[32+i] = start_pos + i (so B_ATTN_SK works UNMODIFIED for C <= 64!
# Wait: B_ATTN_SK reads Meta[16+bi] and Meta[32+bi] which only has 16 entries in Meta[16..31]!
# So P_ATTN_SK reads seq_slot = Meta[1], pos = Meta[2] + bi — cleaner and supports any C up to 64!)
# ==============================================================================

# 1. Parallel 4-tap causal K/V short-conv + Q/R clip + write ALL C K/V to cache FIRST (Way 1)!
# Dispatch: (64, C, 1)
P_QKVR_POST = FAST_COMMON + """
@group(0) @binding(0) var<storage, read_write> QRSG: array<f32>;   // [C, 8600]
@group(0) @binding(1) var<storage, read> Kraw: array<f32>;         // [C, 1056]
@group(0) @binding(2) var<storage, read> Vraw: array<f32>;         // [C, 1056]
@group(0) @binding(3) var<storage, read> QClip: array<f32>;
@group(0) @binding(4) var<storage, read> RClip: array<f32>;
@group(0) @binding(5) var<storage, read> KConv: array<f32>;
@group(0) @binding(6) var<storage, read> VConv: array<f32>;
@group(0) @binding(7) var<storage, read> HistKIn: array<f32>;      // [16, 48, 3, 1024] (read-only during conv!)
@group(0) @binding(8) var<storage, read> HistVIn: array<f32>;      // [16, 48, 3, 1024]
@group(0) @binding(9) var<storage, read_write> KCache0: array<u32>;
@group(0) @binding(10) var<storage, read_write> KCache1: array<u32>;
@group(0) @binding(11) var<storage, read_write> VCache0: array<u32>;
@group(0) @binding(12) var<storage, read_write> VCache1: array<u32>;
@group(0) @binding(13) var<storage, read> Meta: array<u32>;        // [0]=C, [1]=seq_slot, [2]=start_pos
@group(0) @binding(14) var<uniform> params: vec4<u32>;             // x=(l_slot<<8)|cache_idx, y=is_full, z=k_off, w=v_off

var<workgroup> vals: array<f32, 128>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let C = Meta[0];
    let bi = wg.y;
    if (bi >= C) { return; }
    let id = wg.x; let tid = lid.x;
    let l_slot = params.x >> 8u;
    let cache_idx = params.x & 0xFFu;
    let is_full = params.y;
    let seq_slot = Meta[1];
    let pos = Meta[2] + bi;
    let qrsg_b = bi * 8600u;

    if (id < 24u) {
        let base = qrsg_b + id * 128u; let cb = id * 128u;
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
        let h = id - 24u; let base = qrsg_b + 3072u + h * 16u;
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
    var sq: f32 = 0.0;

    for (var j: u32 = 0u; j < 2u; j++) {
        let d = tid + j * 64u;
        let c = head * 128u + d;
        var y: f32;
        if (is_v == 0u) {
            // Read x0 (bi-3), x1 (bi-2), x2 (bi-1), x3 (bi) from HistKIn or Kraw/QRSG!
            var x0: f32; var x1: f32; var x2: f32; var x3: f32;
            if (params.z > 0u) {
                x3 = QRSG[bi * 8600u + params.z + c];
                x2 = select(HistKIn[h_base + 2048u + c], QRSG[(bi - 1u) * 8600u + params.z + c], bi >= 1u);
                x1 = select(HistKIn[h_base + (1u + bi) * 1024u + c], QRSG[(bi - 2u) * 8600u + params.z + c], bi >= 2u);
                x0 = select(HistKIn[h_base + bi * 1024u + c], QRSG[(bi - 3u) * 8600u + params.z + c], bi >= 3u);
            } else {
                x3 = Kraw[bi * 1056u + c];
                x2 = select(HistKIn[h_base + 2048u + c], Kraw[(bi - 1u) * 1056u + c], bi >= 1u);
                x1 = select(HistKIn[h_base + (1u + bi) * 1024u + c], Kraw[(bi - 2u) * 1056u + c], bi >= 2u);
                x0 = select(HistKIn[h_base + bi * 1024u + c], Kraw[(bi - 3u) * 1056u + c], bi >= 3u);
            }
            if (is_full == 0u) {
                y = (x3 + x0*KConv[c*4u] + x1*KConv[c*4u+1u] + x2*KConv[c*4u+2u] + x3*KConv[c*4u+3u]) * 0.4472135954999579;
            } else {
                let db = bi * 1056u + 1024u + head * 4u;
                let out = x3*(KConv[c]+Kraw[db]) + x2*(KConv[1024u+c]+Kraw[db+1u]) + x1*(KConv[2048u+c]+Kraw[db+2u]) + x0*(KConv[3072u+c]+Kraw[db+3u]);
                y = (x3 + out) * 0.3333333333333333;
            }
        } else {
            var x0: f32; var x1: f32; var x2: f32; var x3: f32;
            if (params.w > 0u) {
                x3 = QRSG[bi * 8600u + params.w + c];
                x2 = select(HistVIn[h_base + 2048u + c], QRSG[(bi - 1u) * 8600u + params.w + c], bi >= 1u);
                x1 = select(HistVIn[h_base + (1u + bi) * 1024u + c], QRSG[(bi - 2u) * 8600u + params.w + c], bi >= 2u);
                x0 = select(HistVIn[h_base + bi * 1024u + c], QRSG[(bi - 3u) * 8600u + params.w + c], bi >= 3u);
            } else {
                x3 = Vraw[bi * 1056u + c];
                x2 = select(HistVIn[h_base + 2048u + c], Vraw[(bi - 1u) * 1056u + c], bi >= 1u);
                x1 = select(HistVIn[h_base + (1u + bi) * 1024u + c], Vraw[(bi - 2u) * 1056u + c], bi >= 2u);
                x0 = select(HistVIn[h_base + bi * 1024u + c], Vraw[(bi - 3u) * 1056u + c], bi >= 3u);
            }
            if (is_full == 0u) {
                y = (x3 + x0*VConv[c*4u] + x1*VConv[c*4u+1u] + x2*VConv[c*4u+2u] + x3*VConv[c*4u+3u]) * 0.4472135954999579;
            } else {
                let db = bi * 1056u + 1024u + head * 4u;
                let out = x3*(VConv[c]+Vraw[db]) + x2*(VConv[1024u+c]+Vraw[db+1u]) + x1*(VConv[2048u+c]+Vraw[db+2u]) + x0*(VConv[3072u+c]+Vraw[db+3u]);
                y = (x3 + out) * 0.3333333333333333;
            }
        }
        vals[d] = y; sq += y * y;
    }
    let sq_sum = subgroupAdd(sq);
    let mult = select(min(1.0, 22.627416997969522 / (sqrt(sq_sum) + 1e-6)), inverseSqrt(sq_sum / 128.0 + 1e-6), is_v == 1u);

    let seq_cap = Meta[14];
    let half_b = Meta[15];
    let hi = (seq_slot >= half_b);
    let half_slot = select(seq_slot, seq_slot - half_b, hi);
    let n_layers = select(40u, 8u, is_full == 1u);
    let cap = select(1024u, seq_cap, is_full == 1u);
    let pos_phys = select(pos & 1023u, pos, is_full == 1u);
    let mult_s = mult * select(16.0, 32.0, is_v == 1u);
    vals[tid] = vals[tid] * mult_s;
    vals[tid + 64u] = vals[tid + 64u] * mult_s;
    workgroupBarrier();
    if (tid < 32u) {
        let c_off = ((half_slot * n_layers + cache_idx) * cap + pos_phys) * 256u + head * 32u;
        let packed = encode_fp8(vals[tid])
                   | (encode_fp8(vals[tid + 32u]) << 8u)
                   | (encode_fp8(vals[tid + 64u]) << 16u)
                   | (encode_fp8(vals[tid + 96u]) << 24u);
        if (is_v == 0u) {
            if (!hi) { KCache0[c_off + tid] = packed; }
            else     { KCache1[c_off + tid] = packed; }
        } else {
            if (!hi) { VCache0[c_off + tid] = packed; }
            else     { VCache1[c_off + tid] = packed; }
        }
    }
}
"""

# 2. Shift KV conv histories after P_QKVR_POST (1 tiny 8-WG kernel, race-free!):
# Dispatch: (8, 1, 1) x 256 = 2048 threads (0..1023 -> HistK, 1024..2047 -> HistV)
P_SHIFT_KV_HIST = """
@group(0) @binding(0) var<storage, read> QRSG: array<f32>;
@group(0) @binding(1) var<storage, read> Kraw: array<f32>;
@group(0) @binding(2) var<storage, read> Vraw: array<f32>;
@group(0) @binding(3) var<storage, read_write> HistK: array<f32>;
@group(0) @binding(4) var<storage, read_write> HistV: array<f32>;
@group(0) @binding(5) var<storage, read> Meta: array<u32>;
@group(0) @binding(6) var<uniform> params: vec4<u32>; // x=l_slot, z=k_off, w=v_off
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= 2048u) { return; }
    let is_v = idx >= 1024u;
    let c = idx & 1023u;
    let C = Meta[0]; let seq_slot = Meta[1];
    let hb = (seq_slot * 48u + params.x) * 3072u;
    // New history [0, 1, 2] = tokens [C-3, C-2, C-1] (falling back to old Hist if C < 3)
    var h0: f32; var h1: f32; var h2: f32;
    if (!is_v) {
        let old0 = HistK[hb + c]; let old1 = HistK[hb + 1024u + c]; let old2 = HistK[hb + 2048u + c];
        if (params.z > 0u) {
            h2 = QRSG[(C - 1u) * 8600u + params.z + c];
            h1 = select(old2, QRSG[(C - 2u) * 8600u + params.z + c], C >= 2u);
            h0 = select(select(old1, old2, C == 2u), QRSG[(C - 3u) * 8600u + params.z + c], C >= 3u);
        } else {
            h2 = Kraw[(C - 1u) * 1056u + c];
            h1 = select(old2, Kraw[(C - 2u) * 1056u + c], C >= 2u);
            h0 = select(select(old1, old2, C == 2u), Kraw[(C - 3u) * 1056u + c], C >= 3u);
        }
        HistK[hb + c] = h0; HistK[hb + 1024u + c] = h1; HistK[hb + 2048u + c] = h2;
    } else {
        let old0 = HistV[hb + c]; let old1 = HistV[hb + 1024u + c]; let old2 = HistV[hb + 2048u + c];
        if (params.w > 0u) {
            h2 = QRSG[(C - 1u) * 8600u + params.w + c];
            h1 = select(old2, QRSG[(C - 2u) * 8600u + params.w + c], C >= 2u);
            h0 = select(select(old1, old2, C == 2u), QRSG[(C - 3u) * 8600u + params.w + c], C >= 3u);
        } else {
            h2 = Vraw[(C - 1u) * 1056u + c];
            h1 = select(old2, Vraw[(C - 2u) * 1056u + c], C >= 2u);
            h0 = select(select(old1, old2, C == 2u), Vraw[(C - 3u) * 1056u + c], C >= 3u);
        }
        HistV[hb + c] = h0; HistV[hb + 1024u + c] = h1; HistV[hb + 2048u + c] = h2;
    }
}
"""

# 3. Prefill Split-K Attention: reads seq_slot=Meta[1], pos=Meta[2]+wg.z!
# Dispatch: (24, 16, C)
P_ATTN_SK = B_ATTN_SK.replace("let seq_slot = Meta[16u + bi];\n    let pos = Meta[32u + bi];",
                              "if (bi >= Meta[0]) { return; }\n    let seq_slot = Meta[1];\n    let pos = Meta[2] + bi;")

# 4. Prefill parallel 4-tap conv + resid (for attn_sconv after WO GEMM, and mlp_sconv after grouped W2)!
# First computes all C outputs (reading Hist read-only), then P_SHIFT_2304_HIST updates Hist!
# Dispatch: (9, C, 1)
P_CONV_RESID = """
@group(0) @binding(0) var<storage, read> X: array<f32>;            // [C, 2304]
@group(0) @binding(1) var<storage, read> ConvW: array<f32>;
@group(0) @binding(2) var<storage, read> Resid: array<f32>;        // [C, 2304]
@group(0) @binding(3) var<storage, read_write> Hist: array<f32>;   // [16, 48, 3, 2304] (fused shift!)
@group(0) @binding(4) var<storage, read_write> Out: array<f32>;    // [C, 2304]
@group(0) @binding(5) var<storage, read> Meta: array<u32>;
@group(0) @binding(6) var<uniform> params: vec4<u32>;              // x=l_slot, y=add_resid
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let c = wg.x * 256u + lid.x;
    if (c >= 2304u) { return; }
    let C = Meta[0];
    let hb = (Meta[1] * 48u + params.x) * 6912u;
    var x0 = Hist[hb + c];
    var x1 = Hist[hb + 2304u + c];
    var x2 = Hist[hb + 4608u + c];
    let cw0 = ConvW[c * 4u];
    let cw1 = ConvW[c * 4u + 1u];
    let cw2 = ConvW[c * 4u + 2u];
    let cw3 = ConvW[c * 4u + 3u];
    let add_r = (params.y == 1u);
    for (var bi: u32 = 0u; bi < C; bi++) {
        let ob = bi * 2304u + c;
        let x3 = X[ob];
        let y = (x3 + x0 * cw0 + x1 * cw1 + x2 * cw2 + x3 * cw3) * 0.4472135954999579;
        Out[ob] = select(y, y + Resid[ob], add_r);
        x0 = x1; x1 = x2; x2 = x3;
    }
    Hist[hb + c] = x0;
    Hist[hb + 2304u + c] = x1;
    Hist[hb + 4608u + c] = x2;
}
"""

P_SHIFT_2304_HIST = """
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Hist: array<f32>;
@group(0) @binding(2) var<storage, read> Meta: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>; // x=l_slot
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= 2304u) { return; }
    let C = Meta[0]; let hb = (Meta[1] * 48u + params.x) * 6912u;
    let old0 = Hist[hb + c]; let old1 = Hist[hb + 2304u + c]; let old2 = Hist[hb + 4608u + c];
    let h2 = X[(C - 1u) * 2304u + c];
    let h1 = select(old2, X[(C - 2u) * 2304u + c], C >= 2u);
    let h0 = select(select(old1, old2, C == 2u), X[(C - 3u) * 2304u + c], C >= 3u);
    Hist[hb + c] = h0; Hist[hb + 2304u + c] = h1; Hist[hb + 4608u + c] = h2;
}
"""

# 5. FIX 2: GPU EXPERT GROUPING FOR CHUNKED PREFILL!
# 1 WG of 128 threads (1 thread per expert e=0..127):
# Scans Eids[0..C-1, 0..7] -> writes ExpCnt[e] and ExpToks[e * 64 + k] = (tok << 3) | slot!
P_GROUP_EXPERTS = """
@group(0) @binding(0) var<storage, read> Eids: array<u32>;        // [C, 8]
@group(0) @binding(1) var<storage, read_write> ExpCnt: array<u32>; // [128] -> compacted (eid << 8) | cnt
@group(0) @binding(2) var<storage, read_write> ExpToks: array<u32>;// [128, 64] -> (tok << 3) | slot
@group(0) @binding(3) var<storage, read> Meta: array<u32>;         // Meta[0] = C
var<workgroup> sh_act: array<u32, 128>;
var<workgroup> sh_out: array<u32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let e = lid.x;
    let C = Meta[0];
    var cnt: u32 = 0u;
    let base = e * 64u;
    for (var tok: u32 = 0u; tok < C; tok++) {
        for (var sl: u32 = 0u; sl < 8u; sl++) {
            if (Eids[tok * 8u + sl] == e) {
                ExpToks[base + cnt] = (tok << 3u) | sl;
                cnt += 1u;
            }
        }
    }
    let packed = select(0u, (e << 8u) | cnt, cnt > 0u);
    sh_act[e] = packed;
    sh_out[e] = 0u;
    workgroupBarrier();
    if (cnt > 0u) {
        var pos: u32 = 0u;
        for (var j: u32 = 0u; j < e; j++) {
            if (sh_act[j] != 0u) { pos += 1u; }
        }
        sh_out[pos] = packed;
    }
    workgroupBarrier();
    ExpCnt[e] = sh_out[e];
}
"""

# 6. EXPERT-GROUPED NVFP4 W13 + SiTU: reads each active expert's weights ONCE for up to 8 tokens!
# Dispatch: (576, 128, 1) -> wg.y = eid (0..127)!
P_NV_W13_GRP = FAST_COMMON + """
@group(0) @binding(0) var<storage, read> W_v2: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> S_u32: array<u32>;
@group(0) @binding(2) var<storage, read> Outer: array<f32>;
@group(0) @binding(3) var<storage, read> ExpCnt: array<u32>;
@group(0) @binding(4) var<storage, read> ExpToks: array<u32>;
@group(0) @binding(5) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> H: array<f32>; // [C, 8, 1152]

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let pk_ec = ExpCnt[wg.y];
    let cnt = pk_ec & 0xFFu;
    if (cnt == 0u) { return; }
    let eid = pk_ec >> 8u;
    let tid = lid.x; let sub = tid >> 6u; let lane = tid & 63u;
    let r = wg.x * 2u + sub;
    let row_g = 2u * r;
    let wg_base = (eid * 2304u + row_g) * 144u; let wu_base = wg_base + 144u;
    let sg_base = (eid * 2304u + row_g) * 144u; let su_base = sg_base + 144u;
    let os = Outer[eid];

    for (var t0: u32 = 0u; t0 < cnt; t0 += 8u) {
        let tc = min(8u, cnt - t0);
        var ts = array<u32, 8>(0u,0u,0u,0u,0u,0u,0u,0u);
        var sls = array<u32, 8>(0u,0u,0u,0u,0u,0u,0u,0u);
        for (var j: u32 = 0u; j < tc; j++) {
            let pk = ExpToks[eid * 64u + t0 + j];
            ts[j] = pk >> 3u; sls[j] = pk & 7u;
        }
        var ag0: f32=0.0; var au0: f32=0.0; var ag1: f32=0.0; var au1: f32=0.0;
        var ag2: f32=0.0; var au2: f32=0.0; var ag3: f32=0.0; var au3: f32=0.0;
        var ag4: f32=0.0; var au4: f32=0.0; var ag5: f32=0.0; var au5: f32=0.0;
        var ag6: f32=0.0; var au6: f32=0.0; var ag7: f32=0.0; var au7: f32=0.0;

        for (var b = lane; b < 144u; b += 64u) {
            let wg_w = W_v2[wg_base + b]; let wu_w = W_v2[wu_base + b];
            let sg_idx = sg_base + b; let sg = decode_fp8((S_u32[sg_idx >> 2u] >> ((sg_idx & 3u) * 8u)) & 0xFFu);
            let su_idx = su_base + b; let su = decode_fp8((S_u32[su_idx >> 2u] >> ((su_idx & 3u) * 8u)) & 0xFFu);
            let b4 = b * 4u;
            { let xb=ts[0]*576u+b4; let x0=X[xb]; let x1=X[xb+1u]; let x2=X[xb+2u]; let x3=X[xb+3u];
              ag0 += (dot8_nvfp4(wg_w.x,x0,x1)+dot8_nvfp4(wg_w.y,x2,x3))*sg; au0 += (dot8_nvfp4(wu_w.x,x0,x1)+dot8_nvfp4(wu_w.y,x2,x3))*su; }
            if (tc > 1u) { let xb=ts[1]*576u+b4; let x0=X[xb]; let x1=X[xb+1u]; let x2=X[xb+2u]; let x3=X[xb+3u];
              ag1 += (dot8_nvfp4(wg_w.x,x0,x1)+dot8_nvfp4(wg_w.y,x2,x3))*sg; au1 += (dot8_nvfp4(wu_w.x,x0,x1)+dot8_nvfp4(wu_w.y,x2,x3))*su; }
            if (tc > 2u) { let xb=ts[2]*576u+b4; let x0=X[xb]; let x1=X[xb+1u]; let x2=X[xb+2u]; let x3=X[xb+3u];
              ag2 += (dot8_nvfp4(wg_w.x,x0,x1)+dot8_nvfp4(wg_w.y,x2,x3))*sg; au2 += (dot8_nvfp4(wu_w.x,x0,x1)+dot8_nvfp4(wu_w.y,x2,x3))*su; }
            if (tc > 3u) { let xb=ts[3]*576u+b4; let x0=X[xb]; let x1=X[xb+1u]; let x2=X[xb+2u]; let x3=X[xb+3u];
              ag3 += (dot8_nvfp4(wg_w.x,x0,x1)+dot8_nvfp4(wg_w.y,x2,x3))*sg; au3 += (dot8_nvfp4(wu_w.x,x0,x1)+dot8_nvfp4(wu_w.y,x2,x3))*su; }
            if (tc > 4u) { let xb=ts[4]*576u+b4; let x0=X[xb]; let x1=X[xb+1u]; let x2=X[xb+2u]; let x3=X[xb+3u];
              ag4 += (dot8_nvfp4(wg_w.x,x0,x1)+dot8_nvfp4(wg_w.y,x2,x3))*sg; au4 += (dot8_nvfp4(wu_w.x,x0,x1)+dot8_nvfp4(wu_w.y,x2,x3))*su; }
            if (tc > 5u) { let xb=ts[5]*576u+b4; let x0=X[xb]; let x1=X[xb+1u]; let x2=X[xb+2u]; let x3=X[xb+3u];
              ag5 += (dot8_nvfp4(wg_w.x,x0,x1)+dot8_nvfp4(wg_w.y,x2,x3))*sg; au5 += (dot8_nvfp4(wu_w.x,x0,x1)+dot8_nvfp4(wu_w.y,x2,x3))*su; }
            if (tc > 6u) { let xb=ts[6]*576u+b4; let x0=X[xb]; let x1=X[xb+1u]; let x2=X[xb+2u]; let x3=X[xb+3u];
              ag6 += (dot8_nvfp4(wg_w.x,x0,x1)+dot8_nvfp4(wg_w.y,x2,x3))*sg; au6 += (dot8_nvfp4(wu_w.x,x0,x1)+dot8_nvfp4(wu_w.y,x2,x3))*su; }
            if (tc > 7u) { let xb=ts[7]*576u+b4; let x0=X[xb]; let x1=X[xb+1u]; let x2=X[xb+2u]; let x3=X[xb+3u];
              ag7 += (dot8_nvfp4(wg_w.x,x0,x1)+dot8_nvfp4(wg_w.y,x2,x3))*sg; au7 += (dot8_nvfp4(wu_w.x,x0,x1)+dot8_nvfp4(wu_w.y,x2,x3))*su; }
        }
        var sg_arr = array<f32,8>(subgroupAdd(ag0),subgroupAdd(ag1),subgroupAdd(ag2),subgroupAdd(ag3),subgroupAdd(ag4),subgroupAdd(ag5),subgroupAdd(ag6),subgroupAdd(ag7));
        var su_arr = array<f32,8>(subgroupAdd(au0),subgroupAdd(au1),subgroupAdd(au2),subgroupAdd(au3),subgroupAdd(au4),subgroupAdd(au5),subgroupAdd(au6),subgroupAdd(au7));
        if (lane == 0u) {
            for (var j: u32 = 0u; j < tc; j++) {
                H[(ts[j] * 8u + sls[j]) * 1152u + r] = situ(sg_arr[j] * os, su_arr[j] * os);
            }
        }
    }
}
"""

# 7. EXPERT-GROUPED NVFP4 W2: reads W2 ONCE per active expert for up to 8 tokens, writes MoEPart[tok, slot, row]!
# Dispatch: (1152, 128, 1) -> wg.y = eid (0..127)!
P_NV_W2_GRP = FAST_COMMON + """
@group(0) @binding(0) var<storage, read> W_v2: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> S_u32: array<u32>;
@group(0) @binding(2) var<storage, read> Outer: array<f32>;
@group(0) @binding(3) var<storage, read> ExpCnt: array<u32>;
@group(0) @binding(4) var<storage, read> ExpToks: array<u32>;
@group(0) @binding(5) var<storage, read> Weights: array<f32>;      // [C, 8]
@group(0) @binding(6) var<storage, read> H: array<vec4<f32>>;      // [C, 8, 288]
@group(0) @binding(7) var<storage, read_write> MoEPart: array<f32>;// [C, 8, 2304]

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let pk_ec = ExpCnt[wg.y];
    let cnt = pk_ec & 0xFFu;
    if (cnt == 0u) { return; }
    let eid = pk_ec >> 8u;
    let tid = lid.x; let sub = tid >> 6u; let lane = tid & 63u;
    let row = wg.x * 2u + sub;
    let w_base = (eid * 2304u + row) * 72u;
    let os = Outer[eid];

    for (var t0: u32 = 0u; t0 < cnt; t0 += 8u) {
        let tc = min(8u, cnt - t0);
        var ts = array<u32, 8>(0u,0u,0u,0u,0u,0u,0u,0u);
        var sls = array<u32, 8>(0u,0u,0u,0u,0u,0u,0u,0u);
        for (var j: u32 = 0u; j < tc; j++) {
            let pk = ExpToks[eid * 64u + t0 + j];
            ts[j] = pk >> 3u; sls[j] = pk & 7u;
        }
        var a0: f32=0.0; var a1: f32=0.0; var a2: f32=0.0; var a3: f32=0.0;
        var a4: f32=0.0; var a5: f32=0.0; var a6: f32=0.0; var a7: f32=0.0;

        for (var b = lane; b < 72u; b += 64u) {
            let w = W_v2[w_base + b];
            let s_idx = w_base + b;
            let s = decode_fp8((S_u32[s_idx >> 2u] >> ((s_idx & 3u) * 8u)) & 0xFFu);
            let b4 = b * 4u;
            { let hb=(ts[0]*8u+sls[0])*288u+b4; a0 += (dot8_nvfp4(w.x,H[hb],H[hb+1u])+dot8_nvfp4(w.y,H[hb+2u],H[hb+3u]))*s; }
            if (tc > 1u) { let hb=(ts[1]*8u+sls[1])*288u+b4; a1 += (dot8_nvfp4(w.x,H[hb],H[hb+1u])+dot8_nvfp4(w.y,H[hb+2u],H[hb+3u]))*s; }
            if (tc > 2u) { let hb=(ts[2]*8u+sls[2])*288u+b4; a2 += (dot8_nvfp4(w.x,H[hb],H[hb+1u])+dot8_nvfp4(w.y,H[hb+2u],H[hb+3u]))*s; }
            if (tc > 3u) { let hb=(ts[3]*8u+sls[3])*288u+b4; a3 += (dot8_nvfp4(w.x,H[hb],H[hb+1u])+dot8_nvfp4(w.y,H[hb+2u],H[hb+3u]))*s; }
            if (tc > 4u) { let hb=(ts[4]*8u+sls[4])*288u+b4; a4 += (dot8_nvfp4(w.x,H[hb],H[hb+1u])+dot8_nvfp4(w.y,H[hb+2u],H[hb+3u]))*s; }
            if (tc > 5u) { let hb=(ts[5]*8u+sls[5])*288u+b4; a5 += (dot8_nvfp4(w.x,H[hb],H[hb+1u])+dot8_nvfp4(w.y,H[hb+2u],H[hb+3u]))*s; }
            if (tc > 6u) { let hb=(ts[6]*8u+sls[6])*288u+b4; a6 += (dot8_nvfp4(w.x,H[hb],H[hb+1u])+dot8_nvfp4(w.y,H[hb+2u],H[hb+3u]))*s; }
            if (tc > 7u) { let hb=(ts[7]*8u+sls[7])*288u+b4; a7 += (dot8_nvfp4(w.x,H[hb],H[hb+1u])+dot8_nvfp4(w.y,H[hb+2u],H[hb+3u]))*s; }
        }
        var sa = array<f32,8>(subgroupAdd(a0),subgroupAdd(a1),subgroupAdd(a2),subgroupAdd(a3),subgroupAdd(a4),subgroupAdd(a5),subgroupAdd(a6),subgroupAdd(a7));
        if (lane == 0u) {
            for (var j: u32 = 0u; j < tc; j++) {
                let idx8 = ts[j] * 8u + sls[j];
                MoEPart[idx8 * 2304u + row] = sa[j] * (os * Weights[idx8]);
            }
        }
    }
}
"""

# 8. EXPERT-GROUPED MXFP8 W13 + SiTU: Dispatch (576, 128, 1)
P_MX_W13_GRP = FAST_COMMON + """
@group(0) @binding(0) var<storage, read> W_v4: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> S_u32: array<u32>;
@group(0) @binding(2) var<storage, read> ExpCnt: array<u32>;
@group(0) @binding(3) var<storage, read> ExpToks: array<u32>;
@group(0) @binding(4) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> H: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let pk_ec = ExpCnt[wg.y];
    let cnt = pk_ec & 0xFFu;
    if (cnt == 0u) { return; }
    let eid = pk_ec >> 8u;
    let tid = lid.x; let sub = tid >> 6u; let lane = tid & 63u;
    let r = wg.x * 2u + sub; let row_g = 2u * r;
    let wg_base = (eid * 2304u + row_g) * 144u; let wu_base = wg_base + 144u;
    let sg_base = (eid * 2304u + row_g) * 72u;  let su_base = sg_base + 72u;

    for (var t0: u32 = 0u; t0 < cnt; t0 += 8u) {
        let tc = min(8u, cnt - t0);
        var ts = array<u32, 8>(0u,0u,0u,0u,0u,0u,0u,0u);
        var sls = array<u32, 8>(0u,0u,0u,0u,0u,0u,0u,0u);
        for (var j: u32 = 0u; j < tc; j++) { let pk = ExpToks[eid * 64u + t0 + j]; ts[j] = pk >> 3u; sls[j] = pk & 7u; }
        var ag0: f32=0.0; var au0: f32=0.0; var ag1: f32=0.0; var au1: f32=0.0;
        var ag2: f32=0.0; var au2: f32=0.0; var ag3: f32=0.0; var au3: f32=0.0;
        var ag4: f32=0.0; var au4: f32=0.0; var ag5: f32=0.0; var au5: f32=0.0;
        var ag6: f32=0.0; var au6: f32=0.0; var ag7: f32=0.0; var au7: f32=0.0;

        for (var v = lane; v < 144u; v += 64u) {
            let wg_w = W_v4[wg_base + v]; let wu_w = W_v4[wu_base + v];
            let b = v >> 1u;
            let sg_idx = sg_base + b; let sg = decode_ue8m0((S_u32[sg_idx >> 2u] >> ((sg_idx & 3u) * 8u)) & 0xFFu);
            let su_idx = su_base + b; let su = decode_ue8m0((S_u32[su_idx >> 2u] >> ((su_idx & 3u) * 8u)) & 0xFFu);
            let v4 = v * 4u;
            { let xb=ts[0]*576u+v4; let x0=X[xb]; let x1=X[xb+1u]; let x2=X[xb+2u]; let x3=X[xb+3u];
              ag0 += (dot4_fp8(wg_w.x,x0)+dot4_fp8(wg_w.y,x1)+dot4_fp8(wg_w.z,x2)+dot4_fp8(wg_w.w,x3))*sg;
              au0 += (dot4_fp8(wu_w.x,x0)+dot4_fp8(wu_w.y,x1)+dot4_fp8(wu_w.z,x2)+dot4_fp8(wu_w.w,x3))*su; }
            if (tc > 1u) { let xb=ts[1]*576u+v4; let x0=X[xb]; let x1=X[xb+1u]; let x2=X[xb+2u]; let x3=X[xb+3u];
              ag1 += (dot4_fp8(wg_w.x,x0)+dot4_fp8(wg_w.y,x1)+dot4_fp8(wg_w.z,x2)+dot4_fp8(wg_w.w,x3))*sg;
              au1 += (dot4_fp8(wu_w.x,x0)+dot4_fp8(wu_w.y,x1)+dot4_fp8(wu_w.z,x2)+dot4_fp8(wu_w.w,x3))*su; }
            if (tc > 2u) { let xb=ts[2]*576u+v4; let x0=X[xb]; let x1=X[xb+1u]; let x2=X[xb+2u]; let x3=X[xb+3u];
              ag2 += (dot4_fp8(wg_w.x,x0)+dot4_fp8(wg_w.y,x1)+dot4_fp8(wg_w.z,x2)+dot4_fp8(wg_w.w,x3))*sg;
              au2 += (dot4_fp8(wu_w.x,x0)+dot4_fp8(wu_w.y,x1)+dot4_fp8(wu_w.z,x2)+dot4_fp8(wu_w.w,x3))*su; }
            if (tc > 3u) { let xb=ts[3]*576u+v4; let x0=X[xb]; let x1=X[xb+1u]; let x2=X[xb+2u]; let x3=X[xb+3u];
              ag3 += (dot4_fp8(wg_w.x,x0)+dot4_fp8(wg_w.y,x1)+dot4_fp8(wg_w.z,x2)+dot4_fp8(wg_w.w,x3))*sg;
              au3 += (dot4_fp8(wu_w.x,x0)+dot4_fp8(wu_w.y,x1)+dot4_fp8(wu_w.z,x2)+dot4_fp8(wu_w.w,x3))*su; }
            if (tc > 4u) { let xb=ts[4]*576u+v4; let x0=X[xb]; let x1=X[xb+1u]; let x2=X[xb+2u]; let x3=X[xb+3u];
              ag4 += (dot4_fp8(wg_w.x,x0)+dot4_fp8(wg_w.y,x1)+dot4_fp8(wg_w.z,x2)+dot4_fp8(wg_w.w,x3))*sg;
              au4 += (dot4_fp8(wu_w.x,x0)+dot4_fp8(wu_w.y,x1)+dot4_fp8(wu_w.z,x2)+dot4_fp8(wu_w.w,x3))*su; }
            if (tc > 5u) { let xb=ts[5]*576u+v4; let x0=X[xb]; let x1=X[xb+1u]; let x2=X[xb+2u]; let x3=X[xb+3u];
              ag5 += (dot4_fp8(wg_w.x,x0)+dot4_fp8(wg_w.y,x1)+dot4_fp8(wg_w.z,x2)+dot4_fp8(wg_w.w,x3))*sg;
              au5 += (dot4_fp8(wu_w.x,x0)+dot4_fp8(wu_w.y,x1)+dot4_fp8(wu_w.z,x2)+dot4_fp8(wu_w.w,x3))*su; }
            if (tc > 6u) { let xb=ts[6]*576u+v4; let x0=X[xb]; let x1=X[xb+1u]; let x2=X[xb+2u]; let x3=X[xb+3u];
              ag6 += (dot4_fp8(wg_w.x,x0)+dot4_fp8(wg_w.y,x1)+dot4_fp8(wg_w.z,x2)+dot4_fp8(wg_w.w,x3))*sg;
              au6 += (dot4_fp8(wu_w.x,x0)+dot4_fp8(wu_w.y,x1)+dot4_fp8(wu_w.z,x2)+dot4_fp8(wu_w.w,x3))*su; }
            if (tc > 7u) { let xb=ts[7]*576u+v4; let x0=X[xb]; let x1=X[xb+1u]; let x2=X[xb+2u]; let x3=X[xb+3u];
              ag7 += (dot4_fp8(wg_w.x,x0)+dot4_fp8(wg_w.y,x1)+dot4_fp8(wg_w.z,x2)+dot4_fp8(wg_w.w,x3))*sg;
              au7 += (dot4_fp8(wu_w.x,x0)+dot4_fp8(wu_w.y,x1)+dot4_fp8(wu_w.z,x2)+dot4_fp8(wu_w.w,x3))*su; }
        }
        var sg_arr = array<f32,8>(subgroupAdd(ag0),subgroupAdd(ag1),subgroupAdd(ag2),subgroupAdd(ag3),subgroupAdd(ag4),subgroupAdd(ag5),subgroupAdd(ag6),subgroupAdd(ag7));
        var su_arr = array<f32,8>(subgroupAdd(au0),subgroupAdd(au1),subgroupAdd(au2),subgroupAdd(au3),subgroupAdd(au4),subgroupAdd(au5),subgroupAdd(au6),subgroupAdd(au7));
        if (lane == 0u) {
            for (var j: u32 = 0u; j < tc; j++) { H[(ts[j] * 8u + sls[j]) * 1152u + r] = situ(sg_arr[j], su_arr[j]); }
        }
    }
}
"""

# 9. EXPERT-GROUPED MXFP8 W2: Dispatch (1152, 128, 1)
P_MX_W2_GRP = FAST_COMMON + """
@group(0) @binding(0) var<storage, read> W_v4: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> S_u32: array<u32>;
@group(0) @binding(2) var<storage, read> ExpCnt: array<u32>;
@group(0) @binding(3) var<storage, read> ExpToks: array<u32>;
@group(0) @binding(4) var<storage, read> Weights: array<f32>;
@group(0) @binding(5) var<storage, read> H: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> MoEPart: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let pk_ec = ExpCnt[wg.y];
    let cnt = pk_ec & 0xFFu;
    if (cnt == 0u) { return; }
    let eid = pk_ec >> 8u;
    let tid = lid.x; let sub = tid >> 6u; let lane = tid & 63u;
    let row = wg.x * 2u + sub;
    let w_base = (eid * 2304u + row) * 72u;
    let s_base = (eid * 2304u + row) * 36u;

    for (var t0: u32 = 0u; t0 < cnt; t0 += 8u) {
        let tc = min(8u, cnt - t0);
        var ts = array<u32, 8>(0u,0u,0u,0u,0u,0u,0u,0u);
        var sls = array<u32, 8>(0u,0u,0u,0u,0u,0u,0u,0u);
        for (var j: u32 = 0u; j < tc; j++) { let pk = ExpToks[eid * 64u + t0 + j]; ts[j] = pk >> 3u; sls[j] = pk & 7u; }
        var a0: f32=0.0; var a1: f32=0.0; var a2: f32=0.0; var a3: f32=0.0;
        var a4: f32=0.0; var a5: f32=0.0; var a6: f32=0.0; var a7: f32=0.0;

        for (var v = lane; v < 72u; v += 64u) {
            let w = W_v4[w_base + v];
            let s_idx = s_base + (v >> 1u);
            let s = decode_ue8m0((S_u32[s_idx >> 2u] >> ((s_idx & 3u) * 8u)) & 0xFFu);
            let v4 = v * 4u;
            { let hb=(ts[0]*8u+sls[0])*288u+v4; a0 += (dot4_fp8(w.x,H[hb])+dot4_fp8(w.y,H[hb+1u])+dot4_fp8(w.z,H[hb+2u])+dot4_fp8(w.w,H[hb+3u]))*s; }
            if (tc > 1u) { let hb=(ts[1]*8u+sls[1])*288u+v4; a1 += (dot4_fp8(w.x,H[hb])+dot4_fp8(w.y,H[hb+1u])+dot4_fp8(w.z,H[hb+2u])+dot4_fp8(w.w,H[hb+3u]))*s; }
            if (tc > 2u) { let hb=(ts[2]*8u+sls[2])*288u+v4; a2 += (dot4_fp8(w.x,H[hb])+dot4_fp8(w.y,H[hb+1u])+dot4_fp8(w.z,H[hb+2u])+dot4_fp8(w.w,H[hb+3u]))*s; }
            if (tc > 3u) { let hb=(ts[3]*8u+sls[3])*288u+v4; a3 += (dot4_fp8(w.x,H[hb])+dot4_fp8(w.y,H[hb+1u])+dot4_fp8(w.z,H[hb+2u])+dot4_fp8(w.w,H[hb+3u]))*s; }
            if (tc > 4u) { let hb=(ts[4]*8u+sls[4])*288u+v4; a4 += (dot4_fp8(w.x,H[hb])+dot4_fp8(w.y,H[hb+1u])+dot4_fp8(w.z,H[hb+2u])+dot4_fp8(w.w,H[hb+3u]))*s; }
            if (tc > 5u) { let hb=(ts[5]*8u+sls[5])*288u+v4; a5 += (dot4_fp8(w.x,H[hb])+dot4_fp8(w.y,H[hb+1u])+dot4_fp8(w.z,H[hb+2u])+dot4_fp8(w.w,H[hb+3u]))*s; }
            if (tc > 6u) { let hb=(ts[6]*8u+sls[6])*288u+v4; a6 += (dot4_fp8(w.x,H[hb])+dot4_fp8(w.y,H[hb+1u])+dot4_fp8(w.z,H[hb+2u])+dot4_fp8(w.w,H[hb+3u]))*s; }
            if (tc > 7u) { let hb=(ts[7]*8u+sls[7])*288u+v4; a7 += (dot4_fp8(w.x,H[hb])+dot4_fp8(w.y,H[hb+1u])+dot4_fp8(w.z,H[hb+2u])+dot4_fp8(w.w,H[hb+3u]))*s; }
        }
        var sa = array<f32,8>(subgroupAdd(a0),subgroupAdd(a1),subgroupAdd(a2),subgroupAdd(a3),subgroupAdd(a4),subgroupAdd(a5),subgroupAdd(a6),subgroupAdd(a7));
        if (lane == 0u) {
            for (var j: u32 = 0u; j < tc; j++) {
                let idx8 = ts[j] * 8u + sls[j];
                MoEPart[idx8 * 2304u + row] = sa[j] * Weights[idx8];
            }
        }
    }
}
"""

# 10. Deterministic 8-slot sum -> MoEOut[C, 2304] (feeds directly into P_CONV_RESID + P_SHIFT_2304_HIST!):
# Dispatch: (9, C, 1)
P_MOE_SUM = """
@group(0) @binding(0) var<storage, read> MoEPart: array<f32>;  // [C, 8, 2304]
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;// [C, 2304]
@group(0) @binding(2) var<storage, read> Meta: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let bi = wg.y;
    if (bi >= Meta[0]) { return; }
    let c = wg.x * 256u + lid.x;
    if (c >= 2304u) { return; }
    let b8 = bi * 8u;
    var s: f32 = 0.0;
    for (var k: u32 = 0u; k < 8u; k++) { s += MoEPart[(b8 + k) * 2304u + c]; }
    Out[bi * 2304u + c] = s;
}
"""

PREFILL_SHADERS = [
    ("p_qkvr_post",    P_QKVR_POST),
    ("p_shift_kv",     P_SHIFT_KV_HIST),
    ("p_attn_sk",      P_ATTN_SK),
    ("p_conv_resid",   P_CONV_RESID),
    ("p_shift_2304",   P_SHIFT_2304_HIST),
    ("p_group_exp",    P_GROUP_EXPERTS),
    ("p_nv_w13_grp",   P_NV_W13_GRP),
    ("p_nv_w2_grp",    P_NV_W2_GRP),
    ("p_mx_w13_grp",   P_MX_W13_GRP),
    ("p_mx_w2_grp",    P_MX_W2_GRP),
    ("p_moe_sum",      P_MOE_SUM),
]

if __name__ == "__main__":
    out_dir = os.environ["DUMP_SPV_DIR"]
    ad = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    dev = ad.request_device_sync(required_features=["subgroup"])
    for name, code in PREFILL_SHADERS:
        m = dev.create_shader_module(code=code)
        dev.create_compute_pipeline(layout="auto", compute={"module": m, "entry_point": "main"})
    for i, (name, _) in enumerate(PREFILL_SHADERS):
        src = f"{out_dir}/shader_{i+2:02d}.spv"
        dst = f"{out_dir}/{name}.spv"
        shutil.copyfile(src, dst)
        raw = open(dst, "rb").read()
        assert raw[:4] == b"\x03\x02\x23\x07", f"Bad SPIR-V in {dst}"
    print(f"Compiled & verified all {len(PREFILL_SHADERS)} chunked-prefill SPIR-V shaders into {out_dir}!")
