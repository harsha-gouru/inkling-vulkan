import ctypes, glob, json, mmap, os, struct, time
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

FULL_ATTN = {5, 11, 17, 23}
MAX_BATCH = 16
SEQ_CAP   = 4096
SWA_WIN   = 1024
BUCKETS   = [1, 2, 4, 8, 16]
PREFILL_BUCKETS = [64, 16]  # cmd_slots 5, 6
MAX_SCRATCH = 64

_vk = ctypes.CDLL(str(BASE_DIR / "libvk_inkling.so"))
_vk.vk_init.restype = ctypes.c_int
_vk.vk_create_buffer.argtypes = [ctypes.c_uint64, ctypes.c_int, ctypes.c_int]
_vk.vk_create_buffer.restype = ctypes.c_int
_vk.vk_upload_buffer.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_uint64]
_vk.vk_upload_buffer.restype = ctypes.c_int
_vk.vk_zero_buffer.argtypes = [ctypes.c_int]
_vk.vk_zero_buffer.restype = ctypes.c_int
_vk.vk_zero_slice.argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64]
_vk.vk_zero_slice.restype = ctypes.c_int
_vk.vk_upload_slice.argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64]
_vk.vk_upload_slice.restype = ctypes.c_int
_vk.vk_download_slice.argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64]
_vk.vk_download_slice.restype = ctypes.c_int
_vk.vk_get_mapped.argtypes = [ctypes.c_int]
_vk.vk_get_mapped.restype = ctypes.c_void_p
_vk.vk_create_pipeline.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int, ctypes.c_uint32]
_vk.vk_create_pipeline.restype = ctypes.c_int
_vk.vk_record_begin_slot.argtypes = [ctypes.c_int]
_vk.vk_record_begin_slot.restype = ctypes.c_int
_vk.vk_record_dispatch_slot.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int]
_vk.vk_record_dispatch_slot.restype = ctypes.c_int
_vk.vk_record_end_slot.argtypes = [ctypes.c_int]
_vk.vk_record_end_slot.restype = ctypes.c_int
_vk.vk_run_recorded_slot.argtypes = [ctypes.c_int]
_vk.vk_run_recorded_slot.restype = ctypes.c_int
_vk.vk_shutdown.restype = None

PIPE_SPECS = [
    ("rms",        4, 1 << 3),
    ("copy",       3, 1 << 2),
    ("mix",        5, 1 << 4),
    ("mix_rms",    6, 1 << 5),
    ("rms3",       7, 0),
    ("ctrl_p1",    5, 1 << 4),
    ("ctrl_p2",    7, 1 << 6),
    ("ctrl3_p1",   5, 1 << 4),
    ("ctrl3_p2",   7, 1 << 6),
    ("gemm",       5, 1 << 4),
    ("qkvr_post", 15, 1 << 14),
    ("attn_sk",   10, 1 << 9),
    ("attn_cb",    4, 0),
    ("wo_conv",    8, 1 << 7),
    ("router",     4, 0),
    ("nv_w13",     6, 0),
    ("nv_w2_c",   12, 1 << 11),
    ("mx_w13",     5, 0),
    ("mx_w2_c",   11, 1 << 10),
    ("unembed_p1", 6, 0),
    ("argmax_p2",  5, 0),
]

PREFILL_PIPE_SPECS = [
    ("p_qkvr_post",  15, 1 << 14),
    ("p_shift_kv",    7, 1 << 6),
    ("p_attn_sk",    10, 1 << 9),
    ("p_conv_resid",  7, 1 << 6),
    ("p_shift_2304",  4, 1 << 3),
    ("p_group_exp",   4, 0),
    ("p_nv_w13_grp",  7, 0),
    ("p_nv_w2_grp",   8, 0),
    ("p_mx_w13_grp",  6, 0),
    ("p_mx_w2_grp",   7, 0),
    ("p_moe_sum",     3, 0),
]

class BatchedVulkanEngine:
    def __init__(self, path: str, spv_dir: str = None, prefill_spv_dir: str = None):
        if spv_dir is None:
            spv_dir = str(BASE_DIR / "spv_batch")
        if prefill_spv_dir is None:
            prefill_spv_dir = str(BASE_DIR / "spv_prefill")
        self.f = open(path, "rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        n = struct.unpack("<Q", self.mm[:8])[0]
        self.h = json.loads(self.mm[8 : 8 + n].decode("utf-8"))
        self.base = 8 + n

        es, ee = self.h["model.llm.embed.weight"]["data_offsets"]
        self.embed_u16 = np.frombuffer(memoryview(self.mm)[self.base + es : self.base + ee], dtype="<u2").reshape(self.h["model.llm.embed.weight"]["shape"])

        t0 = time.perf_counter()
        assert _vk.vk_init() == 0, "vk_init failed"
        self.vram_bytes = 0

        self.pipes = {}
        for name, n_bind, umask in PIPE_SPECS:
            spv = open(os.path.join(spv_dir, f"{name}.spv"), "rb").read()
            arr = np.frombuffer(spv, dtype=np.uint8)
            pid = _vk.vk_create_pipeline(arr.ctypes.data, len(spv), n_bind, umask)
            assert pid >= 0, f"vk_create_pipeline failed for {name}"
            self.pipes[name] = pid

        for name, n_bind, umask in PREFILL_PIPE_SPECS:
            spv = open(os.path.join(prefill_spv_dir, f"{name}.spv"), "rb").read()
            arr = np.frombuffer(spv, dtype=np.uint8)
            pid = _vk.vk_create_pipeline(arr.ctypes.data, len(spv), n_bind, umask)
            assert pid >= 0, f"vk_create_pipeline failed for {name}"
            self.pipes[name] = pid

        self._alloc_scratch()
        self._build_and_record()
        self.init_s = time.perf_counter() - t0

    def _raw(self, name):
        s, e = self.h[name]["data_offsets"]
        return memoryview(self.mm)[self.base + s : self.base + e]

    def _bf16_to_f32(self, name):
        u16 = np.frombuffer(self._raw(name), dtype="<u2")
        return (u16.astype(np.uint32) << 16).view(np.float32)

    def _mk(self, nbytes, host_visible=0, is_uniform=0):
        bid = _vk.vk_create_buffer(nbytes, host_visible, is_uniform)
        assert bid >= 0, f"vk_create_buffer failed ({nbytes})"
        self.vram_bytes += nbytes
        return bid

    def _up(self, data, host_visible=0, is_uniform=0):
        if isinstance(data, np.ndarray):
            arr = np.ascontiguousarray(data); nbytes = arr.nbytes; ptr = arr.ctypes.data
        elif isinstance(data, (bytes, bytearray)):
            arr = np.frombuffer(data, dtype=np.uint8); nbytes = len(data); ptr = arr.ctypes.data
        else:
            arr = np.frombuffer(data, dtype=np.uint8); nbytes = arr.nbytes; ptr = arr.ctypes.data
        bid = self._mk(nbytes, host_visible=host_visible, is_uniform=is_uniform)
        assert _vk.vk_upload_buffer(bid, ptr, nbytes) == 0
        return bid

    def _up_f32(self, name):
        dt = self.h[name]["dtype"]
        arr = np.frombuffer(self._raw(name), dtype="<f4") if dt == "F32" else self._bf16_to_f32(name)
        return self._up(arr)

    def _u4(self, a, b=0, c=0, d=0):
        return self._up(np.array([a, b, c, d], dtype=np.uint32), host_visible=1, is_uniform=1)

    def _alloc_scratch(self):
        B = MAX_BATCH
        S = MAX_SCRATCH
        # Host-mapped I/O buffers
        self.buf_meta    = self._mk(128 * 4, host_visible=1)
        self.buf_tmp     = self._mk(S * 2304 * 4, host_visible=1)
        self.buf_tok_out = self._mk(B * 2 * 4, host_visible=2)

        self.meta_arr = np.ctypeslib.as_array(ctypes.cast(_vk.vk_get_mapped(self.buf_meta), ctypes.POINTER(ctypes.c_uint32)), shape=(128,))
        self.tmp_arr  = np.ctypeslib.as_array(ctypes.cast(_vk.vk_get_mapped(self.buf_tmp),  ctypes.POINTER(ctypes.c_float)),  shape=(S, 2304))
        self.out_arr  = np.ctypeslib.as_array(ctypes.cast(_vk.vk_get_mapped(self.buf_tok_out), ctypes.POINTER(ctypes.c_uint32)), shape=(B, 2))

        self.buf_cur     = self._mk(S * 2304 * 4)
        self.buf_ffn_inp = self._mk(S * 2304 * 4)
        self.buf_prelude = self._mk(S * 2304 * 4)
        self.buf_blocks  = self._mk(S * 8 * 2304 * 4)
        self.buf_bank    = self._mk(S * 4 * 2304 * 4)
        self.buf_bank_out= self._mk(S * 2304 * 4)
        self.buf_msgs    = self._mk(S * 4 * 2304 * 4)
        self.buf_mixed3  = self._mk(S * 3 * 2304 * 4)
        self.buf_ctrl3   = self._mk(S * 3 * 2304 * 4)
        self.buf_hq      = self._mk(S * 2304 * 4)
        self.buf_hk      = self._mk(S * 2304 * 4)
        self.buf_hv      = self._mk(S * 2304 * 4)
        self.buf_gate3   = self._mk(S * 3 * 64 * 4)
        self.buf_qrsg    = self._mk(S * 8600 * 4)
        self.buf_kbuf    = self._mk(S * 1056 * 4)
        self.buf_vbuf    = self._mk(S * 1056 * 4)
        self.buf_attn_ctx= self._mk(S * 3072 * 4)
        self.buf_part_ml = self._mk(S * 24 * 16 * 2 * 4)
        self.buf_part_ac = self._mk(S * 24 * 16 * 128 * 4)
        self.buf_ffn_mix = self._mk(S * 2304 * 4)
        self.buf_ffn_ctl = self._mk(S * 2304 * 4)
        self.buf_ffn_nrm = self._mk(S * 2304 * 4)
        self.buf_rlogits = self._mk(S * 128 * 4)
        self.buf_eids    = self._mk(S * 8 * 4)
        self.buf_wts     = self._mk(S * 8 * 4)
        self.buf_moe_h   = self._mk(S * 8 * 1152 * 4)
        # Extra prefill-only scratch (~5.2 MiB total)
        self.buf_wo_raw  = self._mk(S * 2304 * 4)
        self.buf_moe_part= self._mk(S * 8 * 2304 * 4)
        self.buf_exp_cnt = self._mk(128 * 4)
        self.buf_exp_toks= self._mk(128 * 64 * 4)

        # Split KV caches (each <= 1.25 GiB, under Vulkan 2 GiB binding limit!):
        # 40 SWA ring-buffer layers (cap=1024): 8 users x 40 x 1024 x 1024 x 4B = 1.25 GiB per half
        #  8 Full-Attn layers      (cap=4096): 8 users x  8 x 4096 x 1024 x 4B = 1.00 GiB per half
        swa_half_bytes  = (B // 2) * 40 * SWA_WIN * 1024 * 4
        full_half_bytes = (B // 2) *  8 * SEQ_CAP * 1024 * 4
        self.buf_k_swa0  = self._mk(swa_half_bytes)
        self.buf_k_swa1  = self._mk(swa_half_bytes)
        self.buf_v_swa0  = self._mk(swa_half_bytes)
        self.buf_v_swa1  = self._mk(swa_half_bytes)
        self.buf_k_full0 = self._mk(full_half_bytes)
        self.buf_k_full1 = self._mk(full_half_bytes)
        self.buf_v_full0 = self._mk(full_half_bytes)
        self.buf_v_full1 = self._mk(full_half_bytes)
        self.buf_logits  = self._mk(B * 200058 * 4)
        self.buf_hist_k  = self._mk(B * 48 * 3 * 1024 * 4)
        self.buf_hist_v  = self._mk(B * 48 * 3 * 1024 * 4)
        self.buf_hist_a  = self._mk(B * 48 * 3 * 2304 * 4)
        self.buf_hist_m  = self._mk(B * 48 * 3 * 2304 * 4)

        self.n_unembed_wg = (200058 + 3) // 4  # 50015
        self.buf_part_v  = self._mk(B * self.n_unembed_wg * 4)
        self.buf_part_i  = self._mk(B * self.n_unembed_wg * 4)

    def _build_and_record(self):
        phys = []
        for ip in range(24):
            p = f"model.llm.layers.{ip}"
            is_swa = ip not in FULL_ATTN
            L = {"is_swa": is_swa}
            if ip > 0: L["attn_q"] = self._up_f32(f"model.llm.attn_res.attn_queries.{ip - 1}")
            L["mlp_q"] = self._up_f32(f"model.llm.attn_res.mlp_queries.{ip}")
            if is_swa:
                L["attn_norm"] = self._up_f32(f"{p}.attn_norm.weight")
                L["k_sconv"]   = self._up_f32(f"{p}.attn.k_sconv.weight")
                L["v_sconv"]   = self._up_f32(f"{p}.attn.v_sconv.weight")
            else:
                L["q_norm"]    = self._up_f32(f"{p}.q_input_norm.weight")
                L["k_norm"]    = self._up_f32(f"{p}.k_input_norm.weight")
                L["v_norm"]    = self._up_f32(f"{p}.v_input_norm.weight")
                L["k_sconv"]   = self._up_f32(f"{p}.attn.k_sconv.static_w")
                L["v_sconv"]   = self._up_f32(f"{p}.attn.v_sconv.static_w")

            b_wq = bytes(self._raw(f"{p}.attn.wq_du.weight"))
            b_wr = bytes(self._raw(f"{p}.attn.wr_du.weight"))
            b_ws = bytes(self._raw(f"{p}.attn.attn_sink_proj.weight"))
            b_wg = bytes(self._raw(f"{p}.attn.gate_attn.weight"))
            b_wk = bytes(self._raw(f"{p}.attn.wk_dv.weight"))
            b_wv = bytes(self._raw(f"{p}.attn.wv_dv.weight"))
            if is_swa:
                if ip < 6 or ip >= 18:
                    L["w_qrsgkv"] = self._up(b_wq + b_wr + b_ws + b_wg + b_wk + b_wv)
                else:
                    L["w_qrsg"] = self._up(b_wq + b_wr + b_ws + b_wg)
                    L["wk"]     = self._up(b_wk)
                    L["wv"]     = self._up(b_wv)
            else:
                L["w_qrsg"] = self._up(b_wq + b_wr + b_ws + b_wg)
                L["wk_dyn"] = self._up(b_wk + bytes(self._raw(f"{p}.attn.k_sconv.weight_proj")))
                L["wv_dyn"] = self._up(b_wv + bytes(self._raw(f"{p}.attn.v_sconv.weight_proj")))

            L["wo"]        = self._up(self._raw(f"{p}.attn.wo_ud.weight"))
            L["q_clip"]    = self._up_f32(f"{p}.attn.q_l2_clip.scale")
            L["r_clip"]    = self._up_f32(f"{p}.attn.rel_l2_clip.scale")
            L["rel_bias"]  = self._up_f32(f"{p}.attn.rel_logits_proj.proj")
            L["attn_sconv"]= self._up_f32(f"{p}.attn_sconv.weight")
            L["mlp_norm"]  = self._up_f32(f"{p}.mlp_norm.weight")
            L["router_w"]  = self._up(self._raw(f"{p}.mlp.gate.weight"))
            L["router_b"]  = self._up_f32(f"{p}.mlp.gate.bias")
            L["mlp_sconv"] = self._up_f32(f"{p}.mlp_sconv.weight")

            ep = f"{p}.mlp.experts"
            L["w13"] = self._up(self._raw(f"{ep}.w13_weight"))
            L["s13"] = self._up(self._raw(f"{ep}.w13_weight_scale"))
            L["w2"]  = self._up(self._raw(f"{ep}.w2_weight"))
            L["s2"]  = self._up(self._raw(f"{ep}.w2_weight_scale"))
            if self.h[f"{ep}.w13_weight"]["dtype"] == "U8":
                L["exp_kind"] = "nvfp4"
                a13 = np.frombuffer(self._raw(f"{ep}.w13_weight_amax"), dtype="<f4")
                d13 = float(np.frombuffer(self._raw(f"{ep}.w13_weight_outer_scale_denominator"), dtype="<f4")[0])
                a2  = np.frombuffer(self._raw(f"{ep}.w2_weight_amax"), dtype="<f4")
                d2  = float(np.frombuffer(self._raw(f"{ep}.w2_weight_outer_scale_denominator"), dtype="<f4")[0])
                L["o13"] = self._up((a13 / d13).astype(np.float32))
                L["o2"]  = self._up((a2 / d2).astype(np.float32))
            else:
                L["exp_kind"] = "mxfp8"

            if 6 <= ip < 18:
                d3 = b"".join(bytes(self._raw(f"{p}.controllers.{c}.down_weight"))   for c in ("attn", "k", "v"))
                g3 = b"".join(bytes(self._raw(f"{p}.controllers.{c}.gain_weight"))   for c in ("attn", "k", "v"))
                f3 = b"".join(bytes(self._raw(f"{p}.controllers.{c}.forget_weight")) for c in ("attn", "k", "v"))
                L["ctrl3"] = (self._up(d3), self._up(g3), self._up(f3))
                L["ctrl_mlp"] = (
                    self._up(self._raw(f"{p}.controllers.mlp.down_weight")),
                    self._up(self._raw(f"{p}.controllers.mlp.gain_weight")),
                    self._up(self._raw(f"{p}.controllers.mlp.forget_weight")),
                )
            phys.append(L)

        cp_q  = np.frombuffer(self._raw("model.llm.middle_cycle_cross_pass_queries"), dtype="<f4").reshape(24, 2304)
        cp_kv = np.frombuffer(self._raw("model.llm.middle_cycle_cross_pass_kv_queries"), dtype="<f4").reshape(24, 2304)
        msg_q_bufs = [self._up(np.stack([cp_q[2*i], cp_kv[2*i], cp_kv[2*i+1], cp_q[2*i+1]], axis=0).astype(np.float32)) for i in range(12)]

        b_embed_norm = self._up_f32("model.llm.embed_norm.weight")
        b_out_q      = self._up_f32("model.llm.attn_res.output_query")
        b_out_norm   = self._up_f32("model.llm.norm.weight")
        b_unembed    = self._up(self._raw("model.llm.unembed.weight"))

        # Shared uniforms
        u_rms_2304 = self._u4(0, 576)
        u_gemm_8600 = self._u4(288, 8600, 8600)
        u_gemm_6552 = self._u4(288, 6552, 8600)
        u_gemm_1024 = self._u4(288, 1024, 1056)
        u_gemm_1056 = self._u4(288, 1056, 1056)
        u_gemm_128  = self._u4(288, 128, 128)
        u_cp_00 = self._u4(0, 0, 2304, 2304)

        swa_map, full_map = {}, {}
        si, fi = 0, 0
        for sl in range(48):
            phys_ip = sl if sl < 24 else 6 + ((sl - 24) % 12)
            if phys_ip in FULL_ATTN:
                full_map[sl] = fi; fi += 1
            else:
                swa_map[sl] = si; si += 1
        assert si == 40 and fi == 8

        # Pre-record one command buffer per bucket B in [1, 2, 4, 8, 16]!
        for cmd_slot, B in enumerate(BUCKETS):
            assert _vk.vk_record_begin_slot(cmd_slot) == 0
            bt8 = (B + 7) // 8

            def emit(pname, bufs, wg, barrier=1):
                arr = (ctypes.c_int * len(bufs))(*bufs)
                assert _vk.vk_record_dispatch_slot(cmd_slot, self.pipes[pname], len(bufs), arr, wg[0], wg[1], wg[2], barrier) == 0

            emit("rms", [self.buf_tmp, b_embed_norm, self.buf_cur, u_rms_2304], (1, B, 1))

            def emit_layer(ip, slot, nb, has_msg):
                L = phys[ip]; is_swa = L["is_swa"]
                resid_buf = self.buf_cur

                if (not has_msg) and is_swa and ip > 0:
                    emit("mix_rms", [self.buf_blocks, self.buf_cur, L["attn_q"], L["attn_norm"], self.buf_hq, self._u4(8, nb)], (1, B, 1))
                    hk_buf = hv_buf = self.buf_hq
                else:
                    if ip > 0:
                        n_q = 1 if is_swa else 3
                        emit("mix", [self.buf_blocks, self.buf_cur, L["attn_q"], self.buf_mixed3, self._u4(8, nb, n_q, 3)], (1, B, 1))
                        in_buf = self.buf_mixed3
                        in_stride = 1728
                    else:
                        in_buf = self.buf_cur
                        in_stride = 576

                    if has_msg:
                        d3, g3, f3 = L["ctrl3"]
                        c_stride = 0 if is_swa else 2304
                        emit("ctrl3_p1", [in_buf, self.buf_msgs, d3, self.buf_gate3, self._u4(c_stride // 2)], (64, 3, B))
                        emit("ctrl3_p2", [in_buf, self.buf_msgs, self.buf_gate3, g3, f3, self.buf_ctrl3, self._u4(c_stride)], (36, 3, B))
                        if is_swa:
                            nw = L["attn_norm"]
                            emit("rms3", [self.buf_ctrl3, nw, nw, nw, self.buf_hq, self.buf_hk, self.buf_hv], (3, B, 1))
                        else:
                            emit("rms3", [self.buf_ctrl3, L["q_norm"], L["k_norm"], L["v_norm"], self.buf_hq, self.buf_hk, self.buf_hv], (3, B, 1))
                        hk_buf, hv_buf = self.buf_hk, self.buf_hv
                    else:
                        if is_swa:
                            emit("rms", [in_buf, L["attn_norm"], self.buf_hq, self._u4(0, in_stride)], (1, B, 1))
                            hk_buf = hv_buf = self.buf_hq
                        else:
                            emit("rms3", [in_buf, L["q_norm"], L["k_norm"], L["v_norm"], self.buf_hq, self.buf_hk, self.buf_hv], (3, B, 1))
                            hk_buf, hv_buf = self.buf_hk, self.buf_hv

                if "w_qrsgkv" in L:
                    emit("gemm", [L["w_qrsgkv"], self.buf_hq, self.buf_qrsg, self.buf_meta, u_gemm_8600], (4300, bt8, 1), barrier=1)
                    k_src, v_src, k_off, v_off = self.buf_qrsg, self.buf_qrsg, 6552, 7576
                elif is_swa:
                    emit("gemm", [L["w_qrsg"], self.buf_hq, self.buf_qrsg, self.buf_meta, u_gemm_6552], (3276, bt8, 1), barrier=0)
                    emit("gemm", [L["wk"],     hk_buf,      self.buf_kbuf, self.buf_meta, u_gemm_1024], (512,  bt8, 1), barrier=0)
                    emit("gemm", [L["wv"],     hv_buf,      self.buf_vbuf, self.buf_meta, u_gemm_1024], (512,  bt8, 1), barrier=1)
                    k_src, v_src, k_off, v_off = self.buf_kbuf, self.buf_vbuf, 0, 0
                else:
                    emit("gemm", [L["w_qrsg"], self.buf_hq, self.buf_qrsg, self.buf_meta, u_gemm_6552], (3276, bt8, 1), barrier=0)
                    emit("gemm", [L["wk_dyn"], hk_buf,      self.buf_kbuf, self.buf_meta, u_gemm_1056], (528,  bt8, 1), barrier=0)
                    emit("gemm", [L["wv_dyn"], hv_buf,      self.buf_vbuf, self.buf_meta, u_gemm_1056], (528,  bt8, 1), barrier=1)
                    k_src, v_src, k_off, v_off = self.buf_kbuf, self.buf_vbuf, 0, 0

                c_idx = swa_map[slot] if is_swa else full_map[slot]
                k0_b, k1_b = (self.buf_k_swa0, self.buf_k_swa1) if is_swa else (self.buf_k_full0, self.buf_k_full1)
                v0_b, v1_b = (self.buf_v_swa0, self.buf_v_swa1) if is_swa else (self.buf_v_full0, self.buf_v_full1)
                emit("qkvr_post", [
                    self.buf_qrsg,
                    self.buf_kbuf if k_off > 0 else k_src,
                    self.buf_vbuf if v_off > 0 else v_src,
                    L["q_clip"], L["r_clip"], L["k_sconv"], L["v_sconv"],
                    self.buf_hist_k, self.buf_hist_v, k0_b, k1_b, v0_b, v1_b,
                    self.buf_meta, self._u4((slot << 8) | c_idx, 0 if is_swa else 1, k_off, v_off)
                ], (64, B, 1))

                emit("attn_sk", [self.buf_qrsg, L["rel_bias"], k0_b, k1_b, v0_b, v1_b, self.buf_part_ml, self.buf_part_ac, self.buf_meta, self._u4(c_idx, 1 if is_swa else 0)], (24, 16, B))
                emit("attn_cb", [self.buf_part_ml, self.buf_part_ac, self.buf_qrsg, self.buf_attn_ctx], (24, B, 1))

                add_r = 0 if ip % 6 == 0 else 1
                if ip % 6 == 0:
                    emit("copy", [resid_buf, self.buf_blocks, self._u4(0, nb * 2304, 2304, 8 * 2304)], (9, B, 1), barrier=0)
                    nb += 1
                emit("wo_conv", [L["wo"], self.buf_attn_ctx, L["attn_sconv"], resid_buf, self.buf_hist_a, self.buf_ffn_inp, self.buf_meta, self._u4(slot, add_r)], (1152, bt8, 1))

                if has_msg:
                    emit("mix", [self.buf_blocks, self.buf_ffn_inp, L["mlp_q"], self.buf_ffn_mix, self._u4(8, nb, 1, 1)], (1, B, 1))
                    wd, wg_w, wf_w = L["ctrl_mlp"]
                    emit("ctrl_p1", [self.buf_ffn_mix, self.buf_msgs, wd, self.buf_gate3, self._u4(0, 3 * 1152)], (64, 1, B))
                    emit("ctrl_p2", [self.buf_ffn_mix, self.buf_msgs, self.buf_gate3, wg_w, wf_w, self.buf_ffn_ctl, self._u4(0, 3 * 2304, 0)], (36, 1, B))
                    emit("rms", [self.buf_ffn_ctl, L["mlp_norm"], self.buf_ffn_nrm, u_rms_2304], (1, B, 1))
                else:
                    emit("mix_rms", [self.buf_blocks, self.buf_ffn_inp, L["mlp_q"], L["mlp_norm"], self.buf_ffn_nrm, self._u4(8, nb)], (1, B, 1))

                emit("gemm",   [L["router_w"], self.buf_ffn_nrm, self.buf_rlogits, self.buf_meta, u_gemm_128], (64, bt8, 1))
                emit("router", [self.buf_rlogits, L["router_b"], self.buf_eids, self.buf_wts], (1, B, 1))
                if L["exp_kind"] == "nvfp4":
                    emit("nv_w13",  [L["w13"], L["s13"], L["o13"], self.buf_eids, self.buf_ffn_nrm, self.buf_moe_h], (576, 8, B))
                    emit("nv_w2_c", [L["w2"],  L["s2"],  L["o2"],  self.buf_eids, self.buf_wts, self.buf_moe_h, L["mlp_sconv"], self.buf_ffn_inp, self.buf_hist_m, self.buf_cur, self.buf_meta, self._u4(slot)], (1152, B, 1))
                else:
                    emit("mx_w13",  [L["w13"], L["s13"], self.buf_eids, self.buf_ffn_nrm, self.buf_moe_h], (576, 8, B))
                    emit("mx_w2_c", [L["w2"],  L["s2"],  self.buf_eids, self.buf_wts, self.buf_moe_h, L["mlp_sconv"], self.buf_ffn_inp, self.buf_hist_m, self.buf_cur, self.buf_meta, self._u4(slot)], (1152, B, 1))
                return nb

            nb = 0
            for il in range(6): nb = emit_layer(il, il, nb, False)
            prelude_nb = nb
            emit("copy", [self.buf_cur, self.buf_prelude, u_cp_00], (9, B, 1))

            for p_idx in range(3):
                if p_idx > 0:
                    n_bank = nb - (prelude_nb + 1)
                    for b in range(n_bank):
                        emit("copy", [self.buf_blocks, self.buf_bank, self._u4((prelude_nb + 1 + b) * 2304, b * 2304, 8 * 2304, 4 * 2304)], (9, B, 1), barrier=0)
                    emit("copy", [self.buf_cur, self.buf_bank_out, u_cp_00], (9, B, 1), barrier=1)
                    emit("copy", [self.buf_prelude, self.buf_cur, u_cp_00], (9, B, 1), barrier=1)
                    nb = prelude_nb
                for ip in range(6, 18):
                    slot = ip if p_idx == 0 else 24 + (p_idx - 1) * 12 + (ip - 6)
                    if p_idx > 0:
                        emit("mix", [self.buf_bank, self.buf_bank_out, msg_q_bufs[ip - 6], self.buf_msgs, self._u4(4, n_bank, 4, 4)], (1, B, 1))
                    nb = emit_layer(ip, slot, nb, p_idx > 0)

            for il in range(18, 24): nb = emit_layer(il, il, nb, False)

            emit("mix_rms",    [self.buf_blocks, self.buf_cur, b_out_q, b_out_norm, self.buf_ffn_nrm, self._u4(8, nb)], (1, B, 1))
            emit("unembed_p1", [b_unembed, self.buf_ffn_nrm, self.buf_part_v, self.buf_part_i, self.buf_logits, self.buf_meta], (self.n_unembed_wg, bt8, 1))
            emit("argmax_p2",  [self.buf_part_v, self.buf_part_i, self.buf_logits, self.buf_meta, self.buf_tok_out], (1, B, 1), barrier=0)
            assert _vk.vk_record_end_slot(cmd_slot) == 0

        # Pre-record chunked-prefill command buffers (cmd_slots 5=C64, 6=C16) — NO output head (Fix 4)!
        u_gemm_wo = self._u4(384, 2304, 2304)
        for p_i, C in enumerate(PREFILL_BUCKETS):
            cmd_slot = len(BUCKETS) + p_i
            assert _vk.vk_record_begin_slot(cmd_slot) == 0
            ct8 = (C + 7) // 8

            def pemit(pname, bufs, wg, barrier=1):
                arr = (ctypes.c_int * len(bufs))(*bufs)
                assert _vk.vk_record_dispatch_slot(cmd_slot, self.pipes[pname], len(bufs), arr, wg[0], wg[1], wg[2], barrier) == 0

            pemit("rms", [self.buf_tmp, b_embed_norm, self.buf_cur, u_rms_2304], (1, C, 1))

            def pemit_layer(ip, slot, nb, has_msg):
                L = phys[ip]; is_swa = L["is_swa"]
                resid_buf = self.buf_cur

                if (not has_msg) and is_swa and ip > 0:
                    pemit("mix_rms", [self.buf_blocks, self.buf_cur, L["attn_q"], L["attn_norm"], self.buf_hq, self._u4(8, nb)], (1, C, 1))
                    hk_buf = hv_buf = self.buf_hq
                else:
                    if ip > 0:
                        n_q = 1 if is_swa else 3
                        pemit("mix", [self.buf_blocks, self.buf_cur, L["attn_q"], self.buf_mixed3, self._u4(8, nb, n_q, 3)], (1, C, 1))
                        in_buf = self.buf_mixed3; in_stride = 1728
                    else:
                        in_buf = self.buf_cur; in_stride = 576
                    if has_msg:
                        d3, g3, f3 = L["ctrl3"]
                        c_stride = 0 if is_swa else 2304
                        pemit("ctrl3_p1", [in_buf, self.buf_msgs, d3, self.buf_gate3, self._u4(c_stride // 2)], (64, 3, C))
                        pemit("ctrl3_p2", [in_buf, self.buf_msgs, self.buf_gate3, g3, f3, self.buf_ctrl3, self._u4(c_stride)], (36, 3, C))
                        if is_swa:
                            nw = L["attn_norm"]
                            pemit("rms3", [self.buf_ctrl3, nw, nw, nw, self.buf_hq, self.buf_hk, self.buf_hv], (3, C, 1))
                        else:
                            pemit("rms3", [self.buf_ctrl3, L["q_norm"], L["k_norm"], L["v_norm"], self.buf_hq, self.buf_hk, self.buf_hv], (3, C, 1))
                        hk_buf, hv_buf = self.buf_hk, self.buf_hv
                    else:
                        if is_swa:
                            pemit("rms", [in_buf, L["attn_norm"], self.buf_hq, self._u4(0, in_stride)], (1, C, 1))
                            hk_buf = hv_buf = self.buf_hq
                        else:
                            pemit("rms3", [in_buf, L["q_norm"], L["k_norm"], L["v_norm"], self.buf_hq, self.buf_hk, self.buf_hv], (3, C, 1))
                            hk_buf, hv_buf = self.buf_hk, self.buf_hv

                if "w_qrsgkv" in L:
                    pemit("gemm", [L["w_qrsgkv"], self.buf_hq, self.buf_qrsg, self.buf_meta, u_gemm_8600], (4300, ct8, 1), barrier=1)
                    k_src, v_src, k_off, v_off = self.buf_qrsg, self.buf_qrsg, 6552, 7576
                elif is_swa:
                    pemit("gemm", [L["w_qrsg"], self.buf_hq, self.buf_qrsg, self.buf_meta, u_gemm_6552], (3276, ct8, 1), barrier=0)
                    pemit("gemm", [L["wk"],     hk_buf,      self.buf_kbuf, self.buf_meta, u_gemm_1024], (512,  ct8, 1), barrier=0)
                    pemit("gemm", [L["wv"],     hv_buf,      self.buf_vbuf, self.buf_meta, u_gemm_1024], (512,  ct8, 1), barrier=1)
                    k_src, v_src, k_off, v_off = self.buf_kbuf, self.buf_vbuf, 0, 0
                else:
                    pemit("gemm", [L["w_qrsg"], self.buf_hq, self.buf_qrsg, self.buf_meta, u_gemm_6552], (3276, ct8, 1), barrier=0)
                    pemit("gemm", [L["wk_dyn"], hk_buf,      self.buf_kbuf, self.buf_meta, u_gemm_1056], (528,  ct8, 1), barrier=0)
                    pemit("gemm", [L["wv_dyn"], hv_buf,      self.buf_vbuf, self.buf_meta, u_gemm_1056], (528,  ct8, 1), barrier=1)
                    k_src, v_src, k_off, v_off = self.buf_kbuf, self.buf_vbuf, 0, 0

                c_idx = swa_map[slot] if is_swa else full_map[slot]
                k0_b, k1_b = (self.buf_k_swa0, self.buf_k_swa1) if is_swa else (self.buf_k_full0, self.buf_k_full1)
                v0_b, v1_b = (self.buf_v_swa0, self.buf_v_swa1) if is_swa else (self.buf_v_full0, self.buf_v_full1)
                pemit("p_qkvr_post", [
                    self.buf_qrsg,
                    self.buf_kbuf if k_off > 0 else k_src,
                    self.buf_vbuf if v_off > 0 else v_src,
                    L["q_clip"], L["r_clip"], L["k_sconv"], L["v_sconv"],
                    self.buf_hist_k, self.buf_hist_v, k0_b, k1_b, v0_b, v1_b,
                    self.buf_meta, self._u4((slot << 8) | c_idx, 0 if is_swa else 1, k_off, v_off)
                ], (64, C, 1))
                pemit("p_shift_kv", [
                    self.buf_qrsg, self.buf_kbuf if k_off > 0 else k_src, self.buf_vbuf if v_off > 0 else v_src,
                    self.buf_hist_k, self.buf_hist_v, self.buf_meta, self._u4(slot, 0, k_off, v_off)
                ], (8, 1, 1), barrier=0)

                pemit("p_attn_sk", [self.buf_qrsg, L["rel_bias"], k0_b, k1_b, v0_b, v1_b, self.buf_part_ml, self.buf_part_ac, self.buf_meta, self._u4(c_idx, 1 if is_swa else 0)], (24, 16, C))
                pemit("attn_cb", [self.buf_part_ml, self.buf_part_ac, self.buf_qrsg, self.buf_attn_ctx], (24, C, 1))

                add_r = 0 if ip % 6 == 0 else 1
                if ip % 6 == 0:
                    pemit("copy", [resid_buf, self.buf_blocks, self._u4(0, nb * 2304, 2304, 8 * 2304)], (9, C, 1), barrier=0)
                    nb += 1
                pemit("gemm", [L["wo"], self.buf_attn_ctx, self.buf_wo_raw, self.buf_meta, u_gemm_wo], (1152, ct8, 1))
                pemit("p_conv_resid", [self.buf_wo_raw, L["attn_sconv"], resid_buf, self.buf_hist_a, self.buf_ffn_inp, self.buf_meta, self._u4(slot, add_r)], (9, C, 1))
                pemit("p_shift_2304", [self.buf_wo_raw, self.buf_hist_a, self.buf_meta, self._u4(slot)], (9, 1, 1), barrier=0)

                if has_msg:
                    pemit("mix", [self.buf_blocks, self.buf_ffn_inp, L["mlp_q"], self.buf_ffn_mix, self._u4(8, nb, 1, 1)], (1, C, 1))
                    wd, wg_w, wf_w = L["ctrl_mlp"]
                    pemit("ctrl_p1", [self.buf_ffn_mix, self.buf_msgs, wd, self.buf_gate3, self._u4(0, 3 * 1152)], (64, 1, C))
                    pemit("ctrl_p2", [self.buf_ffn_mix, self.buf_msgs, self.buf_gate3, wg_w, wf_w, self.buf_ffn_ctl, self._u4(0, 3 * 2304, 0)], (36, 1, C))
                    pemit("rms", [self.buf_ffn_ctl, L["mlp_norm"], self.buf_ffn_nrm, u_rms_2304], (1, C, 1))
                else:
                    pemit("mix_rms", [self.buf_blocks, self.buf_ffn_inp, L["mlp_q"], L["mlp_norm"], self.buf_ffn_nrm, self._u4(8, nb)], (1, C, 1))

                pemit("gemm",   [L["router_w"], self.buf_ffn_nrm, self.buf_rlogits, self.buf_meta, u_gemm_128], (64, ct8, 1))
                pemit("router", [self.buf_rlogits, L["router_b"], self.buf_eids, self.buf_wts], (1, C, 1))
                pemit("p_group_exp", [self.buf_eids, self.buf_exp_cnt, self.buf_exp_toks, self.buf_meta], (1, 1, 1))
                if L["exp_kind"] == "nvfp4":
                    pemit("p_nv_w13_grp", [L["w13"], L["s13"], L["o13"], self.buf_exp_cnt, self.buf_exp_toks, self.buf_ffn_nrm, self.buf_moe_h], (576, 128, 1))
                    pemit("p_nv_w2_grp",  [L["w2"],  L["s2"],  L["o2"],  self.buf_exp_cnt, self.buf_exp_toks, self.buf_wts, self.buf_moe_h, self.buf_moe_part], (1152, 128, 1))
                else:
                    pemit("p_mx_w13_grp", [L["w13"], L["s13"], self.buf_exp_cnt, self.buf_exp_toks, self.buf_ffn_nrm, self.buf_moe_h], (576, 128, 1))
                    pemit("p_mx_w2_grp",  [L["w2"],  L["s2"],  self.buf_exp_cnt, self.buf_exp_toks, self.buf_wts, self.buf_moe_h, self.buf_moe_part], (1152, 128, 1))
                pemit("p_moe_sum",    [self.buf_moe_part, self.buf_wo_raw, self.buf_meta], (9, C, 1))
                pemit("p_conv_resid", [self.buf_wo_raw, L["mlp_sconv"], self.buf_ffn_inp, self.buf_hist_m, self.buf_cur, self.buf_meta, self._u4(slot, 1)], (9, C, 1))
                pemit("p_shift_2304", [self.buf_wo_raw, self.buf_hist_m, self.buf_meta, self._u4(slot)], (9, 1, 1), barrier=1)
                return nb

            nb = 0
            for il in range(6): nb = pemit_layer(il, il, nb, False)
            prelude_nb = nb
            pemit("copy", [self.buf_cur, self.buf_prelude, u_cp_00], (9, C, 1))

            for p_idx in range(3):
                if p_idx > 0:
                    n_bank = nb - (prelude_nb + 1)
                    for b in range(n_bank):
                        pemit("copy", [self.buf_blocks, self.buf_bank, self._u4((prelude_nb + 1 + b) * 2304, b * 2304, 8 * 2304, 4 * 2304)], (9, C, 1), barrier=0)
                    pemit("copy", [self.buf_cur, self.buf_bank_out, u_cp_00], (9, C, 1), barrier=1)
                    pemit("copy", [self.buf_prelude, self.buf_cur, u_cp_00], (9, C, 1), barrier=1)
                    nb = prelude_nb
                for ip in range(6, 18):
                    slot = ip if p_idx == 0 else 24 + (p_idx - 1) * 12 + (ip - 6)
                    if p_idx > 0:
                        pemit("mix", [self.buf_bank, self.buf_bank_out, msg_q_bufs[ip - 6], self.buf_msgs, self._u4(4, n_bank, 4, 4)], (1, C, 1))
                    nb = pemit_layer(ip, slot, nb, p_idx > 0)

            for il in range(18, 24): nb = pemit_layer(il, il, nb, False)
            # NO output head during prefill (Fix 4)!
            assert _vk.vk_record_end_slot(cmd_slot) == 0

    def reset_slot(self, seq_slot: int):
        sz_kv = 48 * 3 * 1024 * 4
        sz_am = 48 * 3 * 2304 * 4
        _vk.vk_zero_slice(self.buf_hist_k, seq_slot * sz_kv, sz_kv)
        _vk.vk_zero_slice(self.buf_hist_v, seq_slot * sz_kv, sz_kv)
        _vk.vk_zero_slice(self.buf_hist_a, seq_slot * sz_am, sz_am)
        _vk.vk_zero_slice(self.buf_hist_m, seq_slot * sz_am, sz_am)

    def snapshot_slot_conv(self, seq_slot: int):
        sz_kv = 48 * 3 * 1024 * 4
        sz_am = 48 * 3 * 2304 * 4
        hk = np.empty(48 * 3 * 1024, dtype=np.float32)
        hv = np.empty(48 * 3 * 1024, dtype=np.float32)
        ha = np.empty(48 * 3 * 2304, dtype=np.float32)
        hm = np.empty(48 * 3 * 2304, dtype=np.float32)
        assert _vk.vk_download_slice(self.buf_hist_k, seq_slot * sz_kv, hk.ctypes.data, sz_kv) == 0
        assert _vk.vk_download_slice(self.buf_hist_v, seq_slot * sz_kv, hv.ctypes.data, sz_kv) == 0
        assert _vk.vk_download_slice(self.buf_hist_a, seq_slot * sz_am, ha.ctypes.data, sz_am) == 0
        assert _vk.vk_download_slice(self.buf_hist_m, seq_slot * sz_am, hm.ctypes.data, sz_am) == 0
        return (hk, hv, ha, hm)

    def restore_slot_conv(self, seq_slot: int, snap):
        hk, hv, ha, hm = snap
        sz_kv = 48 * 3 * 1024 * 4
        sz_am = 48 * 3 * 2304 * 4
        assert _vk.vk_upload_slice(self.buf_hist_k, seq_slot * sz_kv, hk.ctypes.data, sz_kv) == 0
        assert _vk.vk_upload_slice(self.buf_hist_v, seq_slot * sz_kv, hv.ctypes.data, sz_kv) == 0
        assert _vk.vk_upload_slice(self.buf_hist_a, seq_slot * sz_am, ha.ctypes.data, sz_am) == 0
        assert _vk.vk_upload_slice(self.buf_hist_m, seq_slot * sz_am, hm.ctypes.data, sz_am) == 0

    def prefill_chunk(self, seq_slot: int, token_ids, start_pos: int):
        c = len(token_ids)
        assert 1 <= c <= 64
        cmd_slot = len(BUCKETS) + (1 if c <= 16 else 0) # 6 for <=16, 5 for 17..64
        self.meta_arr[0] = c
        self.meta_arr[1] = seq_slot
        self.meta_arr[2] = start_pos
        u16 = self.embed_u16[token_ids]
        self.tmp_arr[:c] = (u16.astype(np.uint32) << 16).view(np.float32)
        assert _vk.vk_run_recorded_slot(cmd_slot) == 0

    def prefill_tokens(self, seq_slot: int, token_ids, start_pos: int = 0):
        pos = start_pos
        i = 0
        n = len(token_ids)
        while i < n:
            chunk = token_ids[i : i + 64]
            self.prefill_chunk(seq_slot, chunk, pos)
            pos += len(chunk)
            i += len(chunk)
        return pos

    def step_batch(self, seq_slots, token_ids, positions, temps=None, top_ps=None, seeds=None):
        n = len(seq_slots)
        assert 1 <= n <= MAX_BATCH
        cmd_slot = 0
        for i, B in enumerate(BUCKETS):
            if B >= n:
                cmd_slot = i; break
        B = BUCKETS[cmd_slot]

        dummy_slot = 0
        if n < B:
            used = set(seq_slots)
            for s in range(MAX_BATCH):
                if s not in used:
                    dummy_slot = s; break

        self.meta_arr[0] = n
        for bi in range(B):
            if bi < n:
                u16 = self.embed_u16[token_ids[bi]]
                self.tmp_arr[bi] = (u16.astype(np.uint32) << 16).view(np.float32)
                self.meta_arr[16 + bi] = seq_slots[bi]
                self.meta_arr[32 + bi] = positions[bi]
                t_val = float(temps[bi]) if temps is not None else 0.0
                p_val = float(top_ps[bi]) if top_ps is not None else 1.0
                sd_val = int(seeds[bi]) & 0xFFFFFFFF if seeds is not None else 0
                self.meta_arr[48 + bi] = struct.unpack("<I", struct.pack("<f", t_val))[0]
                self.meta_arr[64 + bi] = struct.unpack("<I", struct.pack("<f", p_val))[0]
                self.meta_arr[80 + bi] = sd_val
            else:
                self.meta_arr[16 + bi] = dummy_slot
                self.meta_arr[32 + bi] = 0
                self.meta_arr[48 + bi] = 0
                self.meta_arr[64 + bi] = 0x3F800000
                self.meta_arr[80 + bi] = 0

        assert _vk.vk_run_recorded_slot(cmd_slot) == 0
        res = []
        for bi in range(n):
            tid = int(self.out_arr[bi, 0])
            sc  = struct.unpack("<f", struct.pack("<I", int(self.out_arr[bi, 1])))[0]
            res.append((tid, sc))
        return res

    def close(self):
        _vk.vk_shutdown()

if __name__ == "__main__":
    from tokenizers import Tokenizer
    MODEL_DIR = Path(os.environ.get("INKLING_MODEL_DIR", "./Inkling-2-Mini"))
    MODEL = str(MODEL_DIR / "model-00001-of-00001.safetensors")
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    eng = BatchedVulkanEngine(MODEL)
    vram_paths = sorted(glob.glob("/sys/class/drm/card*/device/mem_info_vram_used"))
    mem_str = ""
    if vram_paths:
        vp = Path(vram_paths[0])
        v = int(vp.read_text()) / (1024**3)
        g = int((vp.parent / "mem_info_gtt_used").read_text()) / (1024**3)
        mem_str = f" | real VRAM={v:.2f} GiB, GTT={g:.2f} GiB"
    print(f"BatchedVulkanEngine ready | buckets={BUCKETS} | SEQ_CAP={SEQ_CAP} | init={eng.init_s:.2f}s{mem_str}\n", flush=True)

    # 1. B=1 5-token check
    eng.reset_slot(0)
    eng.step_batch([0], [32269], [0])
    cur = 150770; t5, s5 = [], []
    for p in range(1, 6):
        (cur, sc), = eng.step_batch([0], [cur], [p])
        t5.append(cur); s5.append(round(sc, 4))
    print(f"B=1 5-tok check: {t5}  scores={s5}  ({'OK' if t5 == [0, 68014, 126311, 149882, 150084] else 'FAIL'})", flush=True)
    assert t5 == [0, 68014, 126311, 149882, 150084]

    # 2. Verify B=4 independent prompts match B=1 bit-exact!
    prompts = [
        [32269, 150770],
        tok.encode("The capital of France is").ids,
        tok.encode("def fib(n):\n    if n <=").ids,
        tok.encode("In 1969, Apollo 11 landed on").ids,
    ]
    ref_outs = []
    for p_ids in prompts:
        eng.reset_slot(0)
        for pos, t in enumerate(p_ids[:-1]): eng.step_batch([0], [t], [pos])
        cur = p_ids[-1]; pos = len(p_ids) - 1; gen = []
        for _ in range(25):
            (cur, _), = eng.step_batch([0], [cur], [pos]); pos += 1; gen.append(cur)
        ref_outs.append(gen)

    # Now run all 4 TOGETHER in continuous batching (different prompt lengths, joining & leaving!)
    for s in range(4): eng.reset_slot(s)
    states = [{"slot": s, "prompt": prompts[s], "pos": 0, "cur": prompts[s][0], "gen": []} for s in range(4)]
    while states:
        slots = [st["slot"] for st in states]
        tids  = [st["cur"]  for st in states]
        poss  = [st["pos"]  for st in states]
        outs  = eng.step_batch(slots, tids, poss)
        nxt_states = []
        for st, (nxt_t, _) in zip(states, outs):
            st["pos"] += 1
            if st["pos"] < len(st["prompt"]):
                st["cur"] = st["prompt"][st["pos"]]
                nxt_states.append(st)
            else:
                st["gen"].append(nxt_t)
                st["cur"] = nxt_t
                if len(st["gen"]) < 25:
                    nxt_states.append(st)
        states = nxt_states
    # Wait: collect completed gens
    # Let's re-run cleanly collecting into bat_outs
    for s in range(4): eng.reset_slot(s)
    all_st = [{"slot": s, "prompt": prompts[s], "pos": 0, "cur": prompts[s][0], "gen": []} for s in range(4)]
    active = list(all_st)
    while active:
        outs = eng.step_batch([st["slot"] for st in active], [st["cur"] for st in active], [st["pos"] for st in active])
        nxt_active = []
        for st, (nxt_t, _) in zip(active, outs):
            st["pos"] += 1
            if st["pos"] < len(st["prompt"]):
                st["cur"] = st["prompt"][st["pos"]]; nxt_active.append(st)
            else:
                st["gen"].append(nxt_t); st["cur"] = nxt_t
                if len(st["gen"]) < 25: nxt_active.append(st)
        active = nxt_active
    matches = [all_st[i]["gen"] == ref_outs[i] for i in range(4)]
    print(f"Continuous batching B=4 staggered join/leave vs B=1: {matches} ({'ALL 100% BIT-EXACT!' if all(matches) else 'DIFF!'})\n", flush=True)
    assert all(matches)

    # 3. STEP 4: MEASURE LIKE A HOST (B = 1, 2, 4, 8, 16 concurrent users at depth ~50!)
    print("=== STEP 4: HOST THROUGHPUT SCALING (1, 2, 4, 8, 16 concurrent users, depth=50..90) ===")
    print(f"{'Users (B)':>9s} | {'Step ms':>8s} | {'Per-user tok/s':>14s} | {'TOTAL tok/s':>12s} | {'Scaling vs B=1':>14s}")
    print("-" * 70)
    b1_tps = None
    # Use distinct prompts per user so experts genuinely fan out!
    p500 = [int(x) for x in (BASE_DIR / "p500_tokens.txt").read_text().split()]
    user_prompts = [(p500 * 2)[i*30 : i*30 + 32] for i in range(16)]
    for B in BUCKETS:
        for s in range(B): eng.reset_slot(s)
        slots = list(range(B))
        # Prefill 32 distinct tokens per user + 18 decode warmup -> depth=50
        for p in range(31):
            eng.step_batch(slots, [user_prompts[s][p] for s in range(B)], [p]*B)
        curs = [user_prompts[s][31] for s in range(B)]
        for p in range(31, 50):
            outs = eng.step_batch(slots, curs, [p]*B)
            curs = [t for t, _ in outs]
        # Measure 40 decode steps at depth 50..90
        dts = []
        for p in range(50, 90):
            t0 = time.perf_counter()
            outs = eng.step_batch(slots, curs, [p]*B)
            dts.append((time.perf_counter() - t0) * 1e3)
            curs = [t for t, _ in outs]
        med = float(np.median(dts))
        per_u = 1000.0 / med
        tot_tps = B * per_u
        if b1_tps is None: b1_tps = tot_tps
        print(f"{B:9d} | {med:8.2f} | {per_u:14.2f} | {tot_tps:12.2f} | {tot_tps/b1_tps:13.2f}x", flush=True)
    eng.close()
