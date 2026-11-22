from concurrent.futures import ThreadPoolExecutor
import ctypes, json, mmap, os, struct, sys, time
import numpy as np

FULL_ATTN = {5, 11, 17, 23}
MAX_BATCH = int(os.environ.get("INKLING_MAX_BATCH", "2"))
SEQ_CAP   = int(os.environ.get("INKLING_SEQ_CAP", "131072"))
SWA_WIN   = 1024
BUCKETS   = [b for b in [1, 2, 4, 8, 16] if b <= MAX_BATCH]
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
_vk.vk_copy_batch.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_uint64),
]
_vk.vk_copy_batch.restype = ctypes.c_int
_vk.vk_quant_bf16_to_fp8.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_vk.vk_quant_bf16_to_fp8.restype = None
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
    ("unembed_p1", 7, 0),
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

DSPARK_PIPE_SPECS = [
    ("d_save_tap",  4, 1 << 3),
    ("d_save_ring", 3, 1 << 2),
    ("d_snap_hist", 4, 1 << 3),
    ("d_rb_hist",   5, 1 << 4),
    ("d_fuse",      5, 0),
    ("d_qkn_rope",  6, 1 << 5),
    ("d_store_v",   4, 1 << 3),
    ("d_copy_anc",  3, 0),
    ("d_attn",      5, 0),
    ("d_add",       4, 0),
    ("d_swiglu",    4, 0),
    ("d_mrnn",     10, 1 << 9),
    ("d_mupd",      7, 1 << 6),
    ("d_am_slot",   4, 1 << 3),
    ("v_attn_sk",  10, 1 << 9),
    ("d_attn_sk",   9, 1 << 8),
    ("d_attn_cb",   3, 0),
]

class BatchedVulkanEngine:
    def __init__(self, path: str, spv_dir: str | None = None, max_batch: int | None = None, seq_cap: int | None = None, mtp_path: str | None = None):
        self.max_batch = int(max_batch) if max_batch is not None else MAX_BATCH
        self.seq_cap = int(seq_cap) if seq_cap is not None else SEQ_CAP
        assert self.max_batch >= 1 and (self.max_batch == 1 or self.max_batch % 2 == 0), "max_batch must be 1 or even"
        self.half_b = max(1, self.max_batch // 2)
        self.buckets = [b for b in [1, 2, 4, 8, 16] if b <= self.max_batch]
        if mtp_path is None:
            cand = os.path.join(os.path.dirname(path), "mtp.safetensors")
            if os.path.exists(cand):
                mtp_path = cand
        self.mtp_path = mtp_path
        self.has_mtp = bool(mtp_path and os.path.exists(mtp_path) and (len(self.buckets) + len(PREFILL_BUCKETS) + 3 <= 8))
        self.mtp_valid = [False] * self.max_batch
        self.mtp_ema   = [4.5] * self.max_batch
        self.mtp_skip  = [0] * self.max_batch
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

        if spv_dir is None:
            spv_dir = str(BASE_DIR / "spv_batch")
        self.pipes = {}
        for name, n_bind, umask in PIPE_SPECS:
            spv = open(os.path.join(spv_dir, f"{name}.spv"), "rb").read()
            arr = np.frombuffer(spv, dtype=np.uint8)
            pid = _vk.vk_create_pipeline(arr.ctypes.data, len(spv), n_bind, umask)
            assert pid >= 0, f"vk_create_pipeline failed for {name}"
            self.pipes[name] = pid

        prefill_spv_dir = str(BASE_DIR / "spv_prefill")
        for name, n_bind, umask in PREFILL_PIPE_SPECS:
            spv = open(os.path.join(prefill_spv_dir, f"{name}.spv"), "rb").read()
            arr = np.frombuffer(spv, dtype=np.uint8)
            pid = _vk.vk_create_pipeline(arr.ctypes.data, len(spv), n_bind, umask)
            assert pid >= 0, f"vk_create_pipeline failed for {name}"
            self.pipes[name] = pid

        if self.has_mtp:
            dspark_spv_dir = str(BASE_DIR / "spv_dspark")
            for name, n_bind, umask in DSPARK_PIPE_SPECS:
                spv = open(os.path.join(dspark_spv_dir, f"{name}.spv"), "rb").read()
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

    def _up_unembed_fp8(self, name, n_rows: int):
        raw_u16 = np.ascontiguousarray(np.frombuffer(self._raw(name), dtype="<u2").reshape(-1, 2304)[:n_rows])
        out_u8 = np.empty((n_rows, 2304), dtype=np.uint8)
        scales = np.empty(n_rows, dtype=np.float32)
        _vk.vk_quant_bf16_to_fp8(raw_u16.ctypes.data, out_u8.ctypes.data, scales.ctypes.data, n_rows, 2304)
        return self._up(out_u8), self._up(scales)

    def _up_fp8_mat(self, raw_b, n_cols: int):
        u16 = np.ascontiguousarray(np.frombuffer(raw_b, dtype="<u2").reshape(-1, n_cols))
        n_rows = u16.shape[0]
        out_u8 = np.zeros(n_rows * n_cols + ((n_rows + 3) // 4) * 16, dtype=np.uint8)
        sc_ptr = out_u8.ctypes.data + n_rows * n_cols
        _vk.vk_quant_bf16_to_fp8(u16.ctypes.data, out_u8.ctypes.data, sc_ptr, n_rows, n_cols)
        return self._up(out_u8)

    def _embed_text(self, token_ids):
        u16 = self.embed_u16[token_ids]
        x = (u16.astype(np.uint32) << 16).view(np.float32)
        rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6)
        return (x / rms) * self.embed_norm_w

    def _u4(self, a, b=0, c=0, d=0):
        return self._up(np.array([a, b, c, d], dtype=np.uint32), host_visible=1, is_uniform=1)

    def _alloc_scratch(self):
        B = self.max_batch
        S = MAX_SCRATCH
        # Host-mapped I/O buffers
        self.buf_meta    = self._mk(128 * 4, host_visible=1)
        self.buf_tmp     = self._mk(S * 2304 * 4, host_visible=1)
        OB = max(B, 8) if self.has_mtp else B
        self.buf_tok_out = self._mk(OB * 2 * 4, host_visible=2)

        self.meta_arr = np.ctypeslib.as_array(ctypes.cast(_vk.vk_get_mapped(self.buf_meta), ctypes.POINTER(ctypes.c_uint32)), shape=(128,))
        self.tmp_arr  = np.ctypeslib.as_array(ctypes.cast(_vk.vk_get_mapped(self.buf_tmp),  ctypes.POINTER(ctypes.c_float)),  shape=(S, 2304))
        self.out_arr  = np.ctypeslib.as_array(ctypes.cast(_vk.vk_get_mapped(self.buf_tok_out), ctypes.POINTER(ctypes.c_uint32)), shape=(OB, 2))

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
        # FP8 E4M3 KV cache (pack4xU8: 256 u32 = 1024 B per tok per layer, 2x smaller than FP16)
        swa_half_bytes  = self.half_b * 40 * SWA_WIN * 256 * 4
        full_half_bytes = self.half_b *  8 * self.seq_cap * 256 * 4
        assert full_half_bytes <= 2147483648, f"full_half_bytes={full_half_bytes} exceeds 2 GiB RADV SSBO limit"
        self.meta_arr[14] = self.seq_cap
        self.meta_arr[15] = self.half_b
        self.buf_k_swa0  = self._mk(swa_half_bytes)
        self.buf_k_swa1  = self._mk(swa_half_bytes)
        self.buf_v_swa0  = self._mk(swa_half_bytes)
        self.buf_v_swa1  = self._mk(swa_half_bytes)
        self.buf_k_full0 = self._mk(full_half_bytes)
        self.buf_k_full1 = self._mk(full_half_bytes)
        self.buf_v_full0 = self._mk(full_half_bytes)
        self.buf_v_full1 = self._mk(full_half_bytes)
        self.buf_logits  = self._mk(OB * 200058 * 4)
        self.buf_hist_k  = self._mk(B * 48 * 3 * 1024 * 4)
        self.buf_hist_v  = self._mk(B * 48 * 3 * 1024 * 4)
        self.buf_hist_a  = self._mk(B * 48 * 3 * 2304 * 4)
        self.buf_hist_m  = self._mk(B * 48 * 3 * 2304 * 4)

        self.n_unembed_wg = (200058 + 3) // 4  # 50015
        self.buf_part_v  = self._mk(OB * self.n_unembed_wg * 4)
        self.buf_part_i  = self._mk(OB * self.n_unembed_wg * 4)

        if self.has_mtp:
            self.b_taps      = self._mk(S * 4 * 2304 * 4)
            self.b_snap_k    = self._mk(48 * 3 * 1024 * 4)
            self.b_snap_v    = self._mk(48 * 3 * 1024 * 4)
            self.b_snap_a    = self._mk(48 * 3 * 2304 * 4)
            self.b_snap_m    = self._mk(48 * 3 * 2304 * 4)
            self.b_ring_k    = self._mk(48 * 8 * 1024 * 4)
            self.b_ring_v    = self._mk(48 * 8 * 1024 * 4)
            self.b_ring_a    = self._mk(48 * 8 * 2304 * 4)
            self.b_ring_m    = self._mk(48 * 8 * 2304 * 4)
            self.b_dk_swa    = self._mk(B * 2 * SWA_WIN * 512 * 4)
            self.b_dv_swa    = self._mk(B * 2 * SWA_WIN * 512 * 4)
            self.b_dk_full   = self._mk(B * 1 * self.seq_cap * 512 * 4)
            self.b_dv_full   = self._mk(B * 1 * self.seq_cap * 512 * 4)
            self.b_ctx_tmp   = self._mk(S * 2304 * 4)
            self.b_ctx_k     = self._mk(S * 1024 * 4)
            self.b_ctx_v     = self._mk(S * 1024 * 4)
            self.b_ctx_anc   = self._mk(B * 2304 * 4)
            self.b_d_cur     = self._mk(8 * 2304 * 4)
            self.b_d_h       = self._mk(8 * 2304 * 4)
            self.b_d_q       = self._mk(8 * 3072 * 4)
            self.b_d_k       = self._mk(8 * 1024 * 4)
            self.b_d_v       = self._mk(8 * 1024 * 4)
            self.b_d_qr      = self._mk(8 * 3072 * 4)
            self.b_d_kr      = self._mk(8 * 1024 * 4)
            self.b_d_ao      = self._mk(8 * 3072 * 4)
            self.b_d_oo      = self._mk(8 * 2304 * 4)
            self.b_d_mg      = self._mk(8 * 9216 * 4)
            self.b_d_mu      = self._mk(8 * 9216 * 4)
            self.b_d_md      = self._mk(8 * 2304 * 4)
            self.b_d_norm    = self._mk(8 * 2304 * 4)
            self.b_mk_jout   = self._mk(768 * 4)
            self.b_mk_comb   = self._mk(2304 * 4)
            self.b_mk_state  = self._mk(256 * 4, host_visible=1)
            self.b_d_toks    = self._mk(8 * 4, host_visible=2)
            self.b_meta_1    = self._up(np.array([1] + [0]*15, dtype=np.uint32))
            self.mk_state_arr = np.ctypeslib.as_array(ctypes.cast(_vk.vk_get_mapped(self.b_mk_state), ctypes.POINTER(ctypes.c_float)), shape=(256,))
            self.d_toks_arr   = np.ctypeslib.as_array(ctypes.cast(_vk.vk_get_mapped(self.b_d_toks),   ctypes.POINTER(ctypes.c_uint32)), shape=(8,))

        # GPU VRAM L1 Prefix Cache Pool (vkCmdCopyBuffer @ 640 GB/s -> ~1ms restore!)
        self.n_prefix_slots = int(os.environ.get("INKLING_PREFIX_SLOTS", "4")) if self.seq_cap >= 4096 else 0
        self.prefix_cap = min(self.seq_cap, 32768)
        self.prefix_pool = []
        for _ in range(self.n_prefix_slots):
            ps = {
                "k_swa":  self._mk(40 * SWA_WIN * 256 * 4),
                "v_swa":  self._mk(40 * SWA_WIN * 256 * 4),
                "k_full": self._mk(8 * self.prefix_cap * 256 * 4),
                "v_full": self._mk(8 * self.prefix_cap * 256 * 4),
                "hist_k": self._mk(48 * 3 * 1024 * 4),
                "hist_v": self._mk(48 * 3 * 1024 * 4),
                "hist_a": self._mk(48 * 3 * 2304 * 4),
                "hist_m": self._mk(48 * 3 * 2304 * 4),
            }
            if self.has_mtp:
                ps["dk_swa"]  = self._mk(2 * SWA_WIN * 512 * 4)
                ps["dv_swa"]  = self._mk(2 * SWA_WIN * 512 * 4)
                ps["dk_full"] = self._mk(self.prefix_cap * 512 * 4)
                ps["dv_full"] = self._mk(self.prefix_cap * 512 * 4)
                ps["ctx_anc"] = self._mk(2304 * 4)
            self.prefix_pool.append(ps)

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
                    L["w_qrsgkv"] = self._up_fp8_mat(b_wq + b_wr + b_ws + b_wg + b_wk + b_wv, 2304)
                else:
                    L["w_qrsg"] = self._up_fp8_mat(b_wq + b_wr + b_ws + b_wg, 2304)
                    L["wk"]     = self._up_fp8_mat(b_wk, 2304)
                    L["wv"]     = self._up_fp8_mat(b_wv, 2304)
            else:
                L["w_qrsg"] = self._up_fp8_mat(b_wq + b_wr + b_ws + b_wg, 2304)
                L["wk_dyn"] = self._up_fp8_mat(b_wk + bytes(self._raw(f"{p}.attn.k_sconv.weight_proj")), 2304)
                L["wv_dyn"] = self._up_fp8_mat(b_wv + bytes(self._raw(f"{p}.attn.v_sconv.weight_proj")), 2304)

            L["wo"]        = self._up_fp8_mat(self._raw(f"{p}.attn.wo_ud.weight"), 3072)
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
                def _up_gf_t(raw_b):
                    arr = np.frombuffer(raw_b, dtype=np.uint32).reshape(-1, 2304, 32)
                    return self._up(np.ascontiguousarray(arr.transpose(0, 2, 1)))
                d3 = b"".join(bytes(self._raw(f"{p}.controllers.{c}.down_weight"))   for c in ("attn", "k", "v"))
                g3 = b"".join(bytes(self._raw(f"{p}.controllers.{c}.gain_weight"))   for c in ("attn", "k", "v"))
                f3 = b"".join(bytes(self._raw(f"{p}.controllers.{c}.forget_weight")) for c in ("attn", "k", "v"))
                L["ctrl3"] = (self._up(d3), _up_gf_t(g3), _up_gf_t(f3))
                L["ctrl_mlp"] = (
                    self._up(self._raw(f"{p}.controllers.mlp.down_weight")),
                    _up_gf_t(bytes(self._raw(f"{p}.controllers.mlp.gain_weight"))),
                    _up_gf_t(bytes(self._raw(f"{p}.controllers.mlp.forget_weight"))),
                )
            phys.append(L)

        cp_q  = np.frombuffer(self._raw("model.llm.middle_cycle_cross_pass_queries"), dtype="<f4").reshape(24, 2304)
        cp_kv = np.frombuffer(self._raw("model.llm.middle_cycle_cross_pass_kv_queries"), dtype="<f4").reshape(24, 2304)
        msg_q_bufs = [self._up(np.stack([cp_q[2*i], cp_kv[2*i], cp_kv[2*i+1], cp_q[2*i+1]], axis=0).astype(np.float32)) for i in range(12)]

        self.embed_norm_w = self._bf16_to_f32("model.llm.embed_norm.weight")
        b_out_q      = self._up_f32("model.llm.attn_res.output_query")
        b_out_norm   = self._up_f32("model.llm.norm.weight")
        b_unembed, b_unembed_sc = self._up_unembed_fp8("model.llm.unembed.weight", 200058)

        # Shared uniforms
        u_rms_2304 = self._u4(0, 576)
        u_gemm_8600 = self._u4(144, 8600, 8600, 0)
        u_gemm_6552 = self._u4(144, 6552, 8600, 0)
        u_gemm_1024 = self._u4(144, 1024, 1056, 0)
        u_gemm_1056 = self._u4(144, 1056, 1056, 0)
        u_gemm_128  = self._u4(288, 128, 128, 1)
        u_cp_00 = self._u4(0, 0, 2304, 2304)
        u_taps = [self._u4(i) for i in range(4)] if self.has_mtp else None

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
        for cmd_slot, B in enumerate(self.buckets):
            assert _vk.vk_record_begin_slot(cmd_slot) == 0
            bt8 = (B + 7) // 8

            def emit(pname, bufs, wg, barrier=1):
                arr = (ctypes.c_int * len(bufs))(*bufs)
                assert _vk.vk_record_dispatch_slot(cmd_slot, self.pipes[pname], len(bufs), arr, wg[0], wg[1], wg[2], barrier) == 0

            emit("copy", [self.buf_tmp, self.buf_cur, u_cp_00], (9, B, 1))

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
                        emit("ctrl3_p1", [in_buf, self.buf_msgs, d3, self.buf_gate3, self._u4(c_stride // 2, B)], (64, 3, bt8))
                        emit("ctrl3_p2", [in_buf, self.buf_msgs, self.buf_gate3, g3, f3, self.buf_ctrl3, self._u4(c_stride, B)], (36, 3, bt8))
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
                    emit("ctrl_p1", [self.buf_ffn_mix, self.buf_msgs, wd, self.buf_gate3, self._u4(0, 3 * 1152, B)], (64, 1, bt8))
                    emit("ctrl_p2", [self.buf_ffn_mix, self.buf_msgs, self.buf_gate3, wg_w, wf_w, self.buf_ffn_ctl, self._u4(0, 3 * 2304, 0, B)], (36, 1, bt8))
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

            nb = 0; visit = 0; do_tap = (self.has_mtp and B == 1)
            for il in range(6):
                if do_tap and visit == 5: emit("d_save_tap", [self.buf_cur, self.b_taps, self.buf_meta, u_taps[0]], (9, 1, 1))
                visit += 1
                nb = emit_layer(il, il, nb, False)
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
                    if do_tap and visit == 23: emit("d_save_tap", [self.buf_cur, self.b_taps, self.buf_meta, u_taps[1]], (9, 1, 1))
                    if do_tap and visit == 41: emit("d_save_tap", [self.buf_cur, self.b_taps, self.buf_meta, u_taps[2]], (9, 1, 1))
                    visit += 1
                    slot = ip if p_idx == 0 else 24 + (p_idx - 1) * 12 + (ip - 6)
                    if p_idx > 0:
                        emit("mix", [self.buf_bank, self.buf_bank_out, msg_q_bufs[ip - 6], self.buf_msgs, self._u4(4, n_bank, 4, 4)], (1, B, 1))
                    nb = emit_layer(ip, slot, nb, p_idx > 0)

            for il in range(18, 24):
                visit += 1
                nb = emit_layer(il, il, nb, False)

            emit("mix_rms",    [self.buf_blocks, self.buf_cur, b_out_q, b_out_norm, self.buf_ffn_nrm, self._u4(8, nb)], (1, B, 1))
            if do_tap: emit("d_save_tap", [self.buf_ffn_nrm, self.b_taps, self.buf_meta, u_taps[3]], (9, 1, 1))
            emit("unembed_p1", [b_unembed, self.buf_ffn_nrm, self.buf_part_v, self.buf_part_i, self.buf_logits, self.buf_meta, b_unembed_sc], (self.n_unembed_wg, bt8, 1))
            emit("argmax_p2",  [self.buf_part_v, self.buf_part_i, self.buf_logits, self.buf_meta, self.buf_tok_out], (1, B, 1), barrier=0)
            assert _vk.vk_record_end_slot(cmd_slot) == 0

        # Pre-record chunked-prefill command buffers (cmd_slots 5=C64, 6=C16) — NO output head (Fix 4)!
        u_gemm_wo = self._u4(192, 2304, 2304, 0)
        for p_i, C in enumerate(PREFILL_BUCKETS):
            cmd_slot = len(self.buckets) + p_i
            assert _vk.vk_record_begin_slot(cmd_slot) == 0
            ct8 = (C + 7) // 8

            def pemit(pname, bufs, wg, barrier=1):
                arr = (ctypes.c_int * len(bufs))(*bufs)
                assert _vk.vk_record_dispatch_slot(cmd_slot, self.pipes[pname], len(bufs), arr, wg[0], wg[1], wg[2], barrier) == 0

            pemit("copy", [self.buf_tmp, self.buf_cur, u_cp_00], (9, C, 1))

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
                        pemit("ctrl3_p1", [in_buf, self.buf_msgs, d3, self.buf_gate3, self._u4(c_stride // 2, C)], (64, 3, ct8))
                        pemit("ctrl3_p2", [in_buf, self.buf_msgs, self.buf_gate3, g3, f3, self.buf_ctrl3, self._u4(c_stride, C)], (36, 3, ct8))
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
                pemit("p_conv_resid", [self.buf_wo_raw, L["attn_sconv"], resid_buf, self.buf_hist_a, self.buf_ffn_inp, self.buf_meta, self._u4(slot, add_r)], (9, 1, 1))

                if has_msg:
                    pemit("mix", [self.buf_blocks, self.buf_ffn_inp, L["mlp_q"], self.buf_ffn_mix, self._u4(8, nb, 1, 1)], (1, C, 1))
                    wd, wg_w, wf_w = L["ctrl_mlp"]
                    pemit("ctrl_p1", [self.buf_ffn_mix, self.buf_msgs, wd, self.buf_gate3, self._u4(0, 3 * 1152, C)], (64, 1, ct8))
                    pemit("ctrl_p2", [self.buf_ffn_mix, self.buf_msgs, self.buf_gate3, wg_w, wf_w, self.buf_ffn_ctl, self._u4(0, 3 * 2304, 0, C)], (36, 1, ct8))
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
                pemit("p_conv_resid", [self.buf_wo_raw, L["mlp_sconv"], self.buf_ffn_inp, self.buf_hist_m, self.buf_cur, self.buf_meta, self._u4(slot, 1)], (9, 1, 1))
                return nb

            nb = 0; visit = 0
            for il in range(6):
                if self.has_mtp and visit == 5: pemit("d_save_tap", [self.buf_cur, self.b_taps, self.buf_meta, u_taps[0]], (9, C, 1))
                visit += 1
                nb = pemit_layer(il, il, nb, False)
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
                    if self.has_mtp and visit == 23: pemit("d_save_tap", [self.buf_cur, self.b_taps, self.buf_meta, u_taps[1]], (9, C, 1))
                    if self.has_mtp and visit == 41: pemit("d_save_tap", [self.buf_cur, self.b_taps, self.buf_meta, u_taps[2]], (9, C, 1))
                    visit += 1
                    slot = ip if p_idx == 0 else 24 + (p_idx - 1) * 12 + (ip - 6)
                    if p_idx > 0:
                        pemit("mix", [self.buf_bank, self.buf_bank_out, msg_q_bufs[ip - 6], self.buf_msgs, self._u4(4, n_bank, 4, 4)], (1, C, 1))
                    nb = pemit_layer(ip, slot, nb, p_idx > 0)

            for il in range(18, 24):
                visit += 1
                nb = pemit_layer(il, il, nb, False)
            if self.has_mtp:
                pemit("mix_rms", [self.buf_blocks, self.buf_cur, b_out_q, b_out_norm, self.buf_ffn_nrm, self._u4(8, nb)], (1, C, 1))
                pemit("d_save_tap", [self.buf_ffn_nrm, self.b_taps, self.buf_meta, u_taps[3]], (9, C, 1))
            assert _vk.vk_record_end_slot(cmd_slot) == 0

        if self.has_mtp:
            self._build_mtp_cmds(phys, msg_q_bufs, b_out_q, b_out_norm, b_unembed, b_unembed_sc,
                                 u_rms_2304, u_gemm_8600, u_gemm_6552, u_gemm_1024, u_gemm_1056, u_gemm_128, u_gemm_wo, u_cp_00,
                                 swa_map, full_map, u_taps)


    def _build_mtp_cmds(self, phys, msg_q_bufs, b_out_q, b_out_norm, b_unembed, b_unembed_sc,
                        u_rms_2304, u_gemm_8600, u_gemm_6552, u_gemm_1024, u_gemm_1056, u_gemm_128, u_gemm_wo, u_cp_00,
                        swa_map, full_map, u_taps):
        mf = open(self.mtp_path, "rb")
        mmm = mmap.mmap(mf.fileno(), 0, access=mmap.ACCESS_READ)
        mn = struct.unpack("<Q", mmm[:8])[0]
        mh = json.loads(mmm[8 : 8 + mn].decode("utf-8"))
        mbase = 8 + mn
        def mraw(name):
            s, e = mh[name]["data_offsets"]
            return memoryview(mmm)[mbase + s : mbase + e]
        def mf32(name):
            u16 = np.frombuffer(mraw(name), dtype="<u2")
            return (u16.astype(np.uint32) << 16).view(np.float32)

        fuse_raw = mf32("fuse_weight").reshape(4, 4)
        b_fuse_rows = []
        for r in range(4):
            e = np.exp(fuse_raw[r] - np.max(fuse_raw[r]))
            b_fuse_rows.append(self._up((e / np.sum(e)).astype(np.float32)))

        b_mask_emb = self._up(mf32("mask_embed"))
        b_mtp_norm = self._up(mf32("norm.weight"))
        b_mk_dnorm = self._up(mf32("markov_head.dflare_norm.weight"))
        b_mk_w1    = self._up(mraw("markov_head.markov_w1.weight"))
        b_mk_rnn_w = self._up(mraw("markov_head.rnn_proj.weight"))
        b_mk_rnn_b = self._up(mf32("markov_head.rnn_proj.bias"))
        b_mk_ho_w  = self._up(mraw("markov_head.hidden_out.weight"))
        b_mk_ho_b  = self._up(mf32("markov_head.hidden_out.bias"))

        mtp_layers = []
        for il in range(3):
            lp = f"layers.{il}"
            mtp_layers.append({
                "dnorm":   self._up(mf32(f"{lp}.dflare_norm.weight")),
                "inorm":   self._up(mf32(f"{lp}.input_layernorm.weight")),
                "pnorm":   self._up(mf32(f"{lp}.post_attention_layernorm.weight")),
                "q_norm":  self._up(mf32(f"{lp}.self_attn.q_norm.weight")),
                "k_norm":  self._up(mf32(f"{lp}.self_attn.k_norm.weight")),
                "kc_norm": self._up(mf32(f"{lp}.self_attn.k_norm_ctx.weight")),
                "wq":      self._up_fp8_mat(mraw(f"{lp}.self_attn.q_proj.weight"), 2304),
                "wk":      self._up_fp8_mat(mraw(f"{lp}.self_attn.k_proj.weight"), 2304),
                "wv":      self._up_fp8_mat(mraw(f"{lp}.self_attn.v_proj.weight"), 2304),
                "wkc":     self._up_fp8_mat(mraw(f"{lp}.self_attn.k_proj_ctx.weight"), 2304),
                "wvc":     self._up_fp8_mat(mraw(f"{lp}.self_attn.v_proj_ctx.weight"), 2304),
                "wo":      self._up_fp8_mat(mraw(f"{lp}.self_attn.o_proj.weight"), 3072),
                "wgate":   self._up_fp8_mat(mraw(f"{lp}.mlp.gate_proj.weight"), 2304),
                "wup":     self._up_fp8_mat(mraw(f"{lp}.mlp.up_proj.weight"), 2304),
                "wdown":   self._up_fp8_mat(mraw(f"{lp}.mlp.down_proj.weight"), 9216),
            })
        mmm.close(); mf.close()

        self.cmd_verify8   = len(self.buckets) + len(PREFILL_BUCKETS)
        self.cmd_rb_inject = self.cmd_verify8 + 1
        self.cmd_draft7    = self.cmd_verify8 + 2

        C = 8; ct8 = 1
        assert _vk.vk_record_begin_slot(self.cmd_verify8) == 0
        def vemit(pname, bufs, wg, barrier=1):
            arr = (ctypes.c_int * len(bufs))(*bufs)
            assert _vk.vk_record_dispatch_slot(self.cmd_verify8, self.pipes[pname], len(bufs), arr, wg[0], wg[1], wg[2], barrier) == 0

        u_1024 = self._u4(1024); u_2304 = self._u4(2304)
        vemit("d_snap_hist", [self.buf_hist_k, self.b_snap_k, self.buf_meta, u_1024], (12, 48, 1), barrier=0)
        vemit("d_snap_hist", [self.buf_hist_v, self.b_snap_v, self.buf_meta, u_1024], (12, 48, 1), barrier=0)
        vemit("d_snap_hist", [self.buf_hist_a, self.b_snap_a, self.buf_meta, u_2304], (27, 48, 1), barrier=0)
        vemit("d_snap_hist", [self.buf_hist_m, self.b_snap_m, self.buf_meta, u_2304], (27, 48, 1), barrier=1)

        vemit("copy", [self.buf_tmp, self.buf_cur, u_cp_00], (9, C, 1))

        def vemit_layer(ip, slot, nb, has_msg):
            L = phys[ip]; is_swa = L["is_swa"]
            resid_buf = self.buf_cur
            if (not has_msg) and is_swa and ip > 0:
                vemit("mix_rms", [self.buf_blocks, self.buf_cur, L["attn_q"], L["attn_norm"], self.buf_hq, self._u4(8, nb)], (1, C, 1))
                hk_buf = hv_buf = self.buf_hq
            else:
                if ip > 0:
                    n_q = 1 if is_swa else 3
                    vemit("mix", [self.buf_blocks, self.buf_cur, L["attn_q"], self.buf_mixed3, self._u4(8, nb, n_q, 3)], (1, C, 1))
                    in_buf = self.buf_mixed3; in_stride = 1728
                else:
                    in_buf = self.buf_cur; in_stride = 576
                if has_msg:
                    d3, g3, f3 = L["ctrl3"]
                    c_stride = 0 if is_swa else 2304
                    vemit("ctrl3_p1", [in_buf, self.buf_msgs, d3, self.buf_gate3, self._u4(c_stride // 2, C)], (64, 3, ct8))
                    vemit("ctrl3_p2", [in_buf, self.buf_msgs, self.buf_gate3, g3, f3, self.buf_ctrl3, self._u4(c_stride, C)], (36, 3, ct8))
                    if is_swa:
                        nw = L["attn_norm"]
                        vemit("rms3", [self.buf_ctrl3, nw, nw, nw, self.buf_hq, self.buf_hk, self.buf_hv], (3, C, 1))
                    else:
                        vemit("rms3", [self.buf_ctrl3, L["q_norm"], L["k_norm"], L["v_norm"], self.buf_hq, self.buf_hk, self.buf_hv], (3, C, 1))
                    hk_buf, hv_buf = self.buf_hk, self.buf_hv
                else:
                    if is_swa:
                        vemit("rms", [in_buf, L["attn_norm"], self.buf_hq, self._u4(0, in_stride)], (1, C, 1))
                        hk_buf = hv_buf = self.buf_hq
                    else:
                        vemit("rms3", [in_buf, L["q_norm"], L["k_norm"], L["v_norm"], self.buf_hq, self.buf_hk, self.buf_hv], (3, C, 1))
                        hk_buf, hv_buf = self.buf_hk, self.buf_hv

            if "w_qrsgkv" in L:
                vemit("gemm", [L["w_qrsgkv"], self.buf_hq, self.buf_qrsg, self.buf_meta, u_gemm_8600], (4300, ct8, 1), barrier=1)
                k_src, v_src, k_off, v_off, kv_stride = self.buf_qrsg, self.buf_qrsg, 6552, 7576, 8600
            elif is_swa:
                vemit("gemm", [L["w_qrsg"], self.buf_hq, self.buf_qrsg, self.buf_meta, u_gemm_6552], (3276, ct8, 1), barrier=0)
                vemit("gemm", [L["wk"],     hk_buf,      self.buf_kbuf, self.buf_meta, u_gemm_1024], (512,  ct8, 1), barrier=0)
                vemit("gemm", [L["wv"],     hv_buf,      self.buf_vbuf, self.buf_meta, u_gemm_1024], (512,  ct8, 1), barrier=1)
                k_src, v_src, k_off, v_off, kv_stride = self.buf_kbuf, self.buf_vbuf, 0, 0, 1056
            else:
                vemit("gemm", [L["w_qrsg"], self.buf_hq, self.buf_qrsg, self.buf_meta, u_gemm_6552], (3276, ct8, 1), barrier=0)
                vemit("gemm", [L["wk_dyn"], hk_buf,      self.buf_kbuf, self.buf_meta, u_gemm_1056], (528,  ct8, 1), barrier=0)
                vemit("gemm", [L["wv_dyn"], hv_buf,      self.buf_vbuf, self.buf_meta, u_gemm_1056], (528,  ct8, 1), barrier=1)
                k_src, v_src, k_off, v_off, kv_stride = self.buf_kbuf, self.buf_vbuf, 0, 0, 1056

            vemit("d_save_ring", [k_src, self.b_ring_k, self._u4(slot, 1024, kv_stride, k_off)], (32, 1, 1), barrier=0)
            vemit("d_save_ring", [v_src, self.b_ring_v, self._u4(slot, 1024, kv_stride, v_off)], (32, 1, 1), barrier=0)

            c_idx = swa_map[slot] if is_swa else full_map[slot]
            k0_b, k1_b = (self.buf_k_swa0, self.buf_k_swa1) if is_swa else (self.buf_k_full0, self.buf_k_full1)
            v0_b, v1_b = (self.buf_v_swa0, self.buf_v_swa1) if is_swa else (self.buf_v_full0, self.buf_v_full1)

            vemit("p_qkvr_post", [self.buf_qrsg, k_src, v_src, L["q_clip"], L["r_clip"], L["k_sconv"], L["v_sconv"],
                                  self.buf_hist_k, self.buf_hist_v, k0_b, k1_b, v0_b, v1_b, self.buf_meta,
                                  self._u4((slot << 8) | c_idx, 0 if is_swa else 1, k_off, v_off)], (64, C, 1))
            vemit("p_shift_kv",  [self.buf_qrsg, k_src, v_src, self.buf_hist_k, self.buf_hist_v, self.buf_meta,
                                  self._u4(slot, 0, k_off, v_off)], (8, 1, 1), barrier=0)
            if is_swa:
                vemit("p_attn_sk", [self.buf_qrsg, L["rel_bias"], k0_b, k1_b, v0_b, v1_b, self.buf_part_ml, self.buf_part_ac,
                                    self.buf_meta, self._u4(c_idx, 1)], (24, 16, C))
            else:
                vemit("v_attn_sk", [self.buf_qrsg, L["rel_bias"], k0_b, k1_b, v0_b, v1_b, self.buf_part_ml, self.buf_part_ac,
                                    self.buf_meta, self._u4(c_idx, 0)], (24, 2, 16))
            vemit("attn_cb",     [self.buf_part_ml, self.buf_part_ac, self.buf_qrsg, self.buf_attn_ctx], (24, C, 1))
            vemit("gemm",        [L["wo"], self.buf_attn_ctx, self.buf_wo_raw, self.buf_meta, u_gemm_wo], (1152, ct8, 1))
            vemit("d_save_ring", [self.buf_wo_raw, self.b_ring_a, self._u4(slot, 2304, 2304, 0)], (72, 1, 1), barrier=0)

            add_r = 1
            if ip % 6 == 0:
                vemit("copy", [resid_buf, self.buf_blocks, self._u4(0, nb * 2304, 2304, 8 * 2304)], (9, C, 1), barrier=0)
                add_r = 0; nb += 1
            vemit("p_conv_resid", [self.buf_wo_raw, L["attn_sconv"], resid_buf, self.buf_hist_a, self.buf_ffn_inp, self.buf_meta, self._u4(slot, add_r)], (9, 1, 1))

            if has_msg:
                vemit("mix", [self.buf_blocks, self.buf_ffn_inp, L["mlp_q"], self.buf_ffn_mix, self._u4(8, nb, 1, 1)], (1, C, 1))
                wd, wg_w, wf_w = L["ctrl_mlp"]
                vemit("ctrl_p1", [self.buf_ffn_mix, self.buf_msgs, wd, self.buf_gate3, self._u4(0, 3 * 1152, C)], (64, 1, ct8))
                vemit("ctrl_p2", [self.buf_ffn_mix, self.buf_msgs, self.buf_gate3, wg_w, wf_w, self.buf_ffn_ctl, self._u4(0, 3 * 2304, 0, C)], (36, 1, ct8))
                vemit("rms", [self.buf_ffn_ctl, L["mlp_norm"], self.buf_ffn_nrm, u_rms_2304], (1, C, 1))
            else:
                vemit("mix_rms", [self.buf_blocks, self.buf_ffn_inp, L["mlp_q"], L["mlp_norm"], self.buf_ffn_nrm, self._u4(8, nb)], (1, C, 1))

            vemit("gemm",   [L["router_w"], self.buf_ffn_nrm, self.buf_rlogits, self.buf_meta, u_gemm_128], (64, ct8, 1))
            vemit("router", [self.buf_rlogits, L["router_b"], self.buf_eids, self.buf_wts], (1, C, 1))
            vemit("p_group_exp", [self.buf_eids, self.buf_exp_cnt, self.buf_exp_toks, self.buf_meta], (1, 1, 1))
            if L["exp_kind"] == "nvfp4":
                vemit("p_nv_w13_grp", [L["w13"], L["s13"], L["o13"], self.buf_exp_cnt, self.buf_exp_toks, self.buf_ffn_nrm, self.buf_moe_h], (576, 64, 1))
                vemit("p_nv_w2_grp",  [L["w2"],  L["s2"],  L["o2"],  self.buf_exp_cnt, self.buf_exp_toks, self.buf_wts, self.buf_moe_h, self.buf_moe_part], (1152, 64, 1))
            else:
                vemit("p_mx_w13_grp", [L["w13"], L["s13"], self.buf_exp_cnt, self.buf_exp_toks, self.buf_ffn_nrm, self.buf_moe_h], (576, 64, 1))
                vemit("p_mx_w2_grp",  [L["w2"],  L["s2"],  self.buf_exp_cnt, self.buf_exp_toks, self.buf_wts, self.buf_moe_h, self.buf_moe_part], (1152, 64, 1))
            vemit("p_moe_sum",    [self.buf_moe_part, self.buf_wo_raw, self.buf_meta], (9, C, 1))
            vemit("d_save_ring",  [self.buf_wo_raw, self.b_ring_m, self._u4(slot, 2304, 2304, 0)], (72, 1, 1), barrier=0)
            vemit("p_conv_resid", [self.buf_wo_raw, L["mlp_sconv"], self.buf_ffn_inp, self.buf_hist_m, self.buf_cur, self.buf_meta, self._u4(slot, 1)], (9, 1, 1))
            return nb

        nb = 0; visit = 0
        for il in range(6):
            if visit == 5: vemit("d_save_tap", [self.buf_cur, self.b_taps, self.buf_meta, u_taps[0]], (9, C, 1))
            visit += 1
            nb = vemit_layer(il, il, nb, False)
        prelude_nb = nb
        vemit("copy", [self.buf_cur, self.buf_prelude, u_cp_00], (9, C, 1))
        for p_idx in range(3):
            if p_idx > 0:
                n_bank = nb - (prelude_nb + 1)
                for b in range(n_bank):
                    vemit("copy", [self.buf_blocks, self.buf_bank, self._u4((prelude_nb + 1 + b) * 2304, b * 2304, 8 * 2304, 4 * 2304)], (9, C, 1), barrier=0)
                vemit("copy", [self.buf_cur, self.buf_bank_out, u_cp_00], (9, C, 1), barrier=1)
                vemit("copy", [self.buf_prelude, self.buf_cur, u_cp_00], (9, C, 1), barrier=1)
                nb = prelude_nb
            for ip in range(6, 18):
                if visit == 23: vemit("d_save_tap", [self.buf_cur, self.b_taps, self.buf_meta, u_taps[1]], (9, C, 1))
                if visit == 41: vemit("d_save_tap", [self.buf_cur, self.b_taps, self.buf_meta, u_taps[2]], (9, C, 1))
                visit += 1
                slot = ip if p_idx == 0 else 24 + (p_idx - 1) * 12 + (ip - 6)
                if p_idx > 0:
                    vemit("mix", [self.buf_bank, self.buf_bank_out, msg_q_bufs[ip - 6], self.buf_msgs, self._u4(4, n_bank, 4, 4)], (1, C, 1))
                nb = vemit_layer(ip, slot, nb, p_idx > 0)
        for il in range(18, 24):
            visit += 1
            nb = vemit_layer(il, il, nb, False)

        vemit("mix_rms",    [self.buf_blocks, self.buf_cur, b_out_q, b_out_norm, self.buf_ffn_nrm, self._u4(8, nb)], (1, C, 1))
        vemit("d_save_tap", [self.buf_ffn_nrm, self.b_taps, self.buf_meta, u_taps[3]], (9, C, 1))
        vemit("unembed_p1", [b_unembed, self.buf_ffn_nrm, self.buf_part_v, self.buf_part_i, self.buf_logits, self.buf_meta, b_unembed_sc], (self.n_unembed_wg, ct8, 1))
        vemit("argmax_p2",  [self.buf_part_v, self.buf_part_i, self.buf_logits, self.buf_meta, self.buf_tok_out], (1, C, 1), barrier=0)
        assert _vk.vk_record_end_slot(self.cmd_verify8) == 0

        # --- Command buffer 2: cmd_rb_inject ---
        assert _vk.vk_record_begin_slot(self.cmd_rb_inject) == 0
        def remit(pname, bufs, wg, barrier=1):
            arr = (ctypes.c_int * len(bufs))(*bufs)
            assert _vk.vk_record_dispatch_slot(self.cmd_rb_inject, self.pipes[pname], len(bufs), arr, wg[0], wg[1], wg[2], barrier) == 0

        remit("d_rb_hist", [self.b_snap_k, self.b_ring_k, self.buf_hist_k, self.buf_meta, u_1024], (4, 48, 1), barrier=0)
        remit("d_rb_hist", [self.b_snap_v, self.b_ring_v, self.buf_hist_v, self.buf_meta, u_1024], (4, 48, 1), barrier=0)
        remit("d_rb_hist", [self.b_snap_a, self.b_ring_a, self.buf_hist_a, self.buf_meta, u_2304], (9, 48, 1), barrier=0)
        remit("d_rb_hist", [self.b_snap_m, self.b_ring_m, self.buf_hist_m, self.buf_meta, u_2304], (9, 48, 1), barrier=0)

        u_dgemm_1024 = self._u4(144, 1024, 1024, 0)
        for il in range(3):
            ML = mtp_layers[il]
            is_swa = 1 if il in (0, 2) else 0
            c_idx = 0 if il == 0 else (1 if il == 2 else 0)
            dk_b = self.b_dk_swa if is_swa else self.b_dk_full
            dv_b = self.b_dv_swa if is_swa else self.b_dv_full
            remit("d_fuse",     [self.b_taps, b_fuse_rows[il], ML["dnorm"], self.b_ctx_tmp, self.buf_meta], (64, 1, 1))
            remit("gemm",       [ML["wkc"], self.b_ctx_tmp, self.b_ctx_k, self.buf_meta, u_dgemm_1024], (512, 8, 1), barrier=0)
            remit("gemm",       [ML["wvc"], self.b_ctx_tmp, self.b_ctx_v, self.buf_meta, u_dgemm_1024], (512, 8, 1), barrier=1)
            remit("d_qkn_rope", [self.b_ctx_k, ML["kc_norm"], self.b_d_qr, dk_b, self.buf_meta, self._u4(8, 1, c_idx, is_swa)], (8, 64, 1), barrier=0)
            remit("d_store_v",  [self.b_ctx_v, dv_b, self.buf_meta, self._u4(c_idx, is_swa)], (8, 64, 1), barrier=1)

        remit("d_fuse",     [self.b_taps, b_fuse_rows[3], b_mk_dnorm, self.b_ctx_tmp, self.buf_meta], (64, 1, 1))
        remit("d_copy_anc", [self.b_ctx_tmp, self.b_ctx_anc, self.buf_meta], (9, 1, 1), barrier=0)
        assert _vk.vk_record_end_slot(self.cmd_rb_inject) == 0

        # --- Command buffer 3: cmd_draft7 ---
        assert _vk.vk_record_begin_slot(self.cmd_draft7) == 0
        def demit(pname, bufs, wg, barrier=1):
            arr = (ctypes.c_int * len(bufs))(*bufs)
            assert _vk.vk_record_dispatch_slot(self.cmd_draft7, self.pipes[pname], len(bufs), arr, wg[0], wg[1], wg[2], barrier) == 0

        u_dgemm_3072 = self._u4(144, 3072, 3072, 0)
        u_dgemm_9216 = self._u4(144, 9216, 9216, 0)
        u_dgemm_down = self._u4(576, 2304, 2304, 0)

        demit("copy", [self.buf_tmp, self.b_d_cur, u_cp_00], (9, 1, 1), barrier=0)
        demit("copy", [b_mask_emb, self.b_d_cur, self._u4(0, 2304, 0, 2304)], (9, 7, 1), barrier=1)

        demit("rms", [self.b_d_cur, mtp_layers[0]["inorm"], self.b_d_h, u_rms_2304], (1, 8, 1))
        for il in range(3):
            ML = mtp_layers[il]
            is_swa = 1 if il in (0, 2) else 0
            c_idx = 0 if il == 0 else (1 if il == 2 else 0)
            dk_b = self.b_dk_swa if is_swa else self.b_dk_full
            dv_b = self.b_dv_swa if is_swa else self.b_dv_full
            demit("gemm",       [ML["wq"], self.b_d_h, self.b_d_q, self.buf_meta, u_dgemm_3072], (1536, 1, 1), barrier=0)
            demit("gemm",       [ML["wk"], self.b_d_h, self.b_d_k, self.buf_meta, u_dgemm_1024], (512,  1, 1), barrier=0)
            demit("gemm",       [ML["wv"], self.b_d_h, self.b_d_v, self.buf_meta, u_dgemm_1024], (512,  1, 1), barrier=1)
            demit("d_qkn_rope", [self.b_d_q, ML["q_norm"], self.b_d_qr, self.b_dk_swa, self.buf_meta, self._u4(24, 0, 0, 0)], (24, 8, 1), barrier=0)
            demit("d_qkn_rope", [self.b_d_k, ML["k_norm"], self.b_d_kr, self.b_dk_swa, self.buf_meta, self._u4(8,  0, 0, 0)], (8,  8, 1), barrier=1)
            demit("d_attn_sk",  [self.b_d_qr, self.b_d_kr, self.b_d_v, dk_b, dv_b, self.buf_part_ml, self.buf_part_ac, self.buf_meta, self._u4(c_idx, is_swa)], (24, 2, 16))
            demit("d_attn_cb",  [self.buf_part_ml, self.buf_part_ac, self.b_d_ao], (24, 8, 1))
            demit("gemm",       [ML["wo"], self.b_d_ao, self.b_d_oo, self.buf_meta, u_gemm_wo], (1152, 1, 1))
            demit("d_add",      [self.b_d_cur, self.b_d_oo, ML["pnorm"], self.b_d_h], (1, 8, 1))

            demit("d_swiglu",   [ML["wgate"], ML["wup"], self.b_d_h, self.b_d_mg], (4608, 1, 1))
            demit("gemm",       [ML["wdown"], self.b_d_mg, self.b_d_md, self.buf_meta, u_dgemm_down], (1152, 1, 1))
            nxt_norm = mtp_layers[il + 1]["inorm"] if il < 2 else b_mtp_norm
            nxt_out  = self.b_d_h if il < 2 else self.b_d_norm
            demit("d_add",      [self.b_d_cur, self.b_d_md, nxt_norm, nxt_out], (1, 8, 1))
        for step in range(1, 8):
            u_st = self._u4(step)
            demit("d_mrnn",     [self.b_mk_state, b_mk_w1, self.b_d_norm, self.b_ctx_anc, b_mk_rnn_w, b_mk_rnn_b, self.b_d_toks, self.b_mk_jout, self.buf_meta, u_st], (768, 1, 1))
            demit("d_mupd",     [self.b_mk_jout, self.b_mk_state, self.b_d_norm, b_mk_ho_w, b_mk_ho_b, self.b_mk_comb, u_st], (9, 1, 1))
            demit("d_attn", [b_unembed, self.b_mk_comb, self.buf_part_v, self.buf_part_i, b_unembed_sc], (12504, 1, 1))
            demit("d_am_slot",  [self.buf_part_v, self.buf_part_i, self.b_d_toks, u_st], (1, 1, 1), barrier=(0 if step == 7 else 1))
        assert _vk.vk_record_end_slot(self.cmd_draft7) == 0

    def _mtp_inject_only(self, seq_slot: int, start_pos: int, n_toks: int):
        self.meta_arr[0] = n_toks
        self.meta_arr[1] = seq_slot
        self.meta_arr[2] = start_pos
        self.meta_arr[4] = n_toks
        self.meta_arr[5] = 1
        assert _vk.vk_run_recorded_slot(self.cmd_rb_inject) == 0

    def _mtp_draft7(self, seq_slot: int, anchor_id: int, ctx_len: int):
        self.tmp_arr[0] = self._embed_text(anchor_id)
        self.mk_state_arr[:] = 0.0
        self.d_toks_arr[0] = anchor_id
        self.meta_arr[0] = 8
        self.meta_arr[1] = seq_slot
        self.meta_arr[2] = ctx_len
        assert _vk.vk_run_recorded_slot(self.cmd_draft7) == 0
        return [int(self.d_toks_arr[i]) for i in range(1, 8)]

    def _mtp_verify8(self, seq_slot: int, toks8, start_pos: int):
        self.meta_arr[0] = 8
        self.meta_arr[1] = seq_slot
        self.meta_arr[2] = start_pos
        self.meta_arr[48:56] = 0
        self.tmp_arr[:8] = self._embed_text(toks8)
        assert _vk.vk_run_recorded_slot(self.cmd_verify8) == 0
        preds = [int(self.out_arr[i, 0]) for i in range(8)]
        scores = [struct.unpack("<f", struct.pack("<I", int(self.out_arr[i, 1])))[0] for i in range(8)]
        return preds, scores

    def _mtp_rollback_inject(self, seq_slot: int, start_pos: int, n_accept: int):
        self.meta_arr[0] = n_accept
        self.meta_arr[1] = seq_slot
        self.meta_arr[2] = start_pos
        self.meta_arr[4] = n_accept
        self.meta_arr[5] = 0 if n_accept < 8 else 1
        assert _vk.vk_run_recorded_slot(self.cmd_rb_inject) == 0

    def step_speculative(self, seq_slot: int, anchor_id: int, pos: int, adaptive: bool = True):
        if (not self.has_mtp) or (not self.mtp_valid[seq_slot]) or (pos + 8 >= self.seq_cap):
            return self.step_batch([seq_slot], [anchor_id], [pos])
        if adaptive and self.mtp_skip[seq_slot] > 0:
            self.mtp_skip[seq_slot] -= 1
            return self.step_batch([seq_slot], [anchor_id], [pos])
        draft7 = self._mtp_draft7(seq_slot, anchor_id, pos)
        block = [anchor_id] + draft7
        preds, scores = self._mtp_verify8(seq_slot, block, pos)
        n_match = 0
        for j in range(7):
            if draft7[j] == preds[j]:
                n_match += 1
            else:
                break
        n_accept = n_match + 1
        self._mtp_rollback_inject(seq_slot, pos, n_accept)
        if adaptive:
            ema = 0.7 * self.mtp_ema[seq_slot] + 0.3 * float(n_accept)
            self.mtp_ema[seq_slot] = ema
            if ema < 3.4 and n_accept <= 3:
                self.mtp_skip[seq_slot] = 16
        return [(preds[j], scores[j]) for j in range(n_accept)]

    def reset_slot(self, seq_slot: int):
        sz_kv = 48 * 3 * 1024 * 4
        sz_am = 48 * 3 * 2304 * 4
        _vk.vk_zero_slice(self.buf_hist_k, seq_slot * sz_kv, sz_kv)
        _vk.vk_zero_slice(self.buf_hist_v, seq_slot * sz_kv, sz_kv)
        _vk.vk_zero_slice(self.buf_hist_a, seq_slot * sz_am, sz_am)
        _vk.vk_zero_slice(self.buf_hist_m, seq_slot * sz_am, sz_am)
        self.mtp_valid[seq_slot] = self.has_mtp
        self.mtp_ema[seq_slot]   = 4.5
        self.mtp_skip[seq_slot]  = 0

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
        anc = None
        if self.has_mtp:
            anc = np.empty(2304, dtype=np.float32)
            assert _vk.vk_download_slice(self.b_ctx_anc, seq_slot * 2304 * 4, anc.ctypes.data, 2304 * 4) == 0
        return (hk, hv, ha, hm, anc, self.mtp_valid[seq_slot])

    def restore_slot_conv(self, seq_slot: int, snap):
        hk, hv, ha, hm = snap[:4]
        sz_kv = 48 * 3 * 1024 * 4
        sz_am = 48 * 3 * 2304 * 4
        assert _vk.vk_upload_slice(self.buf_hist_k, seq_slot * sz_kv, hk.ctypes.data, sz_kv) == 0
        assert _vk.vk_upload_slice(self.buf_hist_v, seq_slot * sz_kv, hv.ctypes.data, sz_kv) == 0
        assert _vk.vk_upload_slice(self.buf_hist_a, seq_slot * sz_am, ha.ctypes.data, sz_am) == 0
        assert _vk.vk_upload_slice(self.buf_hist_m, seq_slot * sz_am, hm.ctypes.data, sz_am) == 0
        if self.has_mtp and len(snap) >= 6 and snap[4] is not None:
            anc = snap[4]
            assert _vk.vk_upload_slice(self.b_ctx_anc, seq_slot * 2304 * 4, anc.ctypes.data, 2304 * 4) == 0
            self.mtp_valid[seq_slot] = bool(snap[5])
        else:
            self.mtp_valid[seq_slot] = False

    def prefill_chunk(self, seq_slot: int, token_ids, start_pos: int, embed_overrides=None):
        c = len(token_ids)
        assert 1 <= c <= 64
        cmd_slot = len(self.buckets) + (1 if c <= 16 else 0) # 6 for <=16, 5 for 17..64
        self.meta_arr[0] = c
        self.meta_arr[1] = seq_slot
        self.meta_arr[2] = start_pos
        emb = self._embed_text(token_ids)
        if embed_overrides:
            for idx, vec in embed_overrides.items():
                emb[idx] = vec
        self.tmp_arr[:c] = emb
        assert _vk.vk_run_recorded_slot(cmd_slot) == 0
        if self.has_mtp and self.mtp_valid[seq_slot]:
            self._mtp_inject_only(seq_slot, start_pos, c)

    def prefill_tokens(self, seq_slot: int, token_ids, start_pos: int = 0, embed_overrides=None):
        pos = start_pos
        i = 0
        n = len(token_ids)
        while i < n:
            chunk = token_ids[i : i + 64]
            c_ovr = None
            if embed_overrides:
                end = i + len(chunk)
                c_ovr = {k - i: v for k, v in embed_overrides.items() if i <= k < end}
            self.prefill_chunk(seq_slot, chunk, pos, embed_overrides=c_ovr)
            pos += len(chunk)
            i += len(chunk)
        return pos

    def step_batch(self, seq_slots, token_ids, positions, temps=None, top_ps=None, seeds=None, embed_overrides=None):
        n = len(seq_slots)
        assert 1 <= n <= self.max_batch
        cmd_slot = 0
        for i, B in enumerate(self.buckets):
            if B >= n:
                cmd_slot = i; break
        B = self.buckets[cmd_slot]

        dummy_slot = 0
        if n < B:
            used = set(seq_slots)
            for s in range(self.max_batch):
                if s not in used:
                    dummy_slot = s; break

        self.meta_arr[0] = n
        for bi in range(B):
            if bi < n:
                if embed_overrides is not None and bi in embed_overrides:
                    self.tmp_arr[bi] = embed_overrides[bi]
                else:
                    self.tmp_arr[bi] = self._embed_text(token_ids[bi])
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
        if self.has_mtp:
            if n == 1 and B == 1 and self.mtp_valid[seq_slots[0]]:
                self._mtp_inject_only(seq_slots[0], positions[0], 1)
            else:
                for s in seq_slots:
                    self.mtp_valid[s] = False
        res = []
        for bi in range(n):
            tid = int(self.out_arr[bi, 0])
            sc  = struct.unpack("<f", struct.pack("<I", int(self.out_arr[bi, 1])))[0]
            res.append((tid, sc))
        return res

    def _prefix_copy_ops(self, seq_slot: int, prefix_slot: int, pos: int, save: bool):
        assert 0 <= prefix_slot < self.n_prefix_slots
        assert 1 <= pos <= self.prefix_cap
        ps = self.prefix_pool[prefix_slot]
        hi = seq_slot >= self.half_b
        hs = seq_slot - self.half_b if hi else seq_slot
        k_swa_b  = self.buf_k_swa1  if hi else self.buf_k_swa0
        v_swa_b  = self.buf_v_swa1  if hi else self.buf_v_swa0
        k_full_b = self.buf_k_full1 if hi else self.buf_k_full0
        v_full_b = self.buf_v_full1 if hi else self.buf_v_full0
        swa_sz   = 40 * SWA_WIN * 1024
        sz_kv    = 48 * 3 * 1024 * 4
        sz_am    = 48 * 3 * 2304 * 4
        ops = []  # (slot_buf, slot_off, pref_buf, pref_off, nbytes)
        ops.append((k_swa_b, hs * swa_sz, ps["k_swa"], 0, swa_sz))
        ops.append((v_swa_b, hs * swa_sz, ps["v_swa"], 0, swa_sz))
        full_cp = pos * 1024
        for li in range(8):
            s_off = (hs * 8 + li) * self.seq_cap * 1024
            p_off = li * self.prefix_cap * 1024
            ops.append((k_full_b, s_off, ps["k_full"], p_off, full_cp))
            ops.append((v_full_b, s_off, ps["v_full"], p_off, full_cp))
        ops.append((self.buf_hist_k, seq_slot * sz_kv, ps["hist_k"], 0, sz_kv))
        ops.append((self.buf_hist_v, seq_slot * sz_kv, ps["hist_v"], 0, sz_kv))
        ops.append((self.buf_hist_a, seq_slot * sz_am, ps["hist_a"], 0, sz_am))
        ops.append((self.buf_hist_m, seq_slot * sz_am, ps["hist_m"], 0, sz_am))
        if self.has_mtp:
            dswa_sz = 2 * SWA_WIN * 2048
            dfull_cp = pos * 2048
            ops.append((self.b_dk_swa,  seq_slot * dswa_sz,          ps["dk_swa"],  0, dswa_sz))
            ops.append((self.b_dv_swa,  seq_slot * dswa_sz,          ps["dv_swa"],  0, dswa_sz))
            ops.append((self.b_dk_full, seq_slot * self.seq_cap * 2048, ps["dk_full"], 0, dfull_cp))
            ops.append((self.b_dv_full, seq_slot * self.seq_cap * 2048, ps["dv_full"], 0, dfull_cp))
            ops.append((self.b_ctx_anc, seq_slot * 2304 * 4,         ps["ctx_anc"], 0, 2304 * 4))
        n = len(ops)
        db = (ctypes.c_int * n)()
        do = (ctypes.c_uint64 * n)()
        sb = (ctypes.c_int * n)()
        so = (ctypes.c_uint64 * n)()
        sz = (ctypes.c_uint64 * n)()
        for i, (s_b, s_o, p_b, p_o, nb) in enumerate(ops):
            if save:
                db[i], do[i], sb[i], so[i], sz[i] = p_b, p_o, s_b, s_o, nb
            else:
                db[i], do[i], sb[i], so[i], sz[i] = s_b, s_o, p_b, p_o, nb
        t0 = time.perf_counter()
        assert _vk.vk_copy_batch(n, db, do, sb, so, sz) == 0
        return (time.perf_counter() - t0) * 1e3

    def save_vram_prefix(self, seq_slot: int, prefix_slot: int, pos: int) -> float:
        return self._prefix_copy_ops(seq_slot, prefix_slot, pos, save=True)

    def load_vram_prefix(self, seq_slot: int, prefix_slot: int, pos: int) -> float:
        ms = self._prefix_copy_ops(seq_slot, prefix_slot, pos, save=False)
        self.mtp_valid[seq_slot] = self.has_mtp
        self.mtp_ema[seq_slot]   = 4.5
        self.mtp_skip[seq_slot]  = 0
        return ms

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
    print(f"BatchedVulkanEngine ready | buckets={BUCKETS} | SEQ_CAP={SEQ_CAP} | MTP={eng.has_mtp} | init={eng.init_s:.2f}s{mem_str}\n", flush=True)

    eng.reset_slot(0)
    eng.step_batch([0], [32269], [0])
    cur = 150770; t5, s5 = [], []
    for p in range(1, 6):
        (cur, sc), = eng.step_batch([0], [cur], [p])
        t5.append(cur); s5.append(round(sc, 4))
    print(f"B=1 5-tok check: {t5}  scores={s5}", flush=True)
    eng.close()
