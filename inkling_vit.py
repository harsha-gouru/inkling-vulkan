"""Pure-NumPy + Pillow CPU Vision Encoder for Inkling-2-Mini (Inkling2V).

Zero VRAM, zero PyTorch dependency. Matches llama.cpp tools/mtmd/models/inkling2v.cpp
and mtmd-image.cpp (mtmd_image_preprocessor_inkling2v) exactly.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import mmap
import os
import struct
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

DEFAULT_MODEL_PATH = str(Path(os.environ.get("INKLING_MODEL_DIR", "./Inkling-2-Mini")) / "model-00001-of-00001.safetensors")


def _bf16_to_f32(raw: memoryview, shape: tuple[int, ...]) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype="<u2")
    return (u16.astype(np.uint32) << 16).view(np.float32).reshape(shape)


def _layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    diff = x - mean
    var = np.mean(diff * diff, axis=-1, keepdims=True)
    diff *= np.float32(1.0) / np.sqrt(var + np.float32(eps))
    diff *= w
    diff += b
    return diff


def _gelu_erf(x: np.ndarray) -> np.ndarray:
    """Vectorized exact erf-GELU via Abramowitz & Stegun 7.1.26 (max err < 1.5e-7)."""
    ax = np.abs(x)
    ax *= np.float32(0.7071067811865475)
    t = np.float32(1.0) / (np.float32(1.0) + np.float32(0.3275911) * ax)
    poly = t * (
        np.float32(0.254829592)
        + t * (
            np.float32(-0.284496736)
            + t * (
                np.float32(1.421413741)
                + t * (np.float32(-1.453152027) + t * np.float32(1.061405429))
            )
        )
    )
    ax *= ax
    np.negative(ax, out=ax)
    np.exp(ax, out=ax)
    poly *= ax
    erf_ax = np.float32(1.0) - poly
    erf_x = np.copysign(erf_ax, x)
    erf_x += np.float32(1.0)
    erf_x *= np.float32(0.5)
    erf_x *= x
    return erf_x


def _bilinear_resize_pos_emb(pos_emb: np.ndarray, dst_h: int, dst_w: int) -> np.ndarray:
    """Half-pixel bilinear interpolation matching ggml_interpolate(GGML_SCALE_MODE_BILINEAR).

    pos_emb: float32 [src_h, src_w, C] -> float32 [dst_h, dst_w, C]
    """
    src_h, src_w, _ = pos_emb.shape
    if src_h == dst_h and src_w == dst_w:
        return pos_emb

    sf_h = dst_h / float(src_h)
    sf_w = dst_w / float(src_w)

    y = (np.arange(dst_h, dtype=np.float32) + 0.5) / sf_h - 0.5
    y0_raw = np.floor(y)
    y0 = np.clip(y0_raw.astype(np.int64), 0, src_h - 1)
    y1 = np.clip((y0_raw + 1).astype(np.int64), 0, src_h - 1)
    dy = np.clip(y - y0.astype(np.float32), 0.0, 1.0)[:, None, None]

    x = (np.arange(dst_w, dtype=np.float32) + 0.5) / sf_w - 0.5
    x0_raw = np.floor(x)
    x0 = np.clip(x0_raw.astype(np.int64), 0, src_w - 1)
    x1 = np.clip((x0_raw + 1).astype(np.int64), 0, src_w - 1)
    dx = np.clip(x - x0.astype(np.float32), 0.0, 1.0)[None, :, None]

    row_interp = pos_emb[:, x0, :] * (1.0 - dx) + pos_emb[:, x1, :] * dx
    return row_interp[y0, :, :] * (1.0 - dy) + row_interp[y1, :, :] * dy


def _apply_rope_2d_inplace(t: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> None:
    """Apply 2D axial RoPE (mode=0 adjacent pairs) in-place to [n_pos, n_heads, 64]."""
    t0 = t[..., 0::2].copy()
    t1 = t[..., 1::2]
    t[..., 0::2] = t0 * cos - t1 * sin
    t[..., 1::2] = t0 * sin + t1 * cos


class InklingViT:
    """CPU Vision Encoder for Inkling-2-Mini (23-layer ViT + 2x2 merge + 2-layer GELU projector)."""

    PATCH_SIZE = 20
    MERGE_SIZE = 2
    ALIGN = 40  # PATCH_SIZE * MERGE_SIZE
    HIDDEN_SIZE = 1024
    NUM_LAYERS = 23
    NUM_HEADS = 16
    HEAD_DIM = 64
    POS_GRID = 32
    LN_EPS = 1e-5
    ROPE_THETA = 10000.0
    OUT_DIM = 2304

    def __init__(
        self,
        safetensors_path: str = DEFAULT_MODEL_PATH,
        max_pixels: int = 1024 * 1024,
        cache_size: int = 16,
        lazy: bool = True,
    ):
        self.safetensors_path = str(safetensors_path)
        self.max_pixels = int(max_pixels)
        self.cache_size = int(cache_size)
        self._load_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._loaded = False

        self.class_plus_pos: np.ndarray | None = None
        self.patch_emb_T: np.ndarray | None = None
        self.pos_emb_grid: np.ndarray | None = None
        self.ln_pre_w: np.ndarray | None = None
        self.ln_pre_b: np.ndarray | None = None
        self.layers: list[dict[str, np.ndarray]] = []
        self.proj_in_T: np.ndarray | None = None
        self.proj_out_T: np.ndarray | None = None

        if not lazy:
            self.load()

    def load(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            with open(self.safetensors_path, "rb") as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                try:
                    hdr_len = struct.unpack("<Q", mm[:8])[0]
                    hdr = json.loads(mm[8 : 8 + hdr_len])
                    base = 8 + hdr_len
                    mv = memoryview(mm)

                    def get(name: str, transpose: bool = False) -> np.ndarray:
                        info = hdr[name]
                        s, e = info["data_offsets"]
                        arr = _bf16_to_f32(mv[base + s : base + e], tuple(info["shape"]))
                        return np.ascontiguousarray(arr.T if transpose else arr)

                    cls_emb = get("model.visual.backbone.embeddings.class_embedding")
                    cls_pos = get("model.visual.backbone.embeddings.cls_position_embedding")
                    self.class_plus_pos = np.ascontiguousarray((cls_emb + cls_pos)[None, :])
                    self.patch_emb_T = get("model.visual.backbone.embeddings.patch_embedding.weight", transpose=True)
                    self.pos_emb_grid = get("model.visual.backbone.embeddings.position_embedding").reshape(
                        self.POS_GRID, self.POS_GRID, self.HIDDEN_SIZE
                    )
                    self.ln_pre_w = get("model.visual.backbone.ln_pre.weight")
                    self.ln_pre_b = get("model.visual.backbone.ln_pre.bias")

                    layers = []
                    for i in range(self.NUM_LAYERS):
                        p = f"model.visual.backbone.layers.{i}"
                        qw = get(f"{p}.self_attn.q_proj.weight")
                        kw = get(f"{p}.self_attn.k_proj.weight")
                        vw = get(f"{p}.self_attn.v_proj.weight")
                        qkv_wT = np.ascontiguousarray(np.concatenate([qw, kw, vw], axis=0).T)
                        qkv_b = np.ascontiguousarray(np.concatenate([
                            get(f"{p}.self_attn.q_proj.bias"),
                            get(f"{p}.self_attn.k_proj.bias"),
                            get(f"{p}.self_attn.v_proj.bias"),
                        ], axis=0))
                        layers.append({
                            "ln1_w": get(f"{p}.ln_1.weight"),
                            "ln1_b": get(f"{p}.ln_1.bias"),
                            "qkv_wT": qkv_wT,
                            "qkv_b": qkv_b,
                            "o_wT": get(f"{p}.self_attn.out_proj.weight", transpose=True),
                            "o_b": get(f"{p}.self_attn.out_proj.bias"),
                            "ls1": get(f"{p}.ls_1.gamma"),
                            "ln2_w": get(f"{p}.ln_2.weight"),
                            "ln2_b": get(f"{p}.ln_2.bias"),
                            "fc_wT": get(f"{p}.mlp.c_fc.weight", transpose=True),
                            "fc_b": get(f"{p}.mlp.c_fc.bias"),
                            "proj_wT": get(f"{p}.mlp.c_proj.weight", transpose=True),
                            "proj_b": get(f"{p}.mlp.c_proj.bias"),
                            "ls2": get(f"{p}.ls_2.gamma"),
                        })
                    self.layers = layers
                    self.proj_in_T = get("model.visual.proj_in.weight", transpose=True)
                    self.proj_out_T = get("model.visual.proj_out.weight", transpose=True)
                    mv.release()
                finally:
                    mm.close()
            self._loaded = True

    def _preprocess(self, img: Image.Image) -> tuple[np.ndarray, int, int]:
        img = img.convert("RGB")
        w, h = img.size
        align = self.ALIGN
        aw = ((w + align - 1) // align) * align
        ah = ((h + align - 1) // align) * align

        if self.max_pixels > 0 and aw * ah > self.max_pixels:
            scale = math.sqrt(self.max_pixels / float(w * h))
            nw = max(align, int(math.floor((w * scale) / align)) * align)
            nh = max(align, int(math.floor((h * scale) / align)) * align)
            while nw * nh > self.max_pixels and (nw > align or nh > align):
                if nw >= nh and nw > align:
                    nw -= align
                elif nh > align:
                    nh -= align
                else:
                    break
            img = img.resize((nw, nh), Image.Resampling.BICUBIC)
            w, h = nw, nh
            aw, ah = w, h

        padded = np.full((ah, aw, 3), -1.0, dtype=np.float32)
        padded[:h, :w, :] = np.asarray(img, dtype=np.float32)
        padded -= np.float32(127.5)
        padded *= np.float32(1.0 / 127.5)
        return padded, ah // self.PATCH_SIZE, aw // self.PATCH_SIZE

    def _build_rope_tables(self, p_h: int, p_w: int) -> tuple[np.ndarray, np.ndarray]:
        n_pos = 1 + p_h * p_w
        pos_w = np.empty(n_pos, dtype=np.float32)
        pos_h = np.empty(n_pos, dtype=np.float32)
        pos_w[0] = 0.0
        pos_h[0] = 0.0
        yy, xx = np.meshgrid(
            np.arange(1, p_h + 1, dtype=np.float32),
            np.arange(1, p_w + 1, dtype=np.float32),
            indexing="ij",
        )
        pos_w[1:] = xx.ravel()
        pos_h[1:] = yy.ravel()

        half_d = self.HEAD_DIM // 2  # 32
        inv_freq = (1.0 / (self.ROPE_THETA ** (np.arange(0, half_d, 2, dtype=np.float32) / float(half_d)))).astype(np.float32)
        ang = np.concatenate([np.outer(pos_w, inv_freq), np.outer(pos_h, inv_freq)], axis=1)
        cos = np.cos(ang)[:, None, :].astype(np.float32)
        sin = np.sin(ang)[:, None, :].astype(np.float32)
        return cos, sin

    def _forward_pil(self, img: Image.Image) -> np.ndarray:
        self.load()
        padded, p_h, p_w = self._preprocess(img)
        ps = self.PATCH_SIZE
        n_patches = p_h * p_w
        n_pos = n_patches + 1
        d = self.HIDDEN_SIZE

        patches = (
            padded.reshape(p_h, ps, p_w, ps, 3)
            .transpose(0, 2, 1, 3, 4)
            .reshape(n_patches, ps * ps * 3)
        )
        x = patches @ self.patch_emb_T

        pos_emb = _bilinear_resize_pos_emb(self.pos_emb_grid, p_h, p_w).reshape(n_patches, d)
        x += pos_emb
        x = np.concatenate([self.class_plus_pos, x], axis=0)
        x = _layer_norm(x, self.ln_pre_w, self.ln_pre_b, self.LN_EPS)

        cos, sin = self._build_rope_tables(p_h, p_w)
        inv_sqrt_d = np.float32(1.0 / math.sqrt(self.HEAD_DIM))
        nh = self.NUM_HEADS
        hd = self.HEAD_DIM

        attn_buf = np.empty((nh, n_pos, hd), dtype=np.float32)
        for L in self.layers:
            h = _layer_norm(x, L["ln1_w"], L["ln1_b"], self.LN_EPS)
            qkv = h @ L["qkv_wT"]
            qkv += L["qkv_b"]
            q = qkv[:, :d].reshape(n_pos, nh, hd)
            k = qkv[:, d : 2 * d].reshape(n_pos, nh, hd)
            v = qkv[:, 2 * d :].reshape(n_pos, nh, hd)

            _apply_rope_2d_inplace(q, cos, sin)
            _apply_rope_2d_inplace(k, cos, sin)

            qh_all = np.ascontiguousarray(q.transpose(1, 0, 2))
            kh_all = np.ascontiguousarray(k.transpose(1, 0, 2))
            vh_all = np.ascontiguousarray(v.transpose(1, 0, 2))
            for hi in range(nh):
                scores = qh_all[hi] @ kh_all[hi].T
                scores *= inv_sqrt_d
                scores -= scores.max(axis=-1, keepdims=True)
                np.exp(scores, out=scores)
                scores /= scores.sum(axis=-1, keepdims=True)
                attn_buf[hi] = scores @ vh_all[hi]

            o = attn_buf.transpose(1, 0, 2).reshape(n_pos, d) @ L["o_wT"]
            o += L["o_b"]
            o *= L["ls1"]
            x += o

            h = _layer_norm(x, L["ln2_w"], L["ln2_b"], self.LN_EPS)
            fc = h @ L["fc_wT"]
            fc += L["fc_b"]
            m = _gelu_erf(fc) @ L["proj_wT"]
            m += L["proj_b"]
            m *= L["ls2"]
            x += m

        x = x[1:]
        ms = self.MERGE_SIZE
        x = (
            x.reshape(p_h // ms, ms, p_w // ms, ms, d)
            .transpose(0, 2, 1, 3, 4)
            .reshape((p_h // ms) * (p_w // ms), d * ms * ms)
        )
        x = _gelu_erf(x @ self.proj_in_T)
        return np.ascontiguousarray(x @ self.proj_out_T, dtype=np.float32)

    def _cache_get(self, key: str) -> np.ndarray | None:
        if self.cache_size <= 0:
            return None
        with self._cache_lock:
            val = self._cache.get(key)
            if val is not None:
                self._cache.move_to_end(key)
            return val

    def _cache_put(self, key: str, val: np.ndarray) -> None:
        if self.cache_size <= 0:
            return
        with self._cache_lock:
            self._cache[key] = val
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

    def encode_image(self, pil_or_bytes: Union[Image.Image, bytes, bytearray, memoryview, str, Path]) -> np.ndarray:
        """Encode a PIL Image, raw image bytes, or file path into float32[N, 2304] embeddings."""
        if isinstance(pil_or_bytes, (str, Path)):
            raw = Path(pil_or_bytes).read_bytes()
            key = hashlib.sha256(raw).hexdigest()
            cached = self._cache_get(key)
            if cached is not None:
                return cached
            with Image.open(io.BytesIO(raw)) as img:
                out = self._forward_pil(img)
            self._cache_put(key, out)
            return out

        if isinstance(pil_or_bytes, (bytes, bytearray, memoryview)):
            raw = bytes(pil_or_bytes)
            key = hashlib.sha256(raw).hexdigest()
            cached = self._cache_get(key)
            if cached is not None:
                return cached
            with Image.open(io.BytesIO(raw)) as img:
                out = self._forward_pil(img)
            self._cache_put(key, out)
            return out

        if isinstance(pil_or_bytes, Image.Image):
            rgb = pil_or_bytes.convert("RGB")
            h = hashlib.sha256()
            h.update(f"{rgb.width}x{rgb.height}:".encode("ascii"))
            h.update(rgb.tobytes())
            key = h.hexdigest()
            cached = self._cache_get(key)
            if cached is not None:
                return cached
            out = self._forward_pil(rgb)
            self._cache_put(key, out)
            return out

        raise TypeError(f"Unsupported image input type: {type(pil_or_bytes)!r}")

    def encode_data_url(self, url: str) -> tuple[str, np.ndarray]:
        """Decode a data:image/...;base64,... URL (or raw base64 / local file path).

        Returns (sha256_hex, float32[N, 2304]).
        """
        s = url.strip()
        if s.startswith("data:"):
            comma = s.find(",")
            if comma < 0:
                raise ValueError("Malformed data URL: missing comma")
            header = s[:comma].lower()
            payload = s[comma + 1 :]
            if ";base64" in header:
                raw = base64.b64decode(payload)
            else:
                raw = payload.encode("utf-8")
        elif s.startswith("file://"):
            raw = Path(s[7:]).read_bytes()
        elif os.path.isfile(s):
            raw = Path(s).read_bytes()
        else:
            raw = base64.b64decode(s)

        sha = hashlib.sha256(raw).hexdigest()
        cached = self._cache_get(sha)
        if cached is not None:
            return sha, cached

        with Image.open(io.BytesIO(raw)) as img:
            out = self._forward_pil(img)
        self._cache_put(sha, out)
        return sha, out


if __name__ == "__main__":
    import sys
    import time

    t_l0 = time.perf_counter()
    vit = InklingViT(lazy=False)
    t_l1 = time.perf_counter()
    t0 = time.perf_counter()
    if len(sys.argv) > 1:
        path = sys.argv[1]
        raw = Path(path).read_bytes()
        url = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
        sha, emb = vit.encode_data_url(url)
    else:
        arr = np.zeros((320, 480, 3), dtype=np.uint8)
        arr[:, :, 0] = np.linspace(0, 255, 480, dtype=np.uint8)[None, :]
        arr[:, :, 1] = np.linspace(0, 255, 320, dtype=np.uint8)[:, None]
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        sha, emb = vit.encode_data_url(url)
    t1 = time.perf_counter()
    sha2, emb2 = vit.encode_data_url(url)
    t2 = time.perf_counter()
    assert sha == sha2 and emb is emb2
    assert emb.dtype == np.float32 and emb.ndim == 2 and emb.shape[1] == 2304
    assert np.isfinite(emb).all()
    print(
        f"OK: shape={emb.shape} dtype={emb.dtype} finite=True "
        f"mean={emb.mean():.4f} std={emb.std():.4f} norm={np.linalg.norm(emb, axis=-1).mean():.2f} "
        f"load_ms={(t_l1 - t_l0) * 1e3:.1f} cold_ms={(t1 - t0) * 1e3:.1f} "
        f"cache_hit_ms={(t2 - t1) * 1e3:.3f} sha={sha[:12]}"
    )
