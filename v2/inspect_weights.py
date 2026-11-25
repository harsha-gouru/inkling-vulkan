#!/usr/bin/env python3
"""Stage 0 verifier: reads Inkling-2-Mini safetensors headers by hand (stdlib only)."""
import json
import os
import re
import struct
from collections import Counter, defaultdict

MODEL_DIR = os.path.expanduser("~/models/Inkling-2-Mini")


def read_header(path):
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(header_len).decode("utf-8"))
        data_start = 8 + header_len
    hdr.pop("__metadata__", None)
    return data_start, hdr


def summarize_file(fname):
    path = os.path.join(MODEL_DIR, fname)
    file_size = os.path.getsize(path)
    data_start, hdr = read_header(path)
    data_bytes = sum(v["data_offsets"][1] - v["data_offsets"][0] for v in hdr.values())
    assert data_start + data_bytes == file_size, f"Size mismatch in {fname}"

    by_dtype = Counter()
    bytes_by_dtype = Counter()
    for v in hdr.values():
        b = v["data_offsets"][1] - v["data_offsets"][0]
        by_dtype[v["dtype"]] += 1
        bytes_by_dtype[v["dtype"]] += b

    print(f"{fname}: {len(hdr)} tensors, {file_size:,} bytes ({file_size / 1e9:.3f} GB)")
    for dt in sorted(by_dtype):
        print(f"  {dt:7s}: {by_dtype[dt]:4d} tensors, {bytes_by_dtype[dt]:>14,} bytes")
    return data_start, hdr


def active_bytes_per_token(hdr):
    """Inferred minimum weight bytes read per single decode token (batch=1, 48 logical layers)."""
    total = 0
    # Embedding lookup: 1 row of [201024, 2304] BF16 + embed_norm
    total += 2304 * 2 + 2304 * 2
    # Final norm + unembed (all 200,058 active rows of 2304 BF16) + output_query
    total += 2304 * 2 + 200058 * 2304 * 2 + 2304 * 4

    # Logical layers: 0..5 (1x), 6..17 (3x), 18..23 (1x)
    pass_counts = {L: (3 if 6 <= L < 18 else 1) for L in range(24)}
    for L, mult in pass_counts.items():
        prefix = f"model.llm.layers.{L}."
        layer_bytes = 0
        for k, v in hdr.items():
            if not k.startswith(prefix):
                continue
            b = v["data_offsets"][1] - v["data_offsets"][0]
            if ".experts." in k:
                # Only 8 of 128 experts read per token (outer_scale_denominator is scalar)
                if k.endswith("outer_scale_denominator"):
                    layer_bytes += b
                else:
                    layer_bytes += b * 8 // 128
            elif ".controllers." in k:
                # Controllers only run on passes 1 and 2 (2 of the 3 core passes)
                continue
            else:
                layer_bytes += b
        total += layer_bytes * mult
        # AttnRes queries for layer L
        if L > 0:
            q = hdr[f"model.llm.attn_res.attn_queries.{L - 1}"]
            total += (q["data_offsets"][1] - q["data_offsets"][0]) * mult
        mq = hdr[f"model.llm.attn_res.mlp_queries.{L}"]
        total += (mq["data_offsets"][1] - mq["data_offsets"][0]) * mult
        # Loop controllers & cross-pass queries on passes 1 and 2
        if 6 <= L < 18:
            ctrl_b = sum(
                v["data_offsets"][1] - v["data_offsets"][0]
                for k, v in hdr.items()
                if k.startswith(prefix) and ".controllers." in k
            )
            total += ctrl_b * 2
            # 4 cross-pass query rows per core layer (F32, 2304 each), used on passes 1 & 2
            total += 4 * 2304 * 4 * 2
    return total


if __name__ == "__main__":
    _, main_hdr = summarize_file("model-00001-of-00001.safetensors")
    summarize_file("mtp.safetensors")

    llm = {k: v for k, v in main_hdr.items() if k.startswith("model.llm.")}
    vis = {k: v for k, v in main_hdr.items() if k.startswith("model.visual.")}
    llm_bytes = sum(v["data_offsets"][1] - v["data_offsets"][0] for v in llm.values())
    vis_bytes = sum(v["data_offsets"][1] - v["data_offsets"][0] for v in vis.values())
    print(f"\nSplit of model-00001-of-00001.safetensors:")
    print(f"  LLM backbone : {len(llm):4d} tensors, {llm_bytes:>14,} bytes ({llm_bytes / 1e9:.3f} GB)")
    print(f"  Vision tower : {len(vis):4d} tensors, {vis_bytes:>14,} bytes ({vis_bytes / 1e9:.3f} GB)")
    assert len(llm) + len(vis) == len(main_hdr)

    act = active_bytes_per_token(main_hdr)
    print(f"\nInferred weight bytes read per decode token (48 logical layers, 8/128 experts):")
    print(f"  {act:,} bytes ({act / 1e9:.3f} GB, {act / (1024**3):.3f} GiB)")
