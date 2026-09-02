# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode-step attention benchmark: NVFP4 KV vs fp8 KV vs fp16 KV.

Laptop-synthetic measurement of one decode-step ``unified_attention`` call
on Qwen3.8-27B full-attention shapes (24 q : 4 kv heads, head_dim 256,
block_size 16), 8 sequences, causal, greedy-shaped single query token.
Numbers are only meaningful relative to each other on the same GPU.

  python tests/v1/attention/bench_nvfp4_kv_decode.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from triton.testing import do_bench  # noqa: E402

from vllm.v1.attention.ops.triton_nvfp4_kv import (  # noqa: E402
    nvfp4_cache_bytes_per_token,
    nvfp4_quantize_reference,
)
from vllm.v1.attention.ops.triton_unified_attention import (
    unified_attention,  # noqa: E402
)
from vllm.v1.kv_cache_interface import KVQuantMode  # noqa: E402

NUM_KV_HEADS = 4
NUM_Q_HEADS = 24
HEAD_SIZE = 256
BLOCK_SIZE = 16
FULL_DIM = nvfp4_cache_bytes_per_token(HEAD_SIZE)
SM_SCALE = 1.0 / math.sqrt(HEAD_SIZE)
NUM_SEQS = 8
DEVICE = torch.device("cuda")


def build(ctx_len: int):
    torch.manual_seed(0)
    blocks_per_seq = (ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = NUM_SEQS * blocks_per_seq + 8
    q = torch.randn(NUM_SEQS, NUM_Q_HEADS, HEAD_SIZE, device=DEVICE).half()
    cu_q = torch.arange(0, NUM_SEQS + 1, device=DEVICE, dtype=torch.int32)
    seqused = torch.full((NUM_SEQS,), ctx_len, device=DEVICE, dtype=torch.int32)
    bt = torch.stack(
        [
            torch.arange(
                s * blocks_per_seq,
                (s + 1) * blocks_per_seq,
                device=DEVICE,
                dtype=torch.int32,
            )
            for s in range(NUM_SEQS)
        ]
    )
    total = NUM_SEQS * ctx_len
    k_all = torch.randn(total, NUM_KV_HEADS, HEAD_SIZE, device=DEVICE).half()
    v_all = torch.randn_like(k_all)
    pk, sk = nvfp4_quantize_reference(k_all)
    pv, sv = nvfp4_quantize_reference(v_all)

    cache16 = torch.zeros(
        num_blocks,
        NUM_KV_HEADS,
        BLOCK_SIZE,
        2 * HEAD_SIZE,
        dtype=torch.float16,
        device=DEVICE,
    )
    t16 = cache16.transpose(1, 2)
    cache4 = torch.zeros(
        num_blocks,
        2 * NUM_KV_HEADS,
        BLOCK_SIZE,
        FULL_DIM,
        dtype=torch.uint8,
        device=DEVICE,
    )
    k_scale = (k_all.float().abs().max() / 448.0).clamp(min=1e-12)
    v_scale = (v_all.float().abs().max() / 448.0).clamp(min=1e-12)
    cache8 = torch.zeros(
        num_blocks,
        NUM_KV_HEADS,
        BLOCK_SIZE,
        2 * HEAD_SIZE,
        dtype=torch.float8_e4m3fn,
        device=DEVICE,
    )
    t8 = cache8.transpose(1, 2)
    k8 = (k_all.float() / k_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    v8 = (v_all.float() / v_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    for s in range(NUM_SEQS):
        for i in range(ctx_len):
            g = s * ctx_len + i
            blk = int(bt[s, i // BLOCK_SIZE].item())
            off = i % BLOCK_SIZE
            t16[blk, off, :, :HEAD_SIZE] = k_all[g]
            t16[blk, off, :, HEAD_SIZE:] = v_all[g]
            t8[blk, off, :, :HEAD_SIZE] = k8[g]
            t8[blk, off, :, HEAD_SIZE:] = v8[g]
            cache4[blk, :NUM_KV_HEADS, off, : HEAD_SIZE // 2] = pk[g]
            cache4[blk, :NUM_KV_HEADS, off, HEAD_SIZE // 2 :] = sk[g].view(torch.uint8)
            cache4[blk, NUM_KV_HEADS:, off, : HEAD_SIZE // 2] = pv[g]
            cache4[blk, NUM_KV_HEADS:, off, HEAD_SIZE // 2 :] = sv[g].view(torch.uint8)
    descales = (k_scale.reshape(1).to(DEVICE), v_scale.reshape(1).to(DEVICE))
    return q, cu_q, seqused, bt, cache16, cache8, cache4, descales


def make_call(q, cache, mode, cu_q, seqused, bt, descales):
    t = cache.transpose(1, 2)
    if mode == KVQuantMode.NVFP4:
        k = t[:, :, :NUM_KV_HEADS, : HEAD_SIZE // 2]
        v = t[:, :, NUM_KV_HEADS:, : HEAD_SIZE // 2]
        kd = vd = None
    else:
        k, v = t[..., :HEAD_SIZE], t[..., HEAD_SIZE:]
        kd, vd = descales if mode == KVQuantMode.FP8_PER_TENSOR else (None, None)
    out = torch.empty(q.shape, dtype=torch.float16, device=DEVICE)

    def call():
        unified_attention(
            q=q,
            k=k,
            v=v,
            out=out,
            cu_seqlens_q=cu_q,
            max_seqlen_q=1,
            seqused_k=seqused,
            max_seqlen_k=int(seqused.max()),
            softmax_scale=SM_SCALE,
            causal=True,
            window_size=(-1, -1),
            block_table=bt,
            softcap=0,
            q_descale=None,
            k_descale=kd,
            v_descale=vd,
            kv_quant_mode=mode,
        )

    return call


if __name__ == "__main__":
    print(
        "laptop-synthetic: one decode-step unified_attention, 8 seqs, "
        "24q:4kv, head 256, block 16, causal"
    )
    print(
        f"{'ctx':>6} {'fp16 us':>9} {'fp8 us':>9} {'nvfp4 us':>9} "
        f"{'nvfp4/fp8':>10} {'nvfp4/fp16':>11}"
    )
    for ctx in (1024, 4096, 16384):
        q, cu_q, seqused, bt, c16, c8, c4, dsc = build(ctx)
        rows = []
        for cache, mode in (
            (c16, KVQuantMode.NONE),
            (c8, KVQuantMode.FP8_PER_TENSOR),
            (c4, KVQuantMode.NVFP4),
        ):
            call = make_call(q, cache, mode, cu_q, seqused, bt, dsc)
            call()
            rows.append(do_bench(call, warmup=25, rep=100) * 1e3)  # ms -> us
        f16, f8, f4 = rows
        print(
            f"{ctx:>6} {f16:>9.1f} {f8:>9.1f} {f4:>9.1f} "
            f"{f4 / f8:>10.2f}x {f4 / f16:>10.2f}x"
        )
