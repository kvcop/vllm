# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests for the NVFP4 KV cache in the Triton attention backend.

Covers the three port surfaces against the exl3-derived reference in
``triton_nvfp4_kv``:

1. quantize-on-store byte-identity for the fork's logical cache layout in
   both physical orders (HND and NHD), on real Qwen3.8 full-attention
   shapes (4 KV heads x head_dim 256, block_size 16);
2. single-layer attention through the real ``unified_attention`` kernel
   across its tiling variants (prefill 2D, mixed chunked prefill, decode
   2D, decode 3D segments, sliding window, sinks), compared against fp16
   KV and per-tensor fp8 KV on the same synthetic K/V;
3. ``TritonAttentionBackend`` admission surface (shape, dtype support).

Run standalone (``python tests/v1/attention/test_nvfp4_kv_triton_backend.py``)
or via pytest.  Requires a CUDA device (fp8e4nv conversions need SM89+).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from vllm.v1.attention.backend import AttentionType  # noqa: E402
from vllm.v1.attention.backends import triton_attn as ta  # noqa: E402
from vllm.v1.attention.backends.triton_attn import (  # noqa: E402
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
)
from vllm.v1.attention.ops.triton_nvfp4_kv import (  # noqa: E402
    nvfp4_cache_bytes_per_token,
    nvfp4_dequantize_reference,
    nvfp4_quantize_reference,
)
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (  # noqa: E402
    triton_reshape_and_cache_flash_nvfp4,
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

# Regression guards derived from the synthetic-Gaussian numeric baseline
# (see tests/v1/attention/test_nvfp4_kv_numeric.results.md).
ATTENTION_COSINE_REGRESSION_LIMIT = 0.985
ATTENTION_REL_L2_REGRESSION_LIMIT = 0.18
KERNEL_ISOLATION_REL_L2_LIMIT = 5e-3

DEVICE = torch.device("cuda")


def _rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


def _cos(actual: torch.Tensor, expected: torch.Tensor) -> float:
    a, b = actual.float().flatten(), expected.float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def _make_logical_cache(num_blocks: int, layout: str) -> torch.Tensor:
    if layout == "HND":
        return torch.zeros(
            (num_blocks, 2 * NUM_KV_HEADS, BLOCK_SIZE, FULL_DIM),
            dtype=torch.uint8,
            device=DEVICE,
        )
    if layout == "NHD":
        phys = torch.zeros(
            (num_blocks, BLOCK_SIZE, 2 * NUM_KV_HEADS, FULL_DIM),
            dtype=torch.uint8,
            device=DEVICE,
        )
        return phys.permute(0, 2, 1, 3)
    raise ValueError(layout)


def test_store_bytes_match_reference() -> None:
    """The store kernel must be bit-identical to the torch reference."""
    torch.manual_seed(0)
    num_tokens = 4096
    num_blocks = num_tokens // BLOCK_SIZE + 8
    key = torch.randn(num_tokens, NUM_KV_HEADS, HEAD_SIZE, device=DEVICE).half()
    value = torch.randn_like(key)
    slots = torch.arange(num_tokens, device=DEVICE, dtype=torch.int64)
    refs = {
        "K": nvfp4_quantize_reference(key),
        "V": nvfp4_quantize_reference(value),
    }
    for layout in ("HND", "NHD"):
        cache = _make_logical_cache(num_blocks, layout)
        triton_reshape_and_cache_flash_nvfp4(key, value, cache, slots)
        for name, (packed, scale) in refs.items():
            side = cache[:, :NUM_KV_HEADS] if name == "K" else cache[:, NUM_KV_HEADS:]
            data = side[slots // BLOCK_SIZE, :, slots % BLOCK_SIZE, : HEAD_SIZE // 2]
            scales = side[slots // BLOCK_SIZE, :, slots % BLOCK_SIZE, HEAD_SIZE // 2 :]
            assert torch.equal(data, packed), f"{layout} {name} data bytes differ"
            assert torch.equal(scales, scale.view(torch.uint8)), (
                f"{layout} {name} scale bytes differ"
            )


def test_round_trip_rel_rmse() -> None:
    """Gather-dequant through the stored planes matches the reference RMSE."""
    torch.manual_seed(1)
    num_tokens = 8192
    num_blocks = num_tokens // BLOCK_SIZE + 8
    key = torch.randn(num_tokens, NUM_KV_HEADS, HEAD_SIZE, device=DEVICE).half()
    value = torch.randn_like(key)
    slots = torch.arange(num_tokens, device=DEVICE, dtype=torch.int64)
    cache = _make_logical_cache(num_blocks, "HND")
    triton_reshape_and_cache_flash_nvfp4(key, value, cache, slots)
    side = cache[:, :NUM_KV_HEADS]
    k_data = side[slots // BLOCK_SIZE, :, slots % BLOCK_SIZE, : HEAD_SIZE // 2]
    k_scales = side[slots // BLOCK_SIZE, :, slots % BLOCK_SIZE, HEAD_SIZE // 2 :]
    dequant_k = nvfp4_dequantize_reference(
        k_data.reshape(-1, HEAD_SIZE // 2),
        k_scales.contiguous().view(torch.float8_e4m3fn).reshape(-1, HEAD_SIZE // 16),
    ).reshape(num_tokens, NUM_KV_HEADS, HEAD_SIZE)
    v_side = cache[:, NUM_KV_HEADS:]
    v_data = v_side[slots // BLOCK_SIZE, :, slots % BLOCK_SIZE, : HEAD_SIZE // 2]
    v_scales = v_side[slots // BLOCK_SIZE, :, slots % BLOCK_SIZE, HEAD_SIZE // 2 :]
    dequant_v = nvfp4_dequantize_reference(
        v_data.reshape(-1, HEAD_SIZE // 2),
        v_scales.contiguous().view(torch.float8_e4m3fn).reshape(-1, HEAD_SIZE // 16),
    ).reshape(num_tokens, NUM_KV_HEADS, HEAD_SIZE)
    for name, dequant, src in (("K", dequant_k, key), ("V", dequant_v, value)):
        rel_rmse = (
            (dequant.float() - src.float()).square().mean().sqrt()
            / src.float().square().mean().sqrt()
        ).item()
        assert rel_rmse < 0.12, f"{name} round-trip rel-RMSE {rel_rmse:.4f} above guard"


def _build_case(num_seqs: int, ctx: int, q_len: int, seed: int):
    """Build one attention case: q, metadata, and fp16/nvfp4/fp8 caches."""
    torch.manual_seed(seed)
    total_k = ctx + q_len
    blocks_per_seq = (total_k + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = num_seqs * blocks_per_seq + 8
    q = torch.randn(num_seqs * q_len, NUM_Q_HEADS, HEAD_SIZE, device=DEVICE).half()
    cu_q = torch.arange(0, num_seqs + 1, device=DEVICE, dtype=torch.int32) * q_len
    seqused = torch.full((num_seqs,), total_k, device=DEVICE, dtype=torch.int32)
    block_table = torch.stack(
        [
            torch.arange(
                s * blocks_per_seq,
                (s + 1) * blocks_per_seq,
                device=DEVICE,
                dtype=torch.int32,
            )
            for s in range(num_seqs)
        ]
    )
    total = num_seqs * total_k
    k_all = torch.randn(total, NUM_KV_HEADS, HEAD_SIZE, device=DEVICE).half()
    v_all = torch.randn_like(k_all)

    cache16 = torch.zeros(
        num_blocks,
        NUM_KV_HEADS,
        BLOCK_SIZE,
        2 * HEAD_SIZE,
        dtype=torch.float16,
        device=DEVICE,
    )
    cache16d = torch.zeros_like(cache16)
    cache4 = _make_logical_cache(num_blocks, "HND")
    fp8_dtype = torch.float8_e4m3fn
    cache8 = torch.zeros(
        num_blocks,
        NUM_KV_HEADS,
        BLOCK_SIZE,
        2 * HEAD_SIZE,
        dtype=fp8_dtype,
        device=DEVICE,
    )
    pk, sk = nvfp4_quantize_reference(k_all)
    pv, sv = nvfp4_quantize_reference(v_all)
    kd = nvfp4_dequantize_reference(pk, sk)
    vd = nvfp4_dequantize_reference(pv, sv)
    # Per-tensor fp8 on the same source data (amax/448 over all of K, V).
    k_scale = (k_all.float().abs().max() / 448.0).clamp(min=1e-12)
    v_scale = (v_all.float().abs().max() / 448.0).clamp(min=1e-12)
    k8 = (k_all.float() / k_scale).clamp(-448, 448).to(fp8_dtype)
    v8 = (v_all.float() / v_scale).clamp(-448, 448).to(fp8_dtype)

    t16 = cache16.transpose(1, 2)
    t16d = cache16d.transpose(1, 2)
    t8 = cache8.transpose(1, 2)
    for s in range(num_seqs):
        for i in range(total_k):
            g = s * total_k + i
            blk = int(block_table[s, i // BLOCK_SIZE].item())
            off = i % BLOCK_SIZE
            t16[blk, off, :, :HEAD_SIZE] = k_all[g]
            t16[blk, off, :, HEAD_SIZE:] = v_all[g]
            t16d[blk, off, :, :HEAD_SIZE] = kd[g]
            t16d[blk, off, :, HEAD_SIZE:] = vd[g]
            t8[blk, off, :, :HEAD_SIZE] = k8[g]
            t8[blk, off, :, HEAD_SIZE:] = v8[g]
            cache4[blk, :NUM_KV_HEADS, off, : HEAD_SIZE // 2] = pk[g]
            cache4[blk, :NUM_KV_HEADS, off, HEAD_SIZE // 2 :] = sk[g].view(torch.uint8)
            cache4[blk, NUM_KV_HEADS:, off, : HEAD_SIZE // 2] = pv[g]
            cache4[blk, NUM_KV_HEADS:, off, HEAD_SIZE // 2 :] = sv[g].view(torch.uint8)

    descale_k = k_scale.reshape(1).to(DEVICE)
    descale_v = v_scale.reshape(1).to(DEVICE)
    return (
        q,
        cu_q,
        seqused,
        block_table,
        q_len,
        cache16,
        cache16d,
        cache4,
        cache8,
        (
            descale_k,
            descale_v,
        ),
    )


def _run(
    q,
    cache,
    mode,
    cu_q,
    seqused,
    block_table,
    max_q,
    sinks=None,
    window=(-1, -1),
    use_3d=False,
    descales=None,
):
    t = cache.transpose(1, 2)
    if mode == KVQuantMode.NVFP4:
        k = t[:, :, :NUM_KV_HEADS, : HEAD_SIZE // 2]
        v = t[:, :, NUM_KV_HEADS:, : HEAD_SIZE // 2]
    else:
        k, v = t[..., :HEAD_SIZE], t[..., HEAD_SIZE:]
    out = torch.empty(q.shape, dtype=torch.float16, device=DEVICE)
    k_descale = v_descale = None
    if mode == KVQuantMode.FP8_PER_TENSOR and descales is not None:
        k_descale, v_descale = descales
    kwargs = {}
    if use_3d:
        ntok, segs, dpad = q.shape[0], 16, HEAD_SIZE
        kwargs = dict(
            seq_threshold_3D=1,
            num_par_softmax_segments=segs,
            softmax_segm_output=torch.empty(
                (ntok, NUM_Q_HEADS, segs, dpad), dtype=torch.float32, device=DEVICE
            ),
            softmax_segm_max=torch.empty(
                (ntok, NUM_Q_HEADS, segs), dtype=torch.float32, device=DEVICE
            ),
            softmax_segm_expsum=torch.empty(
                (ntok, NUM_Q_HEADS, segs), dtype=torch.float32, device=DEVICE
            ),
        )
    unified_attention(
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=cu_q,
        max_seqlen_q=max_q,
        seqused_k=seqused,
        max_seqlen_k=int(seqused.max()),
        softmax_scale=SM_SCALE,
        causal=True,
        window_size=window,
        block_table=block_table,
        softcap=0,
        q_descale=None,
        k_descale=k_descale,
        v_descale=v_descale,
        kv_quant_mode=mode,
        sinks=sinks,
        **kwargs,
    )
    return out


_VARIANTS = [
    ("prefill_2d", 1, 0, 512, {}),
    ("mixed_2d", 1, 768, 256, {}),
    ("decode_2d", 8, 1024, 1, {}),
    ("decode_3d", 8, 1024, 1, {"use_3d": True}),
    ("decode_swa_2d", 8, 1024, 1, {"window": (127, 0)}),
    ("decode_sinks_2d", 8, 1024, 1, {"sinks": "randn"}),
]


def test_attention_variants_vs_fp16_and_fp8(variant: str) -> None:
    name, num_seqs, ctx, q_len, extra = next(v for v in _VARIANTS if v[0] == variant)
    if extra.get("sinks") == "randn":
        torch.manual_seed(7)
        extra["sinks"] = torch.randn(NUM_Q_HEADS, device=DEVICE)
    (q, cu_q, seqused, bt, max_q, c16, c16d, c4, c8, descales) = _build_case(
        num_seqs, ctx, q_len, seed=hash(name) % (2**31)
    )
    out16 = _run(q, c16, KVQuantMode.NONE, cu_q, seqused, bt, max_q, **extra)
    out16d = _run(q, c16d, KVQuantMode.NONE, cu_q, seqused, bt, max_q, **extra)
    out4 = _run(q, c4, KVQuantMode.NVFP4, cu_q, seqused, bt, max_q, **extra)
    out8 = _run(
        q,
        c8,
        KVQuantMode.FP8_PER_TENSOR,
        cu_q,
        seqused,
        bt,
        max_q,
        descales=descales,
        **extra,
    )

    iso = _rel_l2(out4, out16d)
    assert iso < KERNEL_ISOLATION_REL_L2_LIMIT, f"{name}: kernel iso {iso:.2e}"

    cos16 = _cos(out4, out16)
    l2_16 = _rel_l2(out4, out16)
    assert cos16 > ATTENTION_COSINE_REGRESSION_LIMIT, f"{name}: cos {cos16:.4f}"
    assert l2_16 < ATTENTION_REL_L2_REGRESSION_LIMIT, f"{name}: rel-L2 {l2_16:.4f}"

    # fp8-vs-fp16 on the same data: the error production already tolerates.
    cos8 = _cos(out8, out16)
    l2_8 = _rel_l2(out8, out16)
    cos48 = _cos(out4, out8)
    l2_48 = _rel_l2(out4, out8)
    print(
        f"\n{name}: nvfp4-vs-fp16 cos={cos16:.4f} relL2={l2_16:.4f} | "
        f"fp8-vs-fp16 cos={cos8:.4f} relL2={l2_8:.4f} | "
        f"nvfp4-vs-fp8 cos={cos48:.4f} relL2={l2_48:.4f} | iso={iso:.2e}",
    )


def test_backend_admission_surface() -> None:
    shape = TritonAttentionBackend.get_kv_cache_shape(
        100, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, cache_dtype_str="nvfp4"
    )
    assert shape == (100, 2 * NUM_KV_HEADS, BLOCK_SIZE, FULL_DIM)
    assert TritonAttentionBackend.supports_kv_cache_dtype("nvfp4")
    impl = TritonAttentionImpl(
        num_heads=NUM_Q_HEADS,
        head_size=HEAD_SIZE,
        scale=SM_SCALE,
        num_kv_heads=NUM_KV_HEADS,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="nvfp4",
        attn_type=AttentionType.DECODER,
    )
    assert impl._kv_quant_mode == KVQuantMode.NVFP4
    try:
        TritonAttentionBackend.get_kv_cache_shape(
            4, BLOCK_SIZE, NUM_KV_HEADS, 250, cache_dtype_str="nvfp4"
        )
        raise RuntimeError("bad head_size accepted")
    except ValueError:
        pass


def test_nvfp4_backend_does_not_declare_quantized_query_input() -> None:
    """E361 review blocker: no pre-quantized Q for the nvfp4 engine path.

    ``vllm/model_executor/layers/attention/attention.py`` creates ``QuantFP8``
    and converts Q to ``torch.float8_e4m3fn`` whenever
    ``impl.supports_quant_query_input`` is set and kv_cache_dtype is fp8* or
    nvfp4. The unified kernel's NVFP4 loader decodes K/V to fp16 and then
    converts them to the query dtype, so an FP8 query would silently
    re-quantize the decoded K/V tiles to E4M3: the arm would measure
    NVFP4->fp16->FP8 K/V plus FP8 Q, not the claimed per-16 NVFP4 decode.
    The backend therefore must not declare query quantization for nvfp4,
    which keeps the Q arriving at ``unified_attention`` in the model dtype
    (fp16/bf16). The fp8 path keeps its original declaration.
    """
    impl4 = TritonAttentionImpl(
        num_heads=NUM_Q_HEADS,
        head_size=HEAD_SIZE,
        scale=SM_SCALE,
        num_kv_heads=NUM_KV_HEADS,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="nvfp4",
        attn_type=AttentionType.DECODER,
    )
    assert impl4._kv_quant_mode == KVQuantMode.NVFP4
    assert impl4.supports_quant_query_input is False

    impl8 = TritonAttentionImpl(
        num_heads=NUM_Q_HEADS,
        head_size=HEAD_SIZE,
        scale=SM_SCALE,
        num_kv_heads=NUM_KV_HEADS,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="fp8",
        attn_type=AttentionType.DECODER,
    )
    assert impl8._kv_quant_mode == KVQuantMode.FP8_PER_TENSOR
    assert impl8.supports_quant_query_input is True


def test_nvfp4_fp8_query_double_quantization_oracle() -> None:
    """Packed-row oracle: an FP8 query would corrupt the decoded K/V tiles.

    A single-magnitude 1.6875 group (scale 1.125 x E2M1 1.5) decodes
    exactly through the reference chain, but 1.6875 is not representable in
    E4M3 and rounds to 1.75. Passing an FP8 query to ``unified_attention``
    on the NVFP4 cache (the pre-fix engine behaviour) therefore matches a
    NONE-mode cache whose K/V were pre-cast through E4M3, and differs from
    the model-dtype query result: concrete proof that the query dtype, not
    the NVFP4 store, caused the drift. Post-fix the engine never produces
    that FP8 query; ``TritonAttentionImpl.forward`` additionally raises on
    one reaching the nvfp4 branch.
    """
    torch.manual_seed(11)
    num_seqs, ctx, q_len = 1, 64, 1
    total_k = ctx + q_len
    blocks_per_seq = (total_k + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = num_seqs * blocks_per_seq + 8
    q = torch.randn(num_seqs * q_len, NUM_Q_HEADS, HEAD_SIZE, device=DEVICE).half()
    k_src = torch.full((total_k, NUM_KV_HEADS, HEAD_SIZE), 1.6875, device=DEVICE).half()
    v_src = torch.randn(total_k, NUM_KV_HEADS, HEAD_SIZE, device=DEVICE).half()

    pk, sk = nvfp4_quantize_reference(k_src)
    pv, sv = nvfp4_quantize_reference(v_src)
    kd = nvfp4_dequantize_reference(pk, sk)
    vd = nvfp4_dequantize_reference(pv, sv)
    assert torch.equal(kd, k_src)
    kd8 = kd.to(torch.float8_e4m3fn).half()
    assert (kd8 == 1.75).all() and not torch.equal(kd8, kd)
    vd8 = vd.to(torch.float8_e4m3fn).half()

    cache4 = _make_logical_cache(num_blocks, "HND")
    cache_bug = torch.zeros(
        num_blocks, NUM_KV_HEADS, BLOCK_SIZE, 2 * HEAD_SIZE,
        dtype=torch.float16, device=DEVICE,
    )
    t_bug = cache_bug.transpose(1, 2)
    block_table = torch.arange(
        num_seqs * blocks_per_seq, device=DEVICE, dtype=torch.int32
    ).reshape(num_seqs, blocks_per_seq)
    for i in range(total_k):
        blk = int(block_table[0, i // BLOCK_SIZE].item())
        off = i % BLOCK_SIZE
        cache4[blk, :NUM_KV_HEADS, off, : HEAD_SIZE // 2] = pk[i]
        cache4[blk, :NUM_KV_HEADS, off, HEAD_SIZE // 2 :] = sk[i].view(torch.uint8)
        cache4[blk, NUM_KV_HEADS:, off, : HEAD_SIZE // 2] = pv[i]
        cache4[blk, NUM_KV_HEADS:, off, HEAD_SIZE // 2 :] = sv[i].view(torch.uint8)
        t_bug[blk, off, :, :HEAD_SIZE] = kd8[i]
        t_bug[blk, off, :, HEAD_SIZE:] = vd8[i]

    cu_q = torch.tensor([0, num_seqs * q_len], device=DEVICE, dtype=torch.int32)
    seqused = torch.full((num_seqs,), total_k, device=DEVICE, dtype=torch.int32)

    out_fp16q = _run(q, cache4, KVQuantMode.NVFP4, cu_q, seqused, block_table, q_len)
    out_fp8q = _run(
        q.to(torch.float8_e4m3fn), cache4, KVQuantMode.NVFP4, cu_q, seqused,
        block_table, q_len,
    )
    out_bug_ref = _run(q, cache_bug, KVQuantMode.NONE, cu_q, seqused, block_table, q_len)

    assert _rel_l2(out_fp8q, out_bug_ref) < 1e-4, "FP8-Q run is not the E4M3-cast cache"
    drift = _rel_l2(out_fp8q, out_fp16q)
    assert drift > 1e-3, f"FP8 query did not corrupt the decoded tiles: {drift:.2e}"


def test_query_is_fp8_membership() -> None:
    """Host-side fp8 query check must not call torch.dtype.is_fp8().

    torch.dtype has no ``is_fp8`` attribute on torch 2.11 or 2.13 — the
    upstream ``.is_fp8()`` calls live inside triton.jit kernels, where they
    resolve on triton dtype constexprs. The a1 guard therefore raised
    AttributeError on the first nvfp4 forward (stand boot 04.09 22:54
    follow-up finding). Membership against the two fp8 dtypes is
    version-proof, including for dtype-like objects without the attribute.
    """
    assert ta._query_is_fp8(torch.float8_e4m3fn) is True
    assert ta._query_is_fp8(torch.float8_e5m2) is True
    assert ta._query_is_fp8(torch.float16) is False
    assert ta._query_is_fp8(torch.bfloat16) is False
    # A dtype-like object whose class has no is_fp8 (torch 2.11/2.13
    # torch.dtype) must be classified, not attribute-errored.
    assert ta._query_is_fp8(type("Plain", (), {})()) is False


def _make_forward_metadata(
    cu_q: torch.Tensor,
    seqused: torch.Tensor,
    block_table: torch.Tensor,
    q_len: int,
    total_q: int,
) -> TritonAttentionMetadata:
    """Minimal DECODER metadata for the 2D decode path of forward()."""
    return TritonAttentionMetadata(
        num_actual_tokens=total_q,
        max_query_len=q_len,
        query_start_loc=cu_q,
        max_seq_len=int(seqused.max()),
        seq_lens=seqused,
        block_table=block_table,
        slot_mapping=torch.zeros(total_q, dtype=torch.int32, device=DEVICE),
        seq_threshold_3D=4096,
        num_par_softmax_segments=1,
        softmax_segm_output=None,
        softmax_segm_max=None,
        softmax_segm_expsum=None,
        causal=True,
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
    )


def _forward_case(query_dtype: torch.dtype):
    """Shared setup: impl + filled nvfp4 cache + metadata for forward()."""
    torch.manual_seed(17)
    num_seqs, ctx, q_len = 2, 128, 1
    (q, cu_q, seqused, bt, max_q, c16, c16d, c4, c8, descales) = _build_case(
        num_seqs, ctx, q_len, seed=17
    )
    impl = TritonAttentionImpl(
        num_heads=NUM_Q_HEADS,
        head_size=HEAD_SIZE,
        scale=SM_SCALE,
        num_kv_heads=NUM_KV_HEADS,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="nvfp4",
        attn_type=AttentionType.DECODER,
    )
    total_q = num_seqs * q_len
    # DECODER forward ignores the new-token projections (they were stored by
    # do_kv_cache_update); pass correctly-shaped dummies.
    k_dummy = torch.zeros(total_q, NUM_KV_HEADS, HEAD_SIZE, dtype=query_dtype, device=DEVICE)
    v_dummy = torch.zeros_like(k_dummy)
    md = _make_forward_metadata(cu_q, seqused, bt, q_len, total_q)
    return impl, q.to(query_dtype), k_dummy, v_dummy, c4, md, total_q


def test_nvfp4_forward_accepts_model_dtype_query(query_dtype_name: str) -> None:
    """Through TritonAttentionImpl.forward: fp16/bf16 queries complete (a2).

    On a1 every model-dtype forward died with AttributeError at the
    ``query.dtype.is_fp8()`` guard; the a2 membership check lets the engine
    path run. bf16 is the stand's model dtype, fp16 the laptop harness one.
    """
    query_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[query_dtype_name]
    impl, q, k, v, cache4, md, total_q = _forward_case(query_dtype)
    output = torch.empty(total_q, NUM_Q_HEADS, HEAD_SIZE, dtype=query_dtype, device=DEVICE)

    impl.forward(
        torch.nn.Module(), q, k, v, cache4, md, output
    )

    assert torch.isfinite(output.float()).all()
    assert output.float().abs().max() > 0


def test_nvfp4_forward_rejects_fp8_query() -> None:
    """Through TritonAttentionImpl.forward: an FP8 query raises ValueError.

    On a1 the same call died with AttributeError (missing is_fp8 on the
    torch dtype) instead of the intended fail-closed ValueError.
    """
    impl, q, k, v, cache4, md, total_q = _forward_case(torch.float8_e4m3fn)
    output = torch.empty(total_q, NUM_Q_HEADS, HEAD_SIZE, dtype=torch.float16, device=DEVICE)

    raised = ""
    try:
        impl.forward(torch.nn.Module(), q, k, v, cache4, md, output)
    except ValueError as error:
        raised = str(error)
    assert "pre-quantized FP8 query" in raised, (
        f"forward must fail closed on an FP8 query, got: {raised!r}"
    )


def test_capture_barrier_swallows_hook_failures() -> None:
    """Call-site capture barrier: counting, never raising, warnings-safe.

    The hook itself never raises by contract, but the final barrier must
    hold even when that contract is broken — including under a
    warnings-as-errors filter, where warnings.warn inside the hook's except
    blocks raises instead of printing (PYTHONWARNINGS=error).
    """
    import warnings

    calls: list[str] = []

    def _boom(layer_name, key, value):
        calls.append(layer_name)
        warnings.warn("capture path failed", stacklevel=2)

    host = torch.zeros(2, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.float16)
    saved_hook = ta._nvfp4kv_capture_kv_snapshot
    saved_hits = ta._NVFP4KV_CAPTURE_BARRIER_HITS
    ta._nvfp4kv_capture_kv_snapshot = _boom
    ta._NVFP4KV_CAPTURE_BARRIER_HITS = 0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ta._capture_kv_snapshot_barrier("layer0", host, host)
            ta._capture_kv_snapshot_barrier("layer1", host, host)
        assert calls == ["layer0", "layer1"]
        assert ta._NVFP4KV_CAPTURE_BARRIER_HITS == 2
    finally:
        ta._nvfp4kv_capture_kv_snapshot = saved_hook
        ta._NVFP4KV_CAPTURE_BARRIER_HITS = saved_hits



if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        sys.exit(0)
    failures = 0
    test_store_bytes_match_reference()
    print("PASS test_store_bytes_match_reference")
    test_round_trip_rel_rmse()
    print("PASS test_round_trip_rel_rmse")
    for name, *_ in _VARIANTS:
        try:
            test_attention_variants_vs_fp16_and_fp8(name)
            print(f"PASS test_attention_variants_vs_fp16_and_fp8[{name}]")
        except AssertionError as e:
            failures += 1
            print(f"FAIL test_attention_variants_vs_fp16_and_fp8[{name}]: {e}")
    test_backend_admission_surface()
    print("PASS test_backend_admission_surface")
    test_nvfp4_backend_does_not_declare_quantized_query_input()
    print("PASS test_nvfp4_backend_does_not_declare_quantized_query_input")
    test_nvfp4_fp8_query_double_quantization_oracle()
    print("PASS test_nvfp4_fp8_query_double_quantization_oracle")
    test_query_is_fp8_membership()
    print("PASS test_query_is_fp8_membership")
    for name in ("fp16", "bf16"):
        test_nvfp4_forward_accepts_model_dtype_query(name)
        print(f"PASS test_nvfp4_forward_accepts_model_dtype_query[{name}]")
    test_nvfp4_forward_rejects_fp8_query()
    print("PASS test_nvfp4_forward_rejects_fp8_query")
    test_capture_barrier_swallows_hook_failures()
    print("PASS test_capture_barrier_swallows_hook_failures")
    print(f"SUMMARY failed={failures}")
    sys.exit(1 if failures else 0)
