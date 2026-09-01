# Adapted from the MiaAI-Lab/exllamav3 fork-only NVFP4 implementation, not
# upstream turboderp/exllamav3:
# - exllamav3/cache/nvfp4.py
# - exllamav3/modules/attention_fn/triton_paged.py

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

# Nearest-level boundaries for E2M1 magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6.
_E2M1_BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def nvfp4_quantize_reference(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize groups of 16 values to packed E2M1 and E4M3 scales."""
    if x.shape[-1] % 16 != 0:
        raise ValueError("the last dimension must be divisible by 16")

    x = x.half()
    head_size = x.shape[-1]
    grouped = x.reshape(*x.shape[:-1], head_size // 16, 16)
    amax = grouped.abs().amax(dim=-1, keepdim=True)
    scale = (amax / 6.0).clamp(max=448.0).to(torch.float8_e4m3fn)
    decoded_scale = scale.half()
    divisor = torch.where(
        decoded_scale == 0,
        torch.ones_like(decoded_scale),
        decoded_scale,
    )
    normalized = (grouped / divisor).abs().clamp(max=6.0)
    indices = torch.zeros_like(normalized, dtype=torch.uint8)
    for bound in _E2M1_BOUNDS:
        indices += (normalized > bound).to(torch.uint8)
    codes = indices | ((grouped < 0).to(torch.uint8) << 3)
    codes = codes.reshape(*x.shape[:-1], head_size)
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    return packed.contiguous(), scale.squeeze(-1).contiguous()


def nvfp4_dequantize_reference(
    packed: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    """Dequantize packed E2M1 values using one E4M3 scale per 16 values."""
    if packed.dtype != torch.uint8:
        raise ValueError("packed values must have dtype torch.uint8")
    if packed.shape[-1] != scales.shape[-1] * 8:
        raise ValueError("packed and scale dimensions do not describe the same data")

    codes_low = packed & 0xF
    codes_high = packed >> 4
    codes = torch.stack((codes_low, codes_high), dim=-1).flatten(-2)
    indices = codes & 7
    signs = (codes >> 3).half()
    magnitudes = torch.zeros_like(signs)
    for index, value in enumerate((0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)):
        magnitudes += torch.where(
            indices == index + 1,
            torch.full_like(magnitudes, value),
            torch.zeros_like(magnitudes),
        )
    expanded_scales = scales.half().repeat_interleave(16, dim=-1)
    values = magnitudes * torch.where(
        signs > 0,
        torch.full_like(magnitudes, -1.0),
        torch.ones_like(magnitudes),
    )
    return (values * expanded_scales).contiguous()


def nvfp4_cache_bytes_per_token(head_size: int) -> int:
    """Return NVFP4 cache bytes for one token and one K or V head."""
    if head_size <= 0 or head_size % 16 != 0:
        raise ValueError("head_size must be positive and divisible by 16")
    return head_size // 2 + head_size // 16


def nvfp4_cache_shape(
    num_blocks: int,
    num_kv_heads: int,
    block_size: int,
    head_size: int,
) -> tuple[int, int, int, int]:
    """Return the combined K/V cache shape for the fork's HND layout."""
    if num_blocks <= 0 or num_kv_heads <= 0 or block_size <= 0:
        raise ValueError("cache dimensions must be positive")
    full_dim = nvfp4_cache_bytes_per_token(head_size)
    return num_blocks, 2 * num_kv_heads, block_size, full_dim


def nvfp4_allocate(
    num_blocks: int,
    num_kv_heads: int,
    block_size: int,
    head_size: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Allocate a zero-filled combined uint8 NVFP4 cache."""
    return torch.zeros(
        nvfp4_cache_shape(num_blocks, num_kv_heads, block_size, head_size),
        dtype=torch.uint8,
        device=device,
    )


def nvfp4_split_kv(
    cache_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return zero-copy K data, K scales, V data, and V scales views."""
    if cache_tensor.dtype != torch.uint8 or cache_tensor.ndim != 4:
        raise ValueError("cache_tensor must be a rank-4 uint8 tensor")
    if cache_tensor.shape[1] % 2 != 0:
        raise ValueError("the cache head dimension must contain equal K/V halves")

    full_dim = cache_tensor.shape[-1]
    if full_dim % 9 != 0:
        raise ValueError("the cache last dimension is not an NVFP4 packed dimension")
    data_dim = full_dim * 8 // 9
    num_kv_heads = cache_tensor.shape[1] // 2
    k_side = cache_tensor[:, :num_kv_heads]
    v_side = cache_tensor[:, num_kv_heads:]
    k_data = k_side[..., :data_dim]
    v_data = v_side[..., :data_dim]
    k_scales = k_side[..., data_dim:].view(torch.float8_e4m3fn)
    v_scales = v_side[..., data_dim:].view(torch.float8_e4m3fn)
    return k_data, k_scales, v_data, v_scales


@triton.jit
def _nvfp4_encode(x, scales, HEAD_SIZE: tl.constexpr, BLOCK_D: tl.constexpr):
    expanded_scales = tl.reshape(
        tl.broadcast_to(scales[:, None], (BLOCK_D // 16, 16)),
        (BLOCK_D,),
    )
    normalized = (
        x.to(tl.float32)
        / tl.where(expanded_scales == 0, 1.0, expanded_scales).to(tl.float32)
    ).to(tl.float16)
    magnitude = tl.minimum(tl.abs(normalized), 6.0)
    indices = (
        (magnitude > 0.25).to(tl.uint8)
        + (magnitude > 0.75).to(tl.uint8)
        + (magnitude > 1.25).to(tl.uint8)
        + (magnitude > 1.75).to(tl.uint8)
        + (magnitude > 2.5).to(tl.uint8)
        + (magnitude > 3.5).to(tl.uint8)
        + (magnitude > 5.0).to(tl.uint8)
    )
    codes = indices | ((x < 0).to(tl.uint8) << 3)
    low, high = tl.split(tl.reshape(codes, (BLOCK_D // 2, 2)))
    return (low | (high << 4)).to(tl.uint8)


@triton.jit
def _nvfp4_reshape_and_cache_kernel(
    key,
    value,
    slot_mapping,
    cache,
    num_kv_heads: tl.constexpr,
    block_size: tl.constexpr,
    head_size: tl.constexpr,
    full_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_index = tl.program_id(0)
    head_index = tl.program_id(1)
    slot = tl.load(slot_mapping + token_index)
    valid_slot = slot >= 0
    physical_block = slot // block_size
    block_offset = slot - physical_block * block_size

    offsets = tl.arange(0, BLOCK_D)
    source_offsets = (token_index * num_kv_heads + head_index) * head_size + offsets
    source_mask = valid_slot & (offsets < head_size)

    row = (physical_block * (2 * num_kv_heads) + head_index) * block_size + block_offset
    key_row = row * full_dim
    value_row = (row + num_kv_heads * block_size) * full_dim

    key_values = tl.load(key + source_offsets, mask=source_mask, other=0.0)
    key_amax = tl.max(tl.reshape(tl.abs(key_values), (BLOCK_D // 16, 16)), axis=1)
    key_scale_f16 = tl.minimum((key_amax.to(tl.float32) / 6.0).to(tl.float16), 448.0)
    key_scale = key_scale_f16.to(tl.float8e4nv)

    scale_offsets = tl.arange(0, BLOCK_D // 16)
    scale_mask = valid_slot & (scale_offsets < head_size // 16)
    tl.store(
        cache + key_row + head_size // 2 + scale_offsets,
        key_scale.to(tl.uint8, bitcast=True),
        mask=scale_mask,
    )
    packed_offsets = tl.arange(0, BLOCK_D // 2)
    packed_mask = valid_slot & (packed_offsets < head_size // 2)
    tl.store(
        cache + key_row + packed_offsets,
        _nvfp4_encode(key_values, key_scale.to(tl.float16), head_size, BLOCK_D),
        mask=packed_mask,
    )

    value_values = tl.load(value + source_offsets, mask=source_mask, other=0.0)
    value_amax = tl.max(tl.reshape(tl.abs(value_values), (BLOCK_D // 16, 16)), axis=1)
    value_scale_f16 = tl.minimum(
        (value_amax.to(tl.float32) / 6.0).to(tl.float16), 448.0
    )
    value_scale = value_scale_f16.to(tl.float8e4nv)
    tl.store(
        cache + value_row + head_size // 2 + scale_offsets,
        value_scale.to(tl.uint8, bitcast=True),
        mask=scale_mask,
    )
    tl.store(
        cache + value_row + packed_offsets,
        _nvfp4_encode(value_values, value_scale.to(tl.float16), head_size, BLOCK_D),
        mask=packed_mask,
    )


def _validate_cuda_contiguous(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def nvfp4_reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
    cache_tensor: torch.Tensor,
) -> None:
    """Quantize new K/V tensors and store them at flat physical cache slots."""
    _validate_cuda_contiguous("key", key)
    _validate_cuda_contiguous("value", value)
    _validate_cuda_contiguous("slot_mapping", slot_mapping)
    _validate_cuda_contiguous("cache_tensor", cache_tensor)
    if key.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("key and value must have dtype float16 or bfloat16")
    if value.dtype != key.dtype or value.shape != key.shape:
        raise ValueError("key and value must have the same shape and dtype")
    if slot_mapping.dtype not in (torch.int32, torch.int64):
        raise ValueError("slot_mapping must have dtype int32 or int64")
    if cache_tensor.dtype != torch.uint8 or cache_tensor.ndim != 4:
        raise ValueError("cache_tensor must be a rank-4 uint8 tensor")

    num_kv_heads = cache_tensor.shape[1] // 2
    block_size = cache_tensor.shape[2]
    full_dim = cache_tensor.shape[3]
    if full_dim % 9 != 0:
        raise ValueError("cache_tensor has an invalid NVFP4 last dimension")
    head_size = full_dim * 16 // 9
    num_tokens = slot_mapping.numel()
    if key.ndim == 2:
        expected_shape = (num_tokens, num_kv_heads * head_size)
    elif key.ndim == 3:
        expected_shape = (num_tokens, num_kv_heads, head_size)
    else:
        raise ValueError("key and value must have rank 2 or 3")
    if tuple(key.shape) != expected_shape:
        raise ValueError(f"key and value must have shape {expected_shape}")
    if head_size % 32 != 0:
        raise ValueError("head_size must be divisible by 32")

    block_d = triton.next_power_of_2(head_size)
    with torch.cuda.device(cache_tensor.device):
        _nvfp4_reshape_and_cache_kernel[(num_tokens, num_kv_heads)](
            key,
            value,
            slot_mapping,
            cache_tensor,
            num_kv_heads,
            block_size,
            head_size,
            full_dim,
            block_d,
            num_warps=4,
        )


@triton.jit
def _nvfp4_decode(codes, scale):
    indices = codes & 7
    signs = codes >> 3
    magnitude = tl.zeros(indices.shape, dtype=tl.float16)
    magnitude = tl.where(indices == 1, 0.5, magnitude)
    magnitude = tl.where(indices == 2, 1.0, magnitude)
    magnitude = tl.where(indices == 3, 1.5, magnitude)
    magnitude = tl.where(indices == 4, 2.0, magnitude)
    magnitude = tl.where(indices == 5, 3.0, magnitude)
    magnitude = tl.where(indices == 6, 4.0, magnitude)
    magnitude = tl.where(indices == 7, 6.0, magnitude)
    return tl.where(signs > 0, -magnitude, magnitude) * scale


@triton.jit
def _nvfp4_load_row(
    cache,
    row_base,
    head_size: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    byte_offsets = tl.arange(0, BLOCK_D // 2)
    packed = tl.load(
        cache + row_base[:, None] + byte_offsets[None, :],
        mask=byte_offsets[None, :] < head_size // 2,
        other=0,
    )
    low = (packed & 15).to(tl.int32)
    high = (packed >> 4).to(tl.int32)
    scale_offsets = tl.arange(0, BLOCK_D // 16)
    scales = tl.load(
        cache + row_base[:, None] + head_size // 2 + scale_offsets[None, :],
        mask=scale_offsets[None, :] < head_size // 16,
        other=0,
    ).to(tl.float8e4nv, bitcast=True)
    expanded_scales = tl.reshape(
        tl.broadcast_to(scales[:, :, None], (row_base.shape[0], BLOCK_D // 16, 8)),
        (row_base.shape[0], BLOCK_D // 2),
    ).to(tl.float16)
    low_values = _nvfp4_decode(low, expanded_scales)
    high_values = _nvfp4_decode(high, expanded_scales)
    return tl.reshape(
        tl.interleave(low_values, high_values),
        (row_base.shape[0], BLOCK_D),
    )


@triton.jit
def _nvfp4_gather_dequant_kernel(
    cache,
    slot_mapping,
    key_out,
    value_out,
    num_kv_heads: tl.constexpr,
    block_size: tl.constexpr,
    head_size: tl.constexpr,
    full_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_index = tl.program_id(0)
    head_index = tl.program_id(1)
    slot = tl.load(slot_mapping + token_index)
    physical_block = slot // block_size
    block_offset = slot - physical_block * block_size
    row = (physical_block * (2 * num_kv_heads) + head_index) * block_size + block_offset
    key_row = row * full_dim
    value_row = (row + num_kv_heads * block_size) * full_dim

    singleton = tl.arange(0, 1)
    key_values = _nvfp4_load_row(cache, key_row + singleton, head_size, BLOCK_D)
    value_values = _nvfp4_load_row(cache, value_row + singleton, head_size, BLOCK_D)
    offsets = tl.arange(0, BLOCK_D)
    output_offsets = (token_index * num_kv_heads + head_index) * head_size + offsets
    mask = offsets < head_size
    tl.store(
        key_out + output_offsets,
        tl.reshape(key_values, (BLOCK_D,)),
        mask=mask,
    )
    tl.store(
        value_out + output_offsets,
        tl.reshape(value_values, (BLOCK_D,)),
        mask=mask,
    )


def nvfp4_gather_dequant(
    cache_tensor: torch.Tensor,
    slot_mapping: torch.Tensor | int,
    num_tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize physical cache slots as fp16 K/V tensors."""
    _validate_cuda_contiguous("cache_tensor", cache_tensor)
    if isinstance(slot_mapping, int):
        if num_tokens is None or num_tokens < 0:
            raise ValueError("num_tokens must be non-negative for a start slot")
        slot_mapping = torch.arange(
            slot_mapping,
            slot_mapping + num_tokens,
            device=cache_tensor.device,
            dtype=torch.int64,
        )
    elif num_tokens is not None:
        raise ValueError("num_tokens is only valid when slot_mapping is a start slot")
    _validate_cuda_contiguous("slot_mapping", slot_mapping)
    if slot_mapping.dtype not in (torch.int32, torch.int64):
        raise ValueError("slot_mapping must have dtype int32 or int64")

    num_kv_heads = cache_tensor.shape[1] // 2
    block_size = cache_tensor.shape[2]
    full_dim = cache_tensor.shape[3]
    if full_dim % 9 != 0:
        raise ValueError("cache_tensor has an invalid NVFP4 last dimension")
    head_size = full_dim * 16 // 9
    token_count = slot_mapping.numel()
    output_shape = (token_count, num_kv_heads, head_size)
    key = torch.empty(output_shape, dtype=torch.float16, device=cache_tensor.device)
    value = torch.empty_like(key)
    block_d = triton.next_power_of_2(head_size)
    with torch.cuda.device(cache_tensor.device):
        _nvfp4_gather_dequant_kernel[(token_count, num_kv_heads)](
            cache_tensor,
            slot_mapping,
            key,
            value,
            num_kv_heads,
            block_size,
            head_size,
            full_dim,
            block_d,
            num_warps=4,
        )
    return key, value


@triton.jit
def _nvfp4_paged_attention_simple_kernel(
    query,
    cache,
    output,
    block_table,
    seq_lens,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    pages_per_sequence: tl.constexpr,
    block_size: tl.constexpr,
    head_size: tl.constexpr,
    full_dim: tl.constexpr,
    sm_scale: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    sequence_index = tl.program_id(0)
    query_head = tl.program_id(1)
    kv_head = query_head // (num_q_heads // num_kv_heads)
    offsets_d = tl.arange(0, BLOCK_D)
    query_offsets = (sequence_index * num_q_heads + query_head) * head_size + offsets_d
    dimension_mask = offsets_d < head_size
    query_values = tl.load(query + query_offsets, mask=dimension_mask, other=0.0).to(
        tl.float32
    )

    sequence_length = tl.load(seq_lens + sequence_index)
    running_max = tl.full((), -float("inf"), tl.float32)
    running_sum = tl.full((), 0.0, tl.float32)
    accumulator = tl.zeros((BLOCK_D,), tl.float32)

    for token_start in range(0, sequence_length, BLOCK_N):
        token_offsets = token_start + tl.arange(0, BLOCK_N)
        token_mask = token_offsets < sequence_length
        logical_block = token_offsets // block_size
        block_offset = token_offsets - logical_block * block_size
        physical_block = tl.load(
            block_table + sequence_index * pages_per_sequence + logical_block,
            mask=token_mask,
            other=0,
        )
        key_row = (
            (physical_block * (2 * num_kv_heads) + kv_head) * block_size + block_offset
        ) * full_dim
        value_row = (
            (physical_block * (2 * num_kv_heads) + num_kv_heads + kv_head) * block_size
            + block_offset
        ) * full_dim
        key_values = _nvfp4_load_row(cache, key_row, head_size, BLOCK_D)
        scores = (
            tl.sum(key_values.to(tl.float32) * query_values[None, :], axis=1) * sm_scale
        )
        scores = tl.where(token_mask, scores, -float("inf"))

        block_max = tl.max(scores, axis=0)
        new_max = tl.maximum(running_max, block_max)
        alpha = tl.where(
            running_max == -float("inf"),
            0.0,
            tl.exp(running_max - new_max),
        )
        probabilities = tl.where(token_mask, tl.exp(scores - new_max), 0.0)
        value_values = _nvfp4_load_row(cache, value_row, head_size, BLOCK_D)
        accumulator = accumulator * alpha + tl.sum(
            probabilities[:, None] * value_values.to(tl.float32), axis=0
        )
        running_sum = running_sum * alpha + tl.sum(probabilities, axis=0)
        running_max = new_max

    result = accumulator / tl.where(running_sum == 0.0, 1.0, running_sum)
    tl.store(output + query_offsets, result, mask=dimension_mask)


def nvfp4_paged_attention_simple(
    query: torch.Tensor,
    cache_tensor: torch.Tensor,
    output: torch.Tensor | None,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    sm_scale: float | None,
    *,
    block_n: int = 16,
) -> torch.Tensor:
    """Run single-query paged GQA with online NVFP4 dequantization."""
    _validate_cuda_contiguous("query", query)
    _validate_cuda_contiguous("cache_tensor", cache_tensor)
    _validate_cuda_contiguous("block_table", block_table)
    _validate_cuda_contiguous("seq_lens", seq_lens)
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("query must have dtype float16 or bfloat16")
    if block_table.dtype not in (torch.int32, torch.int64):
        raise ValueError("block_table must have dtype int32 or int64")
    if seq_lens.dtype not in (torch.int32, torch.int64):
        raise ValueError("seq_lens must have dtype int32 or int64")
    if query.ndim == 4:
        if query.shape[1] != 1:
            raise ValueError("rank-4 query must have a singleton query dimension")
        query_view = query[:, 0]
    elif query.ndim == 3:
        query_view = query
    else:
        raise ValueError("query must have shape [B, QH, D] or [B, 1, QH, D]")

    batch_size, num_q_heads, head_size = query_view.shape
    if cache_tensor.dtype != torch.uint8 or cache_tensor.ndim != 4:
        raise ValueError("cache_tensor must be a rank-4 uint8 tensor")
    num_kv_heads = cache_tensor.shape[1] // 2
    block_size = cache_tensor.shape[2]
    full_dim = cache_tensor.shape[3]
    if full_dim != nvfp4_cache_bytes_per_token(head_size):
        raise ValueError("query and cache head dimensions do not match")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    if block_table.ndim != 2 or block_table.shape[0] != batch_size:
        raise ValueError("block_table must have shape [B, pages_per_sequence]")
    if seq_lens.shape != (batch_size,):
        raise ValueError("seq_lens must have shape [B]")
    if head_size % 32 != 0:
        raise ValueError("head_size must be divisible by 32")
    if block_n <= 0 or block_n & (block_n - 1):
        raise ValueError("block_n must be a positive power of two")

    if output is None:
        output = torch.empty_like(query)
    else:
        _validate_cuda_contiguous("output", output)
        if output.shape != query.shape or output.dtype != query.dtype:
            raise ValueError("output must have the same shape and dtype as query")
    output_view = output[:, 0] if output.ndim == 4 else output
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_size)

    block_d = triton.next_power_of_2(head_size)
    with torch.cuda.device(query.device):
        _nvfp4_paged_attention_simple_kernel[(batch_size, num_q_heads)](
            query_view,
            cache_tensor,
            output_view,
            block_table,
            seq_lens,
            num_q_heads,
            num_kv_heads,
            block_table.shape[1],
            block_size,
            head_size,
            full_dim,
            float(sm_scale),
            block_n,
            block_d,
            num_warps=4,
        )
    return output
