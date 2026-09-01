from __future__ import annotations

import importlib.util
import math
import platform
from pathlib import Path

import torch

MODULE_PATH = (
    Path(__file__).resolve().parents[3] / "vllm/v1/attention/ops/triton_nvfp4_kv.py"
)
SPEC = importlib.util.spec_from_file_location("triton_nvfp4_kv", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
nvfp4 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(nvfp4)

DEVICE = torch.device("cuda")
NUM_KV_HEADS = 4
HEAD_SIZE = 256
NUM_Q_HEADS = 24
BLOCK_SIZE = 16

# Regression guards derived from the synthetic-Gaussian numeric baseline.
REFERENCE_REL_RMSE_REGRESSION_LIMIT = 0.12
ATTENTION_COSINE_REGRESSION_LIMIT = 0.985
ATTENTION_REL_L2_REGRESSION_LIMIT = 0.18
KERNEL_REL_L2_REGRESSION_LIMIT = 5e-3


def _relative_rmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    error = (actual.float() - expected.float()).square().mean().sqrt()
    reference_rms = expected.float().square().mean().sqrt()
    return (error / reference_rms).item()


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (
        torch.linalg.vector_norm(actual.float() - expected.float())
        / torch.linalg.vector_norm(expected.float())
    ).item()


def _cosine_similarity(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual_flat = actual.float().flatten()
    expected_flat = expected.float().flatten()
    return torch.nn.functional.cosine_similarity(
        actual_flat, expected_flat, dim=0
    ).item()


def _attention_reference(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    group_size = query.shape[1] // key.shape[1]
    expanded_key = (
        key.repeat_interleave(group_size, dim=1).permute(1, 0, 2).unsqueeze(0)
    )
    expanded_value = (
        value.repeat_interleave(group_size, dim=1).permute(1, 0, 2).unsqueeze(0)
    )
    output = torch.nn.functional.scaled_dot_product_attention(
        query.float().unsqueeze(2),
        expanded_key.float(),
        expanded_value.float(),
        is_causal=False,
    )
    return output.squeeze(2).half()


def _scatter_reference_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    block_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sequence_length, num_kv_heads, head_size = key.shape
    num_blocks = block_table.shape[1]
    cache = nvfp4.nvfp4_allocate(
        num_blocks,
        num_kv_heads,
        BLOCK_SIZE,
        head_size,
        device=key.device,
    )
    key_packed, key_scales = nvfp4.nvfp4_quantize_reference(key)
    value_packed, value_scales = nvfp4.nvfp4_quantize_reference(value)
    k_data, k_sf, v_data, v_sf = nvfp4.nvfp4_split_kv(cache)
    logical_tokens = torch.arange(sequence_length, device=key.device)
    logical_blocks = logical_tokens // BLOCK_SIZE
    offsets = logical_tokens % BLOCK_SIZE
    physical_blocks = block_table[0, logical_blocks].long()
    k_data[physical_blocks, :, offsets] = key_packed
    k_sf[physical_blocks, :, offsets] = key_scales
    v_data[physical_blocks, :, offsets] = value_packed
    v_sf[physical_blocks, :, offsets] = value_scales
    return cache, key_packed, key_scales, value_packed, value_scales


def test_bytes_per_token() -> None:
    bytes_per_token = nvfp4.nvfp4_cache_bytes_per_token(HEAD_SIZE)
    bits_per_element = bytes_per_token * 8 / HEAD_SIZE
    assert bytes_per_token == 144
    assert bits_per_element == 4.5
    print(
        "bytes_per_token "
        f"bytes_per_token_per_head={bytes_per_token} "
        f"bits_per_element={bits_per_element:.4g}"
    )


def test_layout_roundtrip_views() -> None:
    cache = nvfp4.nvfp4_allocate(3, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE, device=DEVICE)
    k_data, k_sf, v_data, v_sf = nvfp4.nvfp4_split_kv(cache)
    data_dim = HEAD_SIZE // 2
    scale_dim = HEAD_SIZE // 16
    expected_data_shape = (3, NUM_KV_HEADS, BLOCK_SIZE, data_dim)
    expected_scale_shape = (3, NUM_KV_HEADS, BLOCK_SIZE, scale_dim)
    assert k_data.shape == v_data.shape == expected_data_shape
    assert k_sf.shape == v_sf.shape == expected_scale_shape
    assert k_data.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    assert k_sf.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    assert v_data.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    assert v_sf.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    assert k_sf.storage_offset() == k_data.storage_offset() + data_dim
    assert v_sf.storage_offset() == v_data.storage_offset() + data_dim
    k_data[0, 0, 0].fill_(0xA5)
    k_sf.view(torch.uint8)[0, 0, 0].fill_(0x3C)
    assert torch.all(cache[0, 0, 0, :data_dim] == 0xA5)
    assert torch.all(cache[0, 0, 0, data_dim:] == 0x3C)
    print(
        "layout_roundtrip_views "
        f"data_end={data_dim} scale_begin={k_sf.storage_offset()} "
        f"scale_bytes={scale_dim} zero_copy=true"
    )


def test_reference_roundtrip_rmse() -> None:
    torch.manual_seed(20260901)
    relative_rmse_values = []
    for token_count in (4096, 8192):
        for tensor_name in ("K", "V"):
            source = torch.randn(
                token_count,
                NUM_KV_HEADS,
                HEAD_SIZE,
                device=DEVICE,
                dtype=torch.float16,
            )
            for distribution in ("normal", "group_outliers"):
                values = source
                if distribution == "group_outliers":
                    grouped = source.reshape(
                        token_count, NUM_KV_HEADS, HEAD_SIZE // 16, 16
                    )
                    group_mask = (
                        torch.rand(
                            token_count,
                            NUM_KV_HEADS,
                            HEAD_SIZE // 16,
                            1,
                            device=DEVICE,
                        )
                        < 0.05
                    )
                    values = torch.where(group_mask, grouped * 20, grouped).reshape_as(
                        source
                    )
                packed, scales = nvfp4.nvfp4_quantize_reference(values)
                restored = nvfp4.nvfp4_dequantize_reference(packed, scales)
                relative_rmse = _relative_rmse(restored, values)
                print(
                    "reference_roundtrip_rmse "
                    f"tokens={token_count} tensor={tensor_name} "
                    f"distribution={distribution} rel_rmse={relative_rmse:.4g}"
                )
                relative_rmse_values.append(relative_rmse)
                assert torch.isfinite(restored).all()
    assert max(relative_rmse_values) < REFERENCE_REL_RMSE_REGRESSION_LIMIT


def test_triton_store_matches_reference() -> None:
    torch.manual_seed(20260902)
    num_tokens = 512
    num_blocks = 40
    key = torch.randn(
        num_tokens,
        NUM_KV_HEADS,
        HEAD_SIZE,
        device=DEVICE,
        dtype=torch.float16,
    )
    value = torch.randn_like(key)
    slot_mapping = torch.randperm(
        num_blocks * BLOCK_SIZE, device=DEVICE, dtype=torch.int64
    )[:num_tokens].contiguous()
    cache = nvfp4.nvfp4_allocate(
        num_blocks, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE, device=DEVICE
    )
    nvfp4.nvfp4_reshape_and_cache(key, value, slot_mapping, cache)

    key_packed, key_scales = nvfp4.nvfp4_quantize_reference(key)
    value_packed, value_scales = nvfp4.nvfp4_quantize_reference(value)
    k_data, k_sf, v_data, v_sf = nvfp4.nvfp4_split_kv(cache)
    blocks = slot_mapping // BLOCK_SIZE
    offsets = slot_mapping % BLOCK_SIZE
    actual_key_packed = k_data[blocks, :, offsets]
    actual_value_packed = v_data[blocks, :, offsets]
    actual_key_scale_bytes = k_sf[blocks, :, offsets].view(torch.uint8)
    actual_value_scale_bytes = v_sf[blocks, :, offsets].view(torch.uint8)

    key_packed_equal = torch.equal(actual_key_packed, key_packed)
    value_packed_equal = torch.equal(actual_value_packed, value_packed)
    key_scales_equal = torch.equal(actual_key_scale_bytes, key_scales.view(torch.uint8))
    value_scales_equal = torch.equal(
        actual_value_scale_bytes, value_scales.view(torch.uint8)
    )
    assert key_packed_equal and value_packed_equal
    assert key_scales_equal and value_scales_equal

    gathered_key, gathered_value = nvfp4.nvfp4_gather_dequant(cache, slot_mapping)
    reference_key = nvfp4.nvfp4_dequantize_reference(key_packed, key_scales)
    reference_value = nvfp4.nvfp4_dequantize_reference(value_packed, value_scales)
    key_max_abs = (gathered_key - reference_key).abs().max().item()
    value_max_abs = (gathered_value - reference_value).abs().max().item()
    assert key_max_abs == 0.0
    assert value_max_abs == 0.0
    print(
        "triton_store_matches_reference "
        "packed_bytes_equal=true scale_bytes_equal=true "
        f"gather_key_max_abs={key_max_abs:.4g} "
        f"gather_value_max_abs={value_max_abs:.4g}"
    )


def test_attention_output_vs_fp16() -> None:
    torch.manual_seed(20260903)
    sequence_length = 1024
    num_blocks = sequence_length // BLOCK_SIZE
    query = torch.randn(1, NUM_Q_HEADS, HEAD_SIZE, device=DEVICE, dtype=torch.float16)
    key = torch.randn(
        sequence_length,
        NUM_KV_HEADS,
        HEAD_SIZE,
        device=DEVICE,
        dtype=torch.float16,
    )
    value = torch.randn_like(key)
    block_table = torch.randperm(num_blocks, device=DEVICE, dtype=torch.int32).reshape(
        1, num_blocks
    )
    seq_lens = torch.tensor([sequence_length], device=DEVICE, dtype=torch.int32)
    scale = 1.0 / math.sqrt(HEAD_SIZE)

    reference_cache, key_packed, key_scales, value_packed, value_scales = (
        _scatter_reference_cache(key, value, block_table)
    )
    nvfp4_output = nvfp4.nvfp4_paged_attention_simple(
        query, reference_cache, None, block_table, seq_lens, scale
    )
    fp16_output = _attention_reference(query, key, value)
    cosine = _cosine_similarity(nvfp4_output, fp16_output)
    relative_l2 = _relative_l2(nvfp4_output, fp16_output)
    print(
        "attention_output_vs_fp16 "
        f"cosine_similarity={cosine:.4g} rel_l2={relative_l2:.4g}"
    )
    triton_cache = nvfp4.nvfp4_allocate(
        num_blocks, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE, device=DEVICE
    )
    logical_tokens = torch.arange(sequence_length, device=DEVICE)
    physical_blocks = block_table[0, logical_tokens // BLOCK_SIZE].long()
    slot_mapping = (
        physical_blocks * BLOCK_SIZE + logical_tokens % BLOCK_SIZE
    ).contiguous()
    nvfp4.nvfp4_reshape_and_cache(key, value, slot_mapping, triton_cache)
    triton_output = nvfp4.nvfp4_paged_attention_simple(
        query, triton_cache, None, block_table, seq_lens, scale
    )
    dequantized_key = nvfp4.nvfp4_dequantize_reference(key_packed, key_scales)
    dequantized_value = nvfp4.nvfp4_dequantize_reference(value_packed, value_scales)
    dequantized_reference = _attention_reference(
        query, dequantized_key, dequantized_value
    )
    kernel_relative_l2 = _relative_l2(triton_output, dequantized_reference)
    print(f"attention_kernel_correctness rel_l2={kernel_relative_l2:.4g}")
    assert cosine > ATTENTION_COSINE_REGRESSION_LIMIT
    assert relative_l2 < ATTENTION_REL_L2_REGRESSION_LIMIT
    assert kernel_relative_l2 < KERNEL_REL_L2_REGRESSION_LIMIT


def test_scale_zero_guard() -> None:
    torch.manual_seed(20260904)
    num_tokens = 32
    key = torch.randn(
        num_tokens,
        NUM_KV_HEADS,
        HEAD_SIZE,
        device=DEVICE,
        dtype=torch.float16,
    )
    value = torch.randn_like(key)
    key[..., :16] = 0
    value[..., 32:48] = 0
    key_packed, key_scales = nvfp4.nvfp4_quantize_reference(key)
    value_packed, value_scales = nvfp4.nvfp4_quantize_reference(value)
    reference_key = nvfp4.nvfp4_dequantize_reference(key_packed, key_scales)
    reference_value = nvfp4.nvfp4_dequantize_reference(value_packed, value_scales)
    assert torch.isfinite(reference_key).all()
    assert torch.isfinite(reference_value).all()
    assert torch.all(key_scales[..., 0].half() == 0)
    assert torch.all(value_scales[..., 2].half() == 0)

    cache = nvfp4.nvfp4_allocate(2, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE, device=DEVICE)
    slot_mapping = torch.arange(num_tokens, device=DEVICE, dtype=torch.int64)
    nvfp4.nvfp4_reshape_and_cache(key, value, slot_mapping, cache)
    gathered_key, gathered_value = nvfp4.nvfp4_gather_dequant(cache, slot_mapping)
    assert torch.isfinite(gathered_key).all()
    assert torch.isfinite(gathered_value).all()
    assert torch.all(gathered_key[..., :16] == 0)
    assert torch.all(gathered_value[..., 32:48] == 0)
    print(
        "scale_zero_guard reference_finite=true triton_finite=true "
        "zero_groups_exact=true"
    )


TESTS = (
    test_bytes_per_token,
    test_layout_roundtrip_views,
    test_reference_roundtrip_rmse,
    test_triton_store_matches_reference,
    test_attention_output_vs_fp16,
    test_scale_zero_guard,
)


def main() -> None:
    import triton

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    print(
        f"environment python={platform.python_version()} torch={torch.__version__} "
        f"triton={triton.__version__} gpu={torch.cuda.get_device_name(0)}"
    )
    passed = 0
    failed = []
    for test in TESTS:
        try:
            test()
            torch.cuda.synchronize()
        except Exception as error:
            failed.append(test.__name__)
            print(f"FAIL {test.__name__}: {type(error).__name__}: {error}")
        else:
            passed += 1
            print(f"PASS {test.__name__}")
    print(f"SUMMARY passed={passed} failed={len(failed)} total={len(TESTS)}")
    if failed:
        raise AssertionError(f"failed tests: {', '.join(failed)}")


if __name__ == "__main__":
    main()
