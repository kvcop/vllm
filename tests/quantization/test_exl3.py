# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 (exllamav3 trellis) backend: config, loader placement and numerics.

Fixture-based tests read tensors range-fetched from
Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw into the directory named by
``EXL3_FIXTURE_DIR`` (layout: ``<dotted tensor name with _ instead of .>.pt``).
They skip when the directory is absent. The numerics tests additionally
require CUDA and the upstream ``exllamav3`` package with its compiled
extension; placement/shape tests are CPU-only.
"""

import os

import pytest
import torch

from vllm.model_executor.layers.quantization.exl3 import (
    CODEBOOK_MUL1_MULT,
    Exl3Config,
    Exl3LinearMethod,
)

FIXTURE_DIR = os.environ.get(
    "EXL3_FIXTURE_DIR",
    "/home/user/code/work/rnd/llm/vllm-qwen38-v0271-exl3/.tmp/exl3/hf/fixture",
)

# Qwen3.8-27B text config (config.json of the EXL3 checkpoint)
HIDDEN = 5120
K_DIM = 2048  # linear_num_key_heads(16) * linear_key_head_dim(128)
V_DIM = 6144  # linear_num_value_heads(48) * linear_value_head_dim(128)
QKV = 2 * K_DIM + V_DIM  # 10240
INTER = 17408
QPROJ_OUT = 12288  # attn_output_gate doubles Q

MANIFEST = {
    "model.language_model.layers.0.linear_attn.in_proj_qkv": {
        "quant_format": "exl3",
        "bits_per_weight": 4,
    },
    "model.language_model.layers.0.linear_attn.in_proj_z": {
        "quant_format": "exl3",
        "bits_per_weight": 4,
    },
    "model.language_model.layers.0.mlp.gate_proj": {
        "quant_format": "exl3",
        "bits_per_weight": 4,
    },
    "model.language_model.layers.0.mlp.down_proj": {
        "quant_format": "exl3",
        "bits_per_weight": 4,
    },
    "model.language_model.layers.0.linear_attn.out_proj": {
        "quant_format": "exl3",
        "bits_per_weight": 4,
    },
    "model.language_model.layers.0.mlp.up_proj": {
        "quant_format": "exl3",
        "bits_per_weight": 4,
    },
    "model.language_model.layers.3.self_attn.q_proj": {
        "quant_format": "exl3",
        "bits_per_weight": 4,
    },
    "lm_head": {"quant_format": "exl3", "bits_per_weight": 6},
}


def make_config():
    return Exl3Config({"quant_method": "exl3", "tensor_storage": MANIFEST})


def fx(name: str) -> torch.Tensor:
    path = os.path.join(FIXTURE_DIR, f"{name.replace('.', '_')}.pt")
    if not os.path.exists(path):
        pytest.skip(f"fixture {path} not present")
    return torch.load(path, weights_only=True)


class _FakeColumn:
    def __init__(self, in_f: int, out_f: int, merged_sizes=None):
        self.input_size = in_f
        self.output_size = out_f
        self.tp_rank = 0
        self.tp_size = 1
        self.output_sizes = merged_sizes
        if merged_sizes is None:
            self.output_sizes = None


class _FakeMerged(_FakeColumn):
    pass


class _FakeRow:
    def __init__(self, in_f: int, out_f: int):
        self.input_size = in_f
        self.output_size = out_f
        self.tp_rank = 0
        self.tp_size = 1
        self.input_size_per_partition = in_f


def fake_column(prefix, in_f, out_f, merged_sizes=None):
    return _FakeColumn(in_f, out_f, merged_sizes)


def fake_row(prefix, in_f, out_f):
    return _FakeRow(in_f, out_f)


def init_layer(config: Exl3Config, layer, prefix: str):
    method = config.get_quant_method(layer, prefix)
    assert isinstance(method, Exl3LinearMethod), type(method)
    method.create_weights(
        layer,
        layer.input_size,
        layer.output_size,
        layer.input_size,
        layer.output_size,
        torch.bfloat16,
    )
    return method


# -- config -----------------------------------------------------------------


def test_registry_lists_exl3():
    # Full get_quantization_config() cannot run on CPU-only checkouts:
    # its quark import pulls vllm_flash_attn (pre-existing, see REPORT).
    from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS

    assert "exl3" in QUANTIZATION_METHODS


def test_candidate_parts_resolution():
    cfg = make_config()
    # Plain linear: vLLM prefix -> checkpoint manifest key.
    assert cfg._candidate_parts("model.layers.3.self_attn.q_proj") == (
        "model.language_model.layers.3.self_attn.q_proj",
    )
    # Fused GDN projection resolves to its two checkpoint parts.
    assert cfg._candidate_parts("model.layers.0.linear_attn.in_proj_qkvz") == (
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
        "model.language_model.layers.0.linear_attn.in_proj_z",
    )
    # Fused gate_up.
    assert cfg._candidate_parts("model.layers.0.mlp.gate_up_proj") == (
        "model.language_model.layers.0.mlp.gate_proj",
        "model.language_model.layers.0.mlp.up_proj",
    )
    # Unquantized layers fall through (norms, GDN in_proj_ba, ...).
    assert cfg._candidate_parts("model.layers.0.linear_attn.in_proj_ba") is None
    assert cfg._candidate_parts("model.layers.0.input_layernorm") is None


def test_quantized_lm_head_refused_loudly():
    cfg = make_config()
    layer = type("ParallelLMHead", (), {})()
    with pytest.raises(NotImplementedError, match="lm_head"):
        cfg.get_quant_method(layer, "lm_head")


def test_mismatched_K_fused_parts_rejected():
    manifest = dict(MANIFEST)
    manifest["model.language_model.layers.0.linear_attn.in_proj_z"] = {
        "quant_format": "exl3",
        "bits_per_weight": 5,
    }
    cfg = Exl3Config({"tensor_storage": manifest})
    layer = fake_column(
        "l0", HIDDEN, QKV + V_DIM, merged_sizes=[K_DIM, K_DIM, V_DIM, V_DIM]
    )
    method = cfg.get_quant_method(layer, "model.layers.0.linear_attn.in_proj_qkvz")
    with pytest.raises(ValueError, match="trellis depths"):
        method._manifest_K()


# -- loader placement (CPU, fixture-backed) ---------------------------------


def test_load_plain_column_q_proj():
    cfg = make_config()
    prefix = "model.layers.3.self_attn.q_proj"
    layer = fake_column(prefix, HIDDEN, QPROJ_OUT)
    init_layer(cfg, layer, prefix)

    assert layer.trellis.dtype == torch.int16
    assert layer.trellis.shape == (HIDDEN // 16, QPROJ_OUT // 16, 64)
    assert layer.suh.shape == (HIDDEN, 1) and layer.suh.dtype == torch.half
    assert layer.svh.shape == (QPROJ_OUT,) and layer.svh.dtype == torch.half
    assert layer.mul1.dtype == torch.int32

    base = "model.language_model.layers.3.self_attn.q_proj"
    for suf in ("suh", "svh", "trellis"):
        layer.weight_loader(suf, fx(f"{base}.{suf}"))
    layer.weight_loader("mul1", fx(f"{base}.mul1"))

    assert torch.equal(layer.trellis.data, fx(f"{base}.trellis"))
    assert torch.equal(layer.suh.data[:, 0], fx(f"{base}.suh"))
    assert torch.equal(layer.svh.data, fx(f"{base}.svh"))
    assert int(layer.mul1.item()) & 0xFFFFFFFF == CODEBOOK_MUL1_MULT


def test_load_fused_in_proj_qkvz():
    cfg = make_config()
    prefix = "model.layers.0.linear_attn.in_proj_qkvz"
    layer = fake_column(
        prefix, HIDDEN, QKV + V_DIM, merged_sizes=[K_DIM, K_DIM, V_DIM, V_DIM]
    )
    init_layer(cfg, layer, prefix)

    assert layer.trellis.shape == (
        HIDDEN // 16,
        (QKV + V_DIM) // 16,
        64,
    )
    b = "model.language_model.layers.0.linear_attn"
    # in_proj_qkv spans shards (0, 1, 2); in_proj_z is shard 3.
    for suf in ("suh", "svh", "trellis"):
        layer.weight_loader(suf, fx(f"{b}.in_proj_qkv.{suf}"), (0, 1, 2))
        layer.weight_loader(suf, fx(f"{b}.in_proj_z.{suf}"), 3)
    layer.weight_loader("mul1", fx(f"{b}.in_proj_qkv.mul1"), (0, 1, 2))
    layer.weight_loader("mul1", fx(f"{b}.in_proj_z.mul1"), 3)
    assert layer.suh.shape == (HIDDEN, 4)
    # qkv fills columns 0-2 with one vector, z fills column 3 with its own.
    for c in (0, 1, 2):
        assert torch.equal(layer.suh.data[:, c], fx(f"{b}.in_proj_qkv.suh"))
    assert torch.equal(layer.suh.data[:, 3], fx(f"{b}.in_proj_z.suh"))

    # trellis columns must equal the checkpoint concat of qkv | z.
    expect = torch.cat(
        [
            fx(f"{b}.in_proj_qkv.trellis"),
            fx(f"{b}.in_proj_z.trellis"),
        ],
        dim=1,
    )
    assert torch.equal(layer.trellis.data, expect)
    assert torch.equal(
        layer.svh.data,
        torch.cat([fx(f"{b}.in_proj_qkv.svh"), fx(f"{b}.in_proj_z.svh")]),
    )


def test_bad_mul1_multiplier_refused():
    cfg = make_config()
    prefix = "model.layers.3.self_attn.q_proj"
    layer = fake_column(prefix, HIDDEN, QPROJ_OUT)
    init_layer(cfg, layer, prefix)
    bad = torch.tensor(1234, dtype=torch.int32)
    with pytest.raises(ValueError, match="multiplier"):
        layer.weight_loader("mul1", bad)


def test_row_parallel_in_dim_placement():
    cfg = make_config()
    prefix = "model.layers.0.mlp.down_proj"
    layer = fake_row(prefix, INTER, HIDDEN)
    init_layer(cfg, layer, prefix)
    assert layer.trellis.shape == (INTER // 16, HIDDEN // 16, 64)
    b = "model.language_model.layers.0.mlp.down_proj"
    for suf in ("suh", "svh", "trellis", "mul1"):
        layer.weight_loader(suf, fx(f"{b}.{suf}"))
    assert torch.equal(layer.trellis.data, fx(f"{b}.trellis"))


# -- numerics (GPU + exllamav3) ---------------------------------------------

gpu_available = torch.cuda.is_available()
exllamav3_available = False
if gpu_available:
    try:
        import exllamav3  # noqa: F401

        exllamav3_available = True
    except Exception:
        exllamav3_available = False

needs_gpu = pytest.mark.skipif(
    not (gpu_available and exllamav3_available),
    reason="requires CUDA and the exllamav3 package",
)


@needs_gpu
def test_dequant_matches_exllamav3_reference():
    from exllamav3.modules.quant.exl3 import LinearEXL3

    from vllm.model_executor.layers.quantization.exl3 import dequant_weight

    base = "model.language_model.layers.3.self_attn.q_proj"
    t = fx(f"{base}.trellis").cuda()
    suh = fx(f"{base}.suh").cuda()
    svh = fx(f"{base}.svh").cuda()
    mul1 = fx(f"{base}.mul1")

    ours = dequant_weight(t, suh, svh, K=4, mul1=True).float()
    lin = LinearEXL3(
        None,
        HIDDEN,
        QPROJ_OUT,
        None,
        None,
        None,
        suh=suh,
        svh=svh,
        trellis=t,
        mul1=mul1.cpu(),
        bias=None,
        key="t",
    )
    theirs = lin.get_weight_tensor().float()
    rel = (ours - theirs).abs().max() / theirs.abs().max()
    assert rel < 5e-3, rel


@needs_gpu
def test_apply_decode_and_prefill_paths():
    cfg = make_config()
    prefix = "model.layers.3.self_attn.q_proj"
    layer = fake_column(prefix, HIDDEN, QPROJ_OUT)
    method = init_layer(cfg, layer, prefix)
    base = "model.language_model.layers.3.self_attn.q_proj"
    layer.trellis.data = layer.trellis.data.cuda()
    layer.suh.data = layer.suh.data.cuda()
    layer.svh.data = layer.svh.data.cuda()
    layer.mul1.data = layer.mul1.data.cuda()
    for suf in ("suh", "svh", "trellis", "mul1"):
        layer.weight_loader(suf, fx(f"{base}.{suf}").cuda())
    method.process_weights_after_loading(layer)
    assert layer.exl3_bc is not None

    from vllm.model_executor.layers.quantization.exl3 import dequant_weight

    W = dequant_weight(
        layer.trellis.data, layer.suh.data[:, 0], layer.svh.data, 4, True
    )  # (in, out) fp16

    torch.manual_seed(0)
    x = torch.randn(8, HIDDEN, dtype=torch.bfloat16, device="cuda")
    y_bc = method.apply(layer, x)
    ref = (x.half() @ W).bfloat16()
    # Tolerance is fp16 order-of-operations between the fused BC kernel and
    # the materialized-W GEMM, not quantization error.
    rel = (y_bc.float() - ref.float()).abs().max() / ref.float().abs().max()
    assert rel < 1e-2, rel

    x_big = torch.randn(1024, HIDDEN, dtype=torch.bfloat16, device="cuda")
    y_slab = method.apply(layer, x_big)
    ref_big = (x_big.half() @ W).bfloat16()
    # Slab GEMM runs in the layer's bf16 activation dtype against an fp16
    # reference; bf16 mantissa (8 bits) bounds the worst element near 1%.
    rel2 = (y_slab.float() - ref_big.float()).abs().max() / ref_big.float().abs().max()
    assert rel2 < 1.5e-2, rel2


@needs_gpu
def test_fused_gate_up_slab_matches_per_part_reference():
    """Fused layers with differing per-part suh decode via the slab path."""
    cfg = make_config()
    prefix = "model.layers.0.mlp.gate_up_proj"
    layer = fake_column(prefix, HIDDEN, 2 * INTER, merged_sizes=[INTER, INTER])
    method = init_layer(cfg, layer, prefix)
    b = "model.language_model.layers.0.mlp"
    layer.trellis.data = layer.trellis.data.cuda()
    layer.suh.data = layer.suh.data.cuda()
    layer.svh.data = layer.svh.data.cuda()
    layer.mul1.data = layer.mul1.data.cuda()
    for part, shard in (("gate_proj", 0), ("up_proj", 1)):
        for suf in ("suh", "svh", "trellis"):
            layer.weight_loader(suf, fx(f"{b}.{part}.{suf}").cuda(), shard)
        layer.weight_loader("mul1", fx(f"{b}.{part}.mul1").cuda(), shard)
    method.process_weights_after_loading(layer)
    # Fused layers must not take the single-suh BC path.
    assert layer.exl3_bc is None

    from vllm.model_executor.layers.quantization.exl3 import dequant_weight

    W_gate = dequant_weight(
        layer.trellis.data[:, : INTER // 16].contiguous(),
        layer.suh.data[:, 0].contiguous(),
        layer.svh.data[:INTER].contiguous(),
        4,
        True,
    )
    W_up = dequant_weight(
        layer.trellis.data[:, INTER // 16 :].contiguous(),
        layer.suh.data[:, 1].contiguous(),
        layer.svh.data[INTER:].contiguous(),
        4,
        True,
    )
    W_ref = torch.cat([W_gate, W_up], dim=1)  # (in, out)

    torch.manual_seed(0)
    x = torch.randn(200, HIDDEN, dtype=torch.bfloat16, device="cuda")
    y = method.apply(layer, x)
    ref = (x.half() @ W_ref).bfloat16()
    # bf16 slab GEMM vs fp16 reference: worst element lands near 1%.
    rel = (y.float() - ref.float()).abs().max() / ref.float().abs().max()
    assert rel < 1.5e-2, rel
