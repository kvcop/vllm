# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer-partition math for the Qwen3.5 (Qwen3.8-27B) PP=2 skew sweep.

A PP=2 x TP=2 boot showed stage 0 at 90-92% GPU busy against stage 1 at
60-62%. The planned response is a `VLLM_PP_LAYER_PARTITION` sweep over
`32,32` (the default), `30,34` and `28,36`. These tests pin what each arm
actually places on each stage — layer index range, hybrid composition and
moved weight bulk — so the arm table used to interpret the GPU window is
checked rather than hand-computed.

Shape source: the pinned Qwen3.8-27B checkpoint config
(`text_config`: 64 layers, `full_attention_interval=4`, hidden 5120,
intermediate 17408, head_dim 256, 24 query heads with `attn_output_gate`,
4 KV heads, GDN 16 key heads x 128 / 48 value heads x 128, conv kernel 4).
"""

import pytest

from vllm.distributed.utils import get_pp_indices

pytestmark = [
    pytest.mark.cpu_test,
    pytest.mark.skip_global_cleanup,
]

NUM_HIDDEN_LAYERS = 64
FULL_ATTENTION_INTERVAL = 4
PP_SIZE = 2

HIDDEN_SIZE = 5120
INTERMEDIATE_SIZE = 17408
VOCAB_SIZE = 248320

# full_attention
HEAD_DIM = 256
NUM_ATTENTION_HEADS = 24
NUM_KV_HEADS = 4
ATTN_OUTPUT_GATE = True

# linear_attention (gated delta net)
LINEAR_NUM_KEY_HEADS = 16
LINEAR_KEY_HEAD_DIM = 128
LINEAR_NUM_VALUE_HEADS = 48
LINEAR_VALUE_HEAD_DIM = 128
LINEAR_CONV_KERNEL_DIM = 4

LAYER_TYPES = tuple(
    "full_attention"
    if (i % FULL_ATTENTION_INTERVAL) == FULL_ATTENTION_INTERVAL - 1
    else "linear_attention"
    for i in range(NUM_HIDDEN_LAYERS)
)

# Sweep arms: env value -> the partition it encodes.
ARMS = {
    "baseline": "32,32",
    "30_34": "30,34",
    "28_36": "28,36",
}


def _mlp_params() -> int:
    gate_up = 2 * INTERMEDIATE_SIZE * HIDDEN_SIZE
    down = INTERMEDIATE_SIZE * HIDDEN_SIZE
    return gate_up + down


def _full_attention_params() -> int:
    # QKVParallelLinear(head_dim, num_heads * (1 + attn_output_gate), num_kv_heads)
    qkv_out = HEAD_DIM * (
        NUM_ATTENTION_HEADS * (1 + int(ATTN_OUTPUT_GATE)) + 2 * NUM_KV_HEADS
    )
    qkv = HIDDEN_SIZE * qkv_out
    o_proj = NUM_ATTENTION_HEADS * HEAD_DIM * HIDDEN_SIZE
    return qkv + o_proj


def _linear_attention_params() -> int:
    key_dim = LINEAR_NUM_KEY_HEADS * LINEAR_KEY_HEAD_DIM
    value_dim = LINEAR_NUM_VALUE_HEADS * LINEAR_VALUE_HEAD_DIM
    in_proj_qkvz = HIDDEN_SIZE * (2 * key_dim + 2 * value_dim)
    in_proj_ba = HIDDEN_SIZE * (2 * LINEAR_NUM_VALUE_HEADS)
    conv1d = (2 * key_dim + value_dim) * LINEAR_CONV_KERNEL_DIM
    out_proj = value_dim * HIDDEN_SIZE
    return in_proj_qkvz + in_proj_ba + conv1d + out_proj


LAYER_PARAMS = {
    "full_attention": _full_attention_params() + _mlp_params(),
    "linear_attention": _linear_attention_params() + _mlp_params(),
}


def stage_profile(pp_rank: int) -> dict:
    """Profile the layers `get_pp_indices` assigns to `pp_rank`.

    Reads whatever `VLLM_PP_LAYER_PARTITION` is currently set to, which is
    how vLLM itself resolves the arm.
    """
    layers = range(*get_pp_indices(NUM_HIDDEN_LAYERS, pp_rank, PP_SIZE))
    types = [LAYER_TYPES[i] for i in layers]
    return {
        "start": layers.start,
        "end": layers.stop,
        "num_layers": len(layers),
        "full_attention": types.count("full_attention"),
        "linear_attention": types.count("linear_attention"),
        "decoder_params": sum(LAYER_PARAMS[t] for t in types),
    }


@pytest.fixture
def arm(monkeypatch: pytest.MonkeyPatch):
    """Apply one arm's env value and return a profile accessor."""

    def _apply(partition: str):
        monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", partition)
        return [stage_profile(rank) for rank in range(PP_SIZE)]

    return _apply


def test_layer_types_match_the_pinned_checkpoint() -> None:
    """The 16 full-attention layers sit at 3, 7, ... 63 — 8 per default stage."""
    full = [i for i, t in enumerate(LAYER_TYPES) if t == "full_attention"]
    assert full == list(range(3, 64, 4))
    assert len(full) == 16
    assert LAYER_TYPES.count("linear_attention") == 48


def test_per_layer_param_counts() -> None:
    """A GDN layer is the *heavier* one, so moving layers barely moves memory."""
    assert LAYER_PARAMS["full_attention"] == 372_244_480
    assert LAYER_PARAMS["linear_attention"] == 383_262_720
    # 11.0M params apart: 2.9% of a layer, 0.09% of a 32-layer stage.
    assert (
        LAYER_PARAMS["linear_attention"] - LAYER_PARAMS["full_attention"] == 11_018_240
    )


def test_env_value_format_is_a_comma_list_summing_to_num_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`VLLM_PP_LAYER_PARTITION` is validated, so a typo fails the boot."""
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "30, 34")  # whitespace is fine
    assert get_pp_indices(NUM_HIDDEN_LAYERS, 0, PP_SIZE) == (0, 30)

    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "30,33")  # 63 != 64
    with pytest.raises(ValueError, match="does not match"):
        get_pp_indices(NUM_HIDDEN_LAYERS, 0, PP_SIZE)

    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "20,22,22")  # len != pp_size
    with pytest.raises(ValueError, match="does not match"):
        get_pp_indices(NUM_HIDDEN_LAYERS, 0, PP_SIZE)

    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "30,34,")
    with pytest.raises(ValueError, match="Invalid partition string"):
        get_pp_indices(NUM_HIDDEN_LAYERS, 0, PP_SIZE)


def test_baseline_arm_is_what_the_default_already_does(
    monkeypatch: pytest.MonkeyPatch, arm
) -> None:
    """`32,32` must reproduce the auto split, so it is a true control."""
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    auto = [stage_profile(rank) for rank in range(PP_SIZE)]
    assert auto == arm(ARMS["baseline"])


def test_arm_baseline_32_32(arm) -> None:
    stage0, stage1 = arm(ARMS["baseline"])
    assert (stage0["start"], stage0["end"]) == (0, 32)
    assert (stage1["start"], stage1["end"]) == (32, 64)
    assert (stage0["full_attention"], stage0["linear_attention"]) == (8, 24)
    assert (stage1["full_attention"], stage1["linear_attention"]) == (8, 24)
    assert stage0["decoder_params"] == stage1["decoder_params"] == 12_176_261_120


def test_arm_30_34(arm) -> None:
    """Moves layers 30 (GDN) and 31 (full attention) onto stage 1."""
    base0, base1 = arm(ARMS["baseline"])
    stage0, stage1 = arm(ARMS["30_34"])
    assert (stage0["start"], stage0["end"]) == (0, 30)
    assert (stage1["start"], stage1["end"]) == (30, 64)
    assert (stage0["full_attention"], stage0["linear_attention"]) == (7, 23)
    assert (stage1["full_attention"], stage1["linear_attention"]) == (9, 25)

    moved = base0["decoder_params"] - stage0["decoder_params"]
    assert moved == stage1["decoder_params"] - base1["decoder_params"]
    assert moved == LAYER_PARAMS["linear_attention"] + LAYER_PARAMS["full_attention"]
    assert moved == 755_507_200


def test_arm_28_36(arm) -> None:
    """Moves layers 28-31: three GDN layers and one full-attention layer."""
    base0, base1 = arm(ARMS["baseline"])
    stage0, stage1 = arm(ARMS["28_36"])
    assert (stage0["start"], stage0["end"]) == (0, 28)
    assert (stage1["start"], stage1["end"]) == (28, 64)
    assert (stage0["full_attention"], stage0["linear_attention"]) == (7, 21)
    assert (stage1["full_attention"], stage1["linear_attention"]) == (9, 27)

    moved = base0["decoder_params"] - stage0["decoder_params"]
    assert (
        moved == 3 * LAYER_PARAMS["linear_attention"] + LAYER_PARAMS["full_attention"]
    )
    assert moved == 1_522_032_640


def test_full_attention_split_moves_only_once_across_the_sweep(arm) -> None:
    """Layer 31 is the only full-attention layer either arm relocates.

    Both `30,34` and `28,36` therefore land on the same 7/9 KV-layer split:
    the arms differ in GDN layers and MLP bulk, not in KV-cache footprint per
    layer type. The extra full-attention layer on stage 1 is what makes stage 1
    the KV-tighter side, and `num_gpu_blocks` is reconciled as the minimum
    across ranks.
    """
    splits = {
        name: tuple(profile["full_attention"] for profile in arm(value))
        for name, value in ARMS.items()
    }
    assert splits == {
        "baseline": (8, 8),
        "30_34": (7, 9),
        "28_36": (7, 9),
    }


def test_stage_totals_stay_balanced_in_bytes(arm) -> None:
    """Weight rebalancing is a side effect here, not the lever.

    Stage 0 also holds `embed_tokens` and stage 1 the untied `lm_head`, both
    `vocab x hidden`, so the non-layer weights already cancel. The most
    aggressive arm shifts ~1.52G params, i.e. ~6.3% of the 24.35G decoder.
    """
    embed = VOCAB_SIZE * HIDDEN_SIZE
    lm_head = VOCAB_SIZE * HIDDEN_SIZE
    assert embed == lm_head == 1_271_398_400

    decoder_total = sum(LAYER_PARAMS[t] for t in LAYER_TYPES)
    assert decoder_total == 24_352_522_240

    base0, _ = arm(ARMS["baseline"])
    stage0, _ = arm(ARMS["28_36"])
    moved = base0["decoder_params"] - stage0["decoder_params"]
    assert moved / decoder_total == pytest.approx(0.0625, abs=1e-3)


def test_only_28_36_keeps_the_hybrid_ratio_on_both_stages(arm) -> None:
    """`get_kv_cache_configs` assumes "the layer ratio of PP stages are similar".

    The checkpoint's global ratio is 1 full-attention layer per 3 GDN layers.
    `28,36` is the only asymmetric arm that preserves it exactly on both
    stages (7:21 and 9:27); `30,34` skews it to 7:23 and 9:25. Whichever arm
    is chosen, stage 1 ends up holding 9 full-attention layers instead of 8,
    so it becomes the KV-tight side and caps the global block count, which
    `get_kv_cache_configs` reconciles as the minimum across workers.
    """
    ratios = {
        name: tuple(
            profile["linear_attention"] / profile["full_attention"]
            for profile in arm(value)
        )
        for name, value in ARMS.items()
    }
    assert ratios["baseline"] == (3.0, 3.0)
    assert ratios["28_36"] == (3.0, 3.0)
    assert ratios["30_34"] != (3.0, 3.0)
