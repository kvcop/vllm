# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.config import ModelConfig, ParallelConfig, SpeculativeConfig
from vllm.transformers_utils.configs.speculators import SpeculatorsConfig


def _write_target_config(path):
    path.mkdir(parents=True, exist_ok=True)
    path.joinpath("config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "vocab_size": 256,
                "max_position_embeddings": 1024,
            }
        )
    )


def _write_qwen_dspark_config(
    path, *, block_size: int = 8, sample_from_anchor: bool = True
):
    path.mkdir(parents=True, exist_ok=True)
    path.joinpath("config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3DSparkModel"],
                "model_type": "qwen3",
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "vocab_size": 256,
                "max_position_embeddings": 1024,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000.0,
                "draft_vocab_size": 256,
                "target_hidden_size": 128,
                "mask_token_id": 1,
                "markov_rank": 16,
                "block_size": block_size,
                "enable_confidence_head": True,
                "confidence_head_with_markov": True,
                "sample_from_anchor": sample_from_anchor,
                "eagle_aux_hidden_state_layer_ids": [1],
                "target_layer_ids": [0],
            }
        )
    )


def _make_qwen_dspark_config(
    tmp_path,
    *,
    sample_from_anchor: bool = True,
    num_speculative_tokens: int = 8,
    dspark_confidence_temperatures: list[float] | None = None,
    **speculative_kwargs,
):
    target_path = tmp_path / "target"
    draft_path = tmp_path / "draft"
    _write_target_config(target_path)
    _write_qwen_dspark_config(draft_path, sample_from_anchor=sample_from_anchor)
    target_config = ModelConfig(
        model=str(target_path), tokenizer_mode="skip", max_model_len=1024
    )
    return SpeculativeConfig(
        model=str(draft_path),
        method="dspark",
        num_speculative_tokens=num_speculative_tokens,
        target_model_config=target_config,
        target_parallel_config=ParallelConfig(),
        dspark_confidence_temperatures=dspark_confidence_temperatures,
        **speculative_kwargs,
    )


@pytest.mark.cpu_test
def test_qwen_dspark_accepts_its_speculators_block_size(tmp_path):
    config = _make_qwen_dspark_config(tmp_path)

    assert config.parallel_drafting
    assert config.draft_model_config.hf_config.block_size == 8
    assert config.draft_model_config.hf_config.sample_from_anchor is True


@pytest.mark.cpu_test
@pytest.mark.parametrize("invalid_num_speculative_tokens", [7, 9])
def test_qwen_dspark_rejects_other_speculative_lengths(
    tmp_path, invalid_num_speculative_tokens
):
    with pytest.raises(ValueError, match="exactly 8"):
        _make_qwen_dspark_config(
            tmp_path, num_speculative_tokens=invalid_num_speculative_tokens
        )


@pytest.mark.cpu_test
def test_legacy_qwen_dspark_excludes_bonus_anchor_from_draft_length(tmp_path):
    config = _make_qwen_dspark_config(
        tmp_path / "valid", sample_from_anchor=False, num_speculative_tokens=7
    )

    assert config.draft_model_config.hf_config.block_size == 8
    assert config.draft_model_config.hf_config.sample_from_anchor is False

    with pytest.raises(ValueError, match="exactly 7"):
        _make_qwen_dspark_config(
            tmp_path / "invalid",
            sample_from_anchor=False,
            num_speculative_tokens=8,
        )


@pytest.mark.cpu_test
def test_qwen_dspark_confidence_temperatures_validate_and_do_not_change_graph_hash(
    tmp_path,
):
    temperatures = [1.0] * 8
    calibrated = _make_qwen_dspark_config(
        tmp_path / "calibrated", dspark_confidence_temperatures=temperatures
    )
    # Reuse the exact same model paths: model provenance is part of the graph
    # hash, while post-hidden confidence calibration deliberately is not.
    raw = _make_qwen_dspark_config(tmp_path / "calibrated")

    assert calibrated.dspark_confidence_temperatures == temperatures
    assert calibrated.compute_hash() == raw.compute_hash()

    with pytest.raises(ValueError, match="must have length 8"):
        _make_qwen_dspark_config(
            tmp_path / "short", dspark_confidence_temperatures=[1.0] * 7
        )
    for index, invalid in enumerate((0.0, float("inf"), float("nan"))):
        with pytest.raises(ValueError, match="finite and positive"):
            _make_qwen_dspark_config(
                tmp_path / f"invalid-{index}",
                dspark_confidence_temperatures=[1.0] * 7 + [invalid],
            )


@pytest.mark.cpu_test
def test_confidence_temperatures_are_dspark_only():
    with pytest.raises(ValueError, match="only supported by DSpark"):
        SpeculativeConfig(
            method="ngram",
            num_speculative_tokens=2,
            dspark_confidence_temperatures=[1.0, 1.0],
        )


@pytest.mark.cpu_test
def test_dspark_confidence_capture_accepts_bounded_probabilistic_identity_mode(
    tmp_path,
):
    capture_path = tmp_path / "capture"
    config = _make_qwen_dspark_config(
        tmp_path / "models",
        draft_sample_method="probabilistic",
        dspark_confidence_capture_path=str(capture_path),
        dspark_confidence_capture_max_rows=100,
        dspark_confidence_capture_shard_rows=11,
        dspark_confidence_temperatures=[1.0] * 8,
    )

    assert config.dspark_confidence_capture_path == str(capture_path)
    assert config.dspark_confidence_capture_max_rows == 100
    assert config.dspark_confidence_capture_shard_rows == 11


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"dspark_confidence_capture_path": "capture"},
            "must be set together",
        ),
        (
            {"dspark_confidence_capture_max_rows": 10},
            "must be set together",
        ),
        (
            {
                "dspark_confidence_capture_path": " ",
                "dspark_confidence_capture_max_rows": 10,
            },
            "must not be empty",
        ),
        (
            {
                "dspark_confidence_capture_path": "capture",
                "dspark_confidence_capture_max_rows": 10,
                "enable_adaptive_verification": True,
            },
            "enable_adaptive_verification=false",
        ),
        (
            {
                "dspark_confidence_capture_path": "capture",
                "dspark_confidence_capture_max_rows": 10,
            },
            "draft_sample_method='probabilistic'",
        ),
        (
            {
                "dspark_confidence_capture_path": "capture",
                "dspark_confidence_capture_max_rows": 10,
                "draft_sample_method": "probabilistic",
                "rejection_sample_method": "block",
            },
            "probabilistic rejection sampling",
        ),
        (
            {
                "dspark_confidence_capture_path": "capture",
                "dspark_confidence_capture_max_rows": 10,
                "draft_sample_method": "probabilistic",
                "dspark_confidence_temperatures": [1.0] * 7 + [2.0],
            },
            "identity confidence temperatures",
        ),
    ],
)
def test_dspark_confidence_capture_rejects_unsafe_modes(tmp_path, kwargs, match):
    kwargs = dict(kwargs)
    temperatures = kwargs.pop("dspark_confidence_temperatures", None)
    with pytest.raises(ValueError, match=match):
        _make_qwen_dspark_config(
            tmp_path,
            dspark_confidence_temperatures=temperatures,
            **kwargs,
        )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dspark_confidence_capture_max_rows", 0),
        ("dspark_confidence_capture_shard_rows", 0),
    ],
)
def test_dspark_confidence_capture_rejects_nonpositive_bounds(tmp_path, field, value):
    kwargs = {
        "draft_sample_method": "probabilistic",
        "dspark_confidence_capture_path": "capture",
        "dspark_confidence_capture_max_rows": 10,
        field: value,
    }
    with pytest.raises(ValueError, match="greater than 0"):
        _make_qwen_dspark_config(tmp_path, **kwargs)


@pytest.mark.cpu_test
def test_confidence_capture_is_dspark_only():
    with pytest.raises(ValueError, match="only supported by DSpark"):
        SpeculativeConfig(
            method="ngram",
            num_speculative_tokens=2,
            draft_sample_method="probabilistic",
            dspark_confidence_capture_path="capture",
            dspark_confidence_capture_max_rows=10,
        )


@pytest.mark.cpu_test
def test_speculators_qwen_dspark_preserves_explicit_anchor_sampling():
    config = {
        "speculators_model_type": "dspark",
        "transformer_layer_config": {"model_type": "qwen3"},
        "draft_vocab_size": 256,
        "target_hidden_size": 128,
        "mask_token_id": 1,
        "markov_rank": 16,
        "block_size": 8,
        "enable_confidence_head": True,
        "confidence_head_with_markov": True,
        "aux_hidden_state_layer_ids": [1],
        "sample_from_anchor": True,
    }

    converted = SpeculatorsConfig.extract_transformers_pre_trained_config(config)

    assert converted["architectures"] == ["Qwen3DSparkModel"]
    assert converted["block_size"] == 8
    assert converted["sample_from_anchor"] is True
