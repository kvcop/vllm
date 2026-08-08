# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm.v1.worker.gpu.spec_decode.adaptive_verification as adaptive_module
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.worker.gpu.async_utils import StepTimingSample
from vllm.v1.worker.gpu.attn_utils import AttentionCGSupportInfo
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
    AdaptiveVerificationManager,
    _assign_draft_token_budget,
    build_cost_tables_from_curves,
    build_exact_adaptive_layout,
    get_adaptive_speculator_cudagraph_mode,
    get_adaptive_verification_cudagraph_mode,
)
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import (
    DSparkSpeculator,
    _calibrate_confidence_logits,
    _confidence_computation_enabled,
)


def test_confidence_temperature_calibrates_request_major_draft_positions():
    logits = torch.tensor([[0.0, 2.0], [-2.0, 4.0]], dtype=torch.float16)
    temperatures = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

    actual = _calibrate_confidence_logits(logits, temperatures)

    expected = torch.sigmoid(
        torch.tensor([[0.0, 1.0], [-2.0, 2.0]], dtype=torch.float32)
    )
    torch.testing.assert_close(actual, expected)
    assert actual.dtype == torch.float32


def test_record_confidence_restores_request_major_step_layout():
    class PositionCodedConfidenceModel:
        def compute_confidence_logits(self, head_hidden, markov_embed):
            torch.testing.assert_close(
                markov_embed.squeeze(-1),
                torch.tensor([0.0, 1.0, 2.0, 10.0, 11.0, 12.0]),
            )
            return head_hidden.squeeze(-1)

    speculator = DSparkSpeculator.__new__(DSparkSpeculator)
    speculator.num_speculative_steps = 3
    speculator.model = PositionCodedConfidenceModel()
    speculator._confidence_temperatures = torch.tensor([[1.0, 2.0, 4.0]])
    speculator.draft_token_confidence_probs = torch.empty((2, 3))
    speculator.draft_token_confidence_logits = torch.empty((2, 3), dtype=torch.float32)
    head_hidden = torch.tensor(
        [[0.0], [2.0], [4.0], [10.0], [20.0], [40.0]], dtype=torch.float16
    )
    markov_by_step = [
        torch.tensor([[float(step)], [float(10 + step)]]) for step in range(3)
    ]

    speculator._record_confidence_probs(2, head_hidden, markov_by_step)

    expected = torch.sigmoid(torch.tensor([[0.0, 1.0, 1.0], [10.0, 10.0, 10.0]]))
    torch.testing.assert_close(speculator.draft_token_confidence_probs, expected)
    torch.testing.assert_close(
        speculator.draft_token_confidence_logits,
        head_hidden.float().view(2, 3),
    )
    assert speculator.draft_token_confidence_logits.dtype == torch.float32


def test_capture_only_enables_confidence_computation():
    assert _confidence_computation_enabled(False, "/private/capture")
    assert _confidence_computation_enabled(True, None)
    assert not _confidence_computation_enabled(False, None)


def test_equal_survival_budget_is_stable_and_prefers_shallow_steps():
    confidences = torch.ones((3, 3), dtype=torch.float32)
    idx_mapping = torch.arange(3)
    expected = torch.tensor([2, 1, 1], dtype=torch.int32)

    for _ in range(3):
        capacities = torch.tensor([3, 3, 3], dtype=torch.int32)
        _assign_draft_token_budget(
            confidences,
            idx_mapping,
            capacities,
            draft_budget=4,
            num_steps=3,
        )
        torch.testing.assert_close(capacities, expected)


def test_uneven_logical_lengths_preserve_bounds_and_total_budget():
    confidences = torch.ones((3, 8), dtype=torch.float32)
    capacities = torch.tensor([0, 1, 8], dtype=torch.int32)

    _assign_draft_token_budget(
        confidences,
        torch.arange(3),
        capacities,
        draft_budget=5,
        num_steps=8,
    )

    # The zero-length request remains empty, the one-token request cannot grow,
    # and the remaining budget is a contiguous prefix of the third request.
    torch.testing.assert_close(capacities, torch.tensor([0, 1, 4], dtype=torch.int32))
    assert int(capacities.sum()) == 5


@pytest.mark.parametrize(
    "support, expected",
    [
        (AttentionCGSupport.ALWAYS, CUDAGraphMode.FULL_AND_PIECEWISE),
        (AttentionCGSupport.UNIFORM_BATCH, CUDAGraphMode.PIECEWISE),
        (AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE, CUDAGraphMode.PIECEWISE),
        (AttentionCGSupport.NEVER, CUDAGraphMode.PIECEWISE),
    ],
)
def test_adaptive_varlen_full_requires_always_attention_support(support, expected):
    assert (
        get_adaptive_verification_cudagraph_mode(
            CUDAGraphMode.FULL, support, enabled=True
        )
        == expected
    )


def test_adaptive_policy_uses_most_restrictive_attention_group():
    support = AttentionCGSupportInfo().narrow(
        AttentionCGSupport.ALWAYS, "TritonAttentionBackend"
    )
    support = support.narrow(AttentionCGSupport.UNIFORM_BATCH, "GDNAttentionBackend")

    assert support.min_cg_attn_backend == "GDNAttentionBackend"
    assert (
        get_adaptive_verification_cudagraph_mode(
            CUDAGraphMode.FULL,
            support.min_cg_support,
            enabled=True,
        )
        == CUDAGraphMode.PIECEWISE
    )


def test_adaptive_off_preserves_configured_cudagraph_mode():
    assert (
        get_adaptive_verification_cudagraph_mode(
            CUDAGraphMode.FULL_DECODE_ONLY,
            AttentionCGSupport.UNIFORM_BATCH,
            enabled=False,
        )
        == CUDAGraphMode.FULL_DECODE_ONLY
    )


@pytest.mark.parametrize(
    "configured_mode, enforce_eager",
    [
        (None, False),
        (CUDAGraphMode.NONE, False),
        (CUDAGraphMode.FULL_AND_PIECEWISE, True),
    ],
)
def test_adaptive_rejects_modes_without_startup_cost_profiling(
    configured_mode, enforce_eager
):
    with pytest.raises(ValueError, match="startup cost profiling"):
        get_adaptive_verification_cudagraph_mode(
            configured_mode,
            AttentionCGSupport.ALWAYS,
            enabled=True,
            enforce_eager=enforce_eager,
        )


def test_target_piecewise_fallback_keeps_speculator_full_eligible():
    assert (
        get_adaptive_speculator_cudagraph_mode(CUDAGraphMode.PIECEWISE, enabled=True)
        == CUDAGraphMode.FULL_AND_PIECEWISE
    )
    assert (
        get_adaptive_speculator_cudagraph_mode(CUDAGraphMode.PIECEWISE, enabled=False)
        == CUDAGraphMode.PIECEWISE
    )


@pytest.mark.parametrize("bad_cost", [float("nan"), float("inf"), -float("inf")])
def test_cost_tables_reject_non_finite_profile_samples(bad_cost):
    with pytest.raises(ValueError, match="finite"):
        build_cost_tables_from_curves(
            [(1, 1.0)],
            [(1, bad_cost)],
            max_num_reqs=2,
            max_batch_tokens=2,
        )


def make_manager(
    confidences: np.ndarray, verify_cost_ms: np.ndarray
) -> AdaptiveVerificationManager:
    num_reqs, num_steps = confidences.shape
    manager = AdaptiveVerificationManager.__new__(AdaptiveVerificationManager)
    manager.num_speculative_steps = num_steps
    manager._stale_confidences = [SimpleNamespace(np=confidences)]
    manager._stale_idx = 0
    manager.req_states = SimpleNamespace(
        req_id_to_index={"low": 0, "high": 1},
        num_computed_tokens_np=np.ones(num_reqs, dtype=np.int32),
        prefill_len=SimpleNamespace(np=np.ones(num_reqs, dtype=np.int32)),
    )
    manager.cost_tables = (np.zeros(num_reqs + 1), verify_cost_ms)
    manager._max_total_logits = 1 << 30
    manager.num_bonus_tokens = 1
    return manager


def test_budget_stops_where_marginal_drafts_stop_paying_for_themselves():
    # Verification is cheap up to two extra tokens, then jumps 100x; only the
    # highest-confidence draft is worth the cheap slot.
    manager = make_manager(
        np.array([[0.1, 0.1], [0.9, 0.9]], dtype=np.float32),
        np.array([1.0, 1.0, 1.0, 1.0, 100.0, 100.0, 100.0]),
    )

    manager.get_num_tokens(
        {"low": 3, "high": 3},
        {"low": [1, 2], "high": [3, 4]},
    )
    valid_drafts, num_non_draft_tokens, draft_budget = manager._batch_budget

    assert draft_budget == 1
    assert valid_drafts == {"low": 2, "high": 2}
    assert num_non_draft_tokens == {"low": 1, "high": 1}


def test_profiled_batches_seed_cost_curves_via_consumer():
    manager = AdaptiveVerificationManager.__new__(AdaptiveVerificationManager)
    manager.req_states = SimpleNamespace(max_num_batched_tokens=4096, max_num_reqs=64)
    manager.num_speculative_steps = 7
    manager.num_bonus_tokens = 1
    curves: dict[str, list[tuple[int, float]]] = {}
    manager.set_cost_curves = lambda draft, verify: curves.update(
        draft=draft, verify=verify
    )

    timings = [
        StepTimingSample(
            forward_ms=float(batch["num_tokens"]),
            drafter_ms=1.0,
            num_target_tokens=batch["num_tokens"],
            num_reqs=batch["num_tokens"] // 8,
            # Only the captured sizes replay a graph; the tail sizes run eager.
            full_cudagraph=batch["num_tokens"] <= 1024,
        )
        for batch in manager.batches_to_profile([8, 1024])
    ]
    manager.set_initial_cost_curves(timings)

    # Tail beyond the last capture size: 1.5x then doubling to the max.
    assert curves["verify"] == [
        (8, 8.0),
        (1024, 1024.0),
        (1536, 1536.0),
        (2048, 2048.0),
        (4096, 4096.0),
    ]
    # Eager batches must not contribute to the draft curve: keyed by request
    # count they would land inside the captured range and, once made monotonic,
    # smear that eager cost across every larger request count.
    assert curves["draft"] == [(1, 1.0), (128, 1.0)]


def test_current_gpu_allocation_returns_exact_cpu_layout(monkeypatch):
    # t-2 confidence sizes the global budget, while current GPU confidence
    # decides which request receives it. The returned CPU capacities become
    # the source of truth for all attention and sampler layout fields.
    manager = make_manager(
        np.array([[0.1, 0.1], [0.9, 0.9]], dtype=np.float32),
        np.array([1.0, 1.0, 1.0, 1.0, 100.0, 100.0, 100.0]),
    )
    manager._confidence_probs = torch.tensor(
        [[0.99, 0.99], [0.01, 0.01]], dtype=torch.float32
    )

    class CapacityBuffer:
        def __init__(self):
            self.gpu = torch.zeros(2, dtype=torch.int32)
            self.np = np.zeros(2, dtype=np.int32)

        def copy_to_cpu(self, n):
            self.np[:n] = self.gpu[:n].numpy()

    class CopyEvent:
        def record(self):
            pass

        def synchronize(self):
            pass

    manager._batch_draft_capacity = CapacityBuffer()
    manager._capacity_copy_event = CopyEvent()
    monkeypatch.setattr(
        adaptive_module,
        "_assign_draft_token_budget_compiled",
        _assign_draft_token_budget,
    )
    monkeypatch.setattr(
        adaptive_module,
        "async_copy_to_gpu",
        lambda values, out: out.copy_(torch.from_numpy(values)),
    )

    num_tokens = manager.get_num_tokens(
        {"low": 3, "high": 3},
        {"low": [1, 2], "high": [3, 4]},
    )
    capacities = manager.allocate_drafts(["low", "high"], torch.arange(2))
    exact_scheduled, exact_cu_num_logits = build_exact_adaptive_layout(
        np.array([3, 3], dtype=np.int32),
        np.array([2, 2], dtype=np.int32),
        capacities,
        num_bonus_tokens=1,
    )

    assert num_tokens == 3
    np.testing.assert_array_equal(capacities, [1, 0])
    np.testing.assert_array_equal(exact_scheduled, [2, 1])
    np.testing.assert_array_equal(exact_cu_num_logits, [0, 2, 3])
    assert int(exact_scheduled.sum()) == num_tokens


def test_empty_cost_curves_fail_loudly(monkeypatch):
    manager = AdaptiveVerificationManager.__new__(AdaptiveVerificationManager)
    manager.req_states = SimpleNamespace(max_num_reqs=2, max_num_batched_tokens=8)
    manager._cudagraph_limit = 0
    monkeypatch.setattr(
        adaptive_module,
        "get_tp_group",
        lambda: SimpleNamespace(broadcast_object=lambda value, src: value),
    )

    with pytest.raises(ValueError, match="non-empty"):
        manager.set_cost_curves([], [(1, 1.0)])


def test_budget_caps_at_one_rejection_sampler_chunk():
    # The chunked verification path cannot address the compacted logits
    # layout, so the budget must keep total logits within a single chunk.
    manager = make_manager(
        np.array([[0.9, 0.9], [0.9, 0.9]], dtype=np.float32),
        np.ones(7),
    )
    manager._max_total_logits = 3  # 2 bonus logits + at most 1 draft
    manager.get_num_tokens(
        {"low": 3, "high": 3},
        {"low": [1, 2], "high": [3, 4]},
    )
    _, _, draft_budget = manager._batch_budget
    assert draft_budget <= 1
