# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import MethodType, SimpleNamespace
from typing import get_args
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm.config.model import PROCESSED_LOGPROBS_MODES, LogprobsMode
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.spec_decode import rejection_sampler as rejection_sampler_module
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import (
    _FP32_BYTES,
    MAX_CHUNK_BYTES,
    RejectionSampler,
    _iter_request_chunks,
)


def test_iter_request_chunks_preserves_request_boundaries():
    cu_num_logits = np.array([0, 3, 4, 11, 13], dtype=np.int32)

    assert list(_iter_request_chunks(cu_num_logits, max_chunk_logits=5)) == [
        (0, 2),
        (2, 3),
        (3, 4),
    ]


@pytest.mark.parametrize("num_speculative_tokens", [7, 15])
def test_iter_request_chunks_bounds_qwen38_verify_scratch(num_speculative_tokens: int):
    vocab_size = 151_936
    verify_width = num_speculative_tokens + 1
    cu_num_logits = np.arange(33, dtype=np.int32) * verify_width
    max_chunk_logits = MAX_CHUNK_BYTES // (vocab_size * _FP32_BYTES)

    chunks = list(_iter_request_chunks(cu_num_logits, max_chunk_logits))

    assert len(chunks) > 1
    assert chunks[0][0] == 0
    assert chunks[-1][1] == 32
    assert all(left[1] == right[0] for left, right in zip(chunks, chunks[1:]))
    for start, end in chunks:
        num_chunk_logits = int(cu_num_logits[end] - cu_num_logits[start])
        assert num_chunk_logits <= max_chunk_logits or end == start + 1


@pytest.mark.parametrize("num_speculative_tokens", [7, 15])
def test_dummy_input_batch_expands_full_verify_width(num_speculative_tokens: int):
    num_reqs = 3
    verify_width = num_speculative_tokens + 1
    input_batch = InputBatch.make_dummy(
        num_reqs,
        num_reqs * verify_width,
        InputBuffers(num_reqs, num_reqs * verify_width, torch.device("cpu")),
        num_logits_per_req=verify_width,
    )

    assert input_batch.logits_indices.tolist() == list(range(num_reqs * verify_width))
    assert input_batch.logits_indices.dtype == torch.int32
    assert input_batch.expanded_idx_mapping.tolist() == [
        req_idx for req_idx in range(num_reqs) for _ in range(verify_width)
    ]
    assert input_batch.expanded_local_pos.tolist() == list(range(verify_width)) * 3
    assert input_batch.cu_num_logits_np.tolist() == [
        req_idx * verify_width for req_idx in range(num_reqs + 1)
    ]


@pytest.mark.parametrize("num_speculative_tokens", [7, 15])
def test_profile_run_holds_largest_fp32_request_chunk(
    monkeypatch: pytest.MonkeyPatch, num_speculative_tokens: int
):
    num_reqs = 5
    vocab_size = 17
    verify_width = num_speculative_tokens + 1
    logits = torch.empty(num_reqs * verify_width, vocab_size)
    input_batch = SimpleNamespace(
        cu_num_logits_np=np.arange(num_reqs + 1, dtype=np.int32) * verify_width
    )
    events = []

    class Scratch:
        def copy_(self, source: torch.Tensor) -> "Scratch":
            events.append(("copy", source.shape))
            return self

    def fake_empty_like(source: torch.Tensor, *, dtype: torch.dtype) -> Scratch:
        events.append(("allocate", source.shape, dtype))
        return Scratch()

    def fake_call(self, call_logits, call_input_batch):
        events.append(("call", call_logits is logits, call_input_batch is input_batch))

    monkeypatch.setattr(
        rejection_sampler_module,
        "MAX_CHUNK_BYTES",
        vocab_size * _FP32_BYTES * (2 * verify_width + 1),
    )
    monkeypatch.setattr(rejection_sampler_module.torch, "empty_like", fake_empty_like)
    monkeypatch.setattr(RejectionSampler, "__call__", fake_call)

    RejectionSampler.profile_run(object.__new__(RejectionSampler), logits, input_batch)

    expected_chunk_shape = torch.Size((2 * verify_width, vocab_size))
    assert events == [
        ("allocate", expected_chunk_shape, torch.float32),
        ("copy", expected_chunk_shape),
        ("call", True, True),
    ]


@pytest.mark.parametrize("num_speculative_tokens", [7, 15])
def test_dummy_sampler_run_profiles_full_verify_width(
    monkeypatch: pytest.MonkeyPatch, num_speculative_tokens: int
):
    num_reqs = 3
    hidden_size = 5
    verify_width = num_speculative_tokens + 1
    hidden_states = torch.arange(num_reqs * hidden_size).view(num_reqs, hidden_size)
    logits = torch.empty(num_reqs * verify_width, 11)
    compute_logits = MagicMock(return_value=logits)
    rejection_sampler = SimpleNamespace(profile_run=MagicMock())
    dummy_input_batch = object()
    make_dummy = MagicMock(return_value=dummy_input_batch)
    monkeypatch.setattr(InputBatch, "make_dummy", make_dummy)

    runner = object.__new__(GPUModelRunner)
    runner.decode_query_len = verify_width
    runner.model = SimpleNamespace(compute_logits=compute_logits)
    runner.input_buffers = object()
    runner.sampler = MagicMock()
    runner.rejection_sampler = rejection_sampler

    GPUModelRunner._dummy_sampler_run(runner, hidden_states)

    profiled_hidden_states = compute_logits.call_args.args[0]
    assert profiled_hidden_states.shape == (num_reqs * verify_width, hidden_size)
    assert torch.equal(
        profiled_hidden_states,
        hidden_states.repeat_interleave(verify_width, dim=0),
    )
    make_dummy.assert_called_once_with(
        num_reqs,
        num_reqs * verify_width,
        runner.input_buffers,
        num_logits_per_req=verify_width,
    )
    rejection_sampler.profile_run.assert_called_once_with(logits, dummy_input_batch)
    runner.sampler.assert_not_called()


def test_dummy_sampler_run_preserves_non_spec_sampler(
    monkeypatch: pytest.MonkeyPatch,
):
    num_reqs = 3
    hidden_size = 5
    hidden_states = torch.arange(num_reqs * hidden_size).view(num_reqs, hidden_size)
    logits = torch.empty(num_reqs, 11)
    compute_logits = MagicMock(return_value=logits)
    sampler = MagicMock()
    dummy_input_batch = object()
    make_dummy = MagicMock(return_value=dummy_input_batch)
    monkeypatch.setattr(InputBatch, "make_dummy", make_dummy)

    runner = object.__new__(GPUModelRunner)
    runner.decode_query_len = 1
    runner.model = SimpleNamespace(compute_logits=compute_logits)
    runner.input_buffers = object()
    runner.sampler = sampler
    runner.rejection_sampler = None

    GPUModelRunner._dummy_sampler_run(runner, hidden_states)

    compute_logits.assert_called_once_with(hidden_states)
    make_dummy.assert_called_once_with(
        num_reqs,
        num_reqs,
        runner.input_buffers,
        num_logits_per_req=1,
    )
    sampler.assert_called_once_with(logits, dummy_input_batch)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize("logprobs_mode", get_args(LogprobsMode))
def test_chunked_scores_match_full_batch(logprobs_mode: str):
    device = torch.device("cuda")
    cu_num_logits_np = np.array([0, 3, 4, 8, 10], dtype=np.int32)
    num_logits_per_req = np.diff(cu_num_logits_np)
    idx_mapping_np = np.array([7, 2, 9, 1], dtype=np.int32)
    input_batch = SimpleNamespace(
        num_reqs=4,
        cu_num_logits_np=cu_num_logits_np,
        cu_num_logits=torch.from_numpy(cu_num_logits_np).to(device),
        idx_mapping_np=idx_mapping_np,
        idx_mapping=torch.from_numpy(idx_mapping_np).to(device),
        expanded_idx_mapping=torch.from_numpy(
            np.repeat(idx_mapping_np, num_logits_per_req)
        ).to(device),
        expanded_local_pos=torch.from_numpy(
            np.concatenate(
                [np.arange(count, dtype=np.int32) for count in num_logits_per_req]
            )
        ).to(device),
    )
    rejection_sampler = object.__new__(RejectionSampler)
    rejection_sampler.sampler = SimpleNamespace(logprobs_mode=logprobs_mode)
    rejection_sampler.num_speculative_steps = 3

    def fake_verify(
        self,
        logits,
        _draft_logits,
        _draft_sampled,
        _pos,
        cu_num_logits,
        idx_mapping,
        *_mappings,
    ):
        num_sampled = torch.diff(cu_num_logits).to(torch.int32)
        sampled = (
            idx_mapping.to(torch.int64).unsqueeze(1) + torch.arange(4, device=device)
        ) % logits.shape[1]
        return logits.float() + 1, sampled, num_sampled

    rejection_sampler._verify = MethodType(fake_verify, rejection_sampler)
    logits = torch.arange(170, dtype=torch.float32, device=device).view(10, 17)

    sampled, num_sampled, chunked_logprobs = rejection_sampler._verify_in_chunks(
        logits,
        input_batch,
        draft_logits=None,
        draft_sampled=torch.arange(10, device=device),
        pos=torch.arange(10, device=device),
        max_chunk_logits=5,
        max_num_logprobs=2,
    )
    score_logits = logits + 1 if logprobs_mode in PROCESSED_LOGPROBS_MODES else logits
    full_logprobs = rejection_sampler._get_logprobs_tensors(
        sampled,
        num_sampled,
        score_logits,
        input_batch.cu_num_logits,
        input_batch.cu_num_logits_np,
        max_num_logprobs=2,
    )

    assert sampled[:, 0].tolist() == idx_mapping_np.tolist()
    assert num_sampled.tolist() == num_logits_per_req.tolist()
    assert chunked_logprobs is not None
    assert full_logprobs is not None
    assert torch.equal(
        chunked_logprobs.logprob_token_ids,
        full_logprobs.logprob_token_ids,
    )
    assert torch.equal(chunked_logprobs.logprobs, full_logprobs.logprobs)
    assert torch.equal(
        chunked_logprobs.selected_token_ranks,
        full_logprobs.selected_token_ranks,
    )
    assert (
        chunked_logprobs.cu_num_generated_tokens
        == full_logprobs.cu_num_generated_tokens
    )
