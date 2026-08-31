# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.v1.worker.gpu import pp_utils
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.pp_utils import PPHandler, _pad_sampled_tokens

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("width", [1, 2])
def test_pp_broadcast_padding_uses_fixed_sample_width(width: int):
    sampled = torch.arange(3 * width, dtype=torch.int64).reshape(3, width)

    padded = _pad_sampled_tokens(sampled, max_sample_len=4)

    assert padded.shape == (3, 4)
    torch.testing.assert_close(padded[:, :width], sampled)
    assert torch.all(padded[:, width:] == -1)


def test_pp_broadcast_draft_relays_third_collective(monkeypatch):
    handler = PPHandler.__new__(PPHandler)
    handler.is_last_rank = True
    handler.relay_draft_tokens = True
    handler.last_rank = 1
    handler.broadcast_group = object()
    handler.broadcast_stream = Mock()
    handler.main_stream = Mock()

    draft_tokens = Mock()
    contiguous_drafts = Mock()
    draft_tokens.to.return_value.contiguous.return_value = contiguous_drafts
    input_batch = SimpleNamespace()
    broadcast = Mock()
    monkeypatch.setattr(pp_utils, "compute_need_sampled_mask", lambda _: [True])
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)

    handler.broadcast_draft(draft_tokens, input_batch)

    draft_tokens.to.assert_called_once_with(torch.int64)
    handler.broadcast_stream.wait_stream.assert_called_once_with(handler.main_stream)
    broadcast.assert_called_once_with(
        contiguous_drafts,
        src=handler.last_rank,
        group=handler.broadcast_group,
    )
    contiguous_drafts.record_stream.assert_called_once_with(handler.broadcast_stream)


def test_pp_deferred_update_applies_relayed_draft_tokens():
    idx_mapping = torch.tensor([1, -1, 3], dtype=torch.int64)
    relayed_drafts = torch.tensor(
        [[11, 12, 13, 14], [21, 22, 23, 24], [31, 32, 33, 34]],
        dtype=torch.int64,
    )
    outputs = {
        "sampled_tokens": torch.tensor([[1], [2], [3]], dtype=torch.int64),
        "num_sampled": torch.ones(3, dtype=torch.int32),
        "num_rejected": torch.zeros(3, dtype=torch.int32),
        "idx_mapping": idx_mapping,
        "draft_tokens": relayed_drafts,
    }

    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.pp_handler = Mock()
    runner.pp_handler.get_prev_sampled_outputs.return_value = outputs
    runner.postprocess_sampled = Mock()  # type: ignore[method-assign]
    runner.req_states = SimpleNamespace(
        draft_tokens=torch.full((5, 4), -1, dtype=torch.int64)
    )

    runner.update_pp_decode_requests()

    runner.postprocess_sampled.assert_called_once_with(
        sampled_tokens=outputs["sampled_tokens"],
        num_sampled=outputs["num_sampled"],
        num_rejected=outputs["num_rejected"],
        idx_mapping=idx_mapping,
    )
    torch.testing.assert_close(runner.req_states.draft_tokens[1], relayed_drafts[0])
    torch.testing.assert_close(runner.req_states.draft_tokens[3], relayed_drafts[2])
    assert torch.all(runner.req_states.draft_tokens[0] == -1)
