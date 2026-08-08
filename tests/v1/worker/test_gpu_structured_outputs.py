# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest

from vllm.v1.worker.gpu.structured_outputs import _prepare_grammar_bitmask


def test_adaptive_trim_selects_each_requests_mask_prefix():
    input_batch = SimpleNamespace(
        # Adaptive verification may reorder requests before the GPU run.
        req_ids=["plain", "grammar-b", "grammar-a"],
        cu_num_logits_np=np.array([0, 1, 3, 6], dtype=np.int32),
        original_num_logits_per_req=np.array([1, 5, 4], dtype=np.int32),
    )
    grammar_bitmask = np.array(
        [[10], [11], [12], [13], [20], [21], [22], [23], [24]],
        dtype=np.int32,
    )

    mapping, selected = _prepare_grammar_bitmask(
        input_batch,
        ["grammar-a", "grammar-b"],
        grammar_bitmask,
        mask_stride=8,
    )

    # grammar-a keeps three of four rows; grammar-b keeps two of five. The
    # row immediately after each retained draft prefix becomes its bonus row.
    assert mapping == [16, 17, 18, 8, 9]
    np.testing.assert_array_equal(selected[:, 0], [10, 11, 12, 20, 21])


def test_untrimmed_layout_reuses_original_bitmask():
    input_batch = SimpleNamespace(
        req_ids=["grammar"],
        cu_num_logits_np=np.array([0, 3], dtype=np.int32),
        original_num_logits_per_req=np.array([3], dtype=np.int32),
    )
    grammar_bitmask = np.arange(6, dtype=np.int32).reshape(3, 2)

    mapping, selected = _prepare_grammar_bitmask(
        input_batch, ["grammar"], grammar_bitmask, mask_stride=8
    )

    assert mapping == [0, 1, 2]
    assert selected is grammar_bitmask


def test_adaptive_layout_rejects_more_logits_than_scheduler_masks():
    input_batch = SimpleNamespace(
        req_ids=["grammar"],
        cu_num_logits_np=np.array([0, 4], dtype=np.int32),
        original_num_logits_per_req=np.array([3], dtype=np.int32),
    )

    with pytest.raises(RuntimeError, match="exceeds its scheduler mask layout"):
        _prepare_grammar_bitmask(
            input_batch,
            ["grammar"],
            np.zeros((3, 2), dtype=np.int32),
            mask_stride=8,
        )


def test_missing_original_layout_fails_loudly():
    input_batch = SimpleNamespace(
        req_ids=["grammar"],
        cu_num_logits_np=np.array([0, 2], dtype=np.int32),
        original_num_logits_per_req=None,
    )

    with pytest.raises(RuntimeError, match="requires the original per-request"):
        _prepare_grammar_bitmask(
            input_batch,
            ["grammar"],
            np.zeros((3, 2), dtype=np.int32),
            mask_stride=8,
        )


def test_scheduler_mask_row_count_mismatch_fails_loudly():
    input_batch = SimpleNamespace(
        req_ids=["grammar"],
        cu_num_logits_np=np.array([0, 2], dtype=np.int32),
        original_num_logits_per_req=np.array([3], dtype=np.int32),
    )

    with pytest.raises(RuntimeError, match="masks=4, expected=3"):
        _prepare_grammar_bitmask(
            input_batch,
            ["grammar"],
            np.zeros((4, 2), dtype=np.int32),
            mask_stride=8,
        )
