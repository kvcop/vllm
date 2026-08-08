# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import stat

import numpy as np
import pytest

from vllm.v1.worker.gpu.spec_decode.dspark.calibration_collector import (
    CAPTURE_SCHEMA,
    DSparkCalibrationCollector,
    build_prefix_mask,
    is_capture_writer,
)


@pytest.mark.parametrize(
    ("accepted", "verified", "width", "expected"),
    [
        (3, 3, 3, [1, 1, 1]),
        (0, 3, 3, [0, 0, 0]),
        (2, 4, 5, [1, 1, 0, 0, 0]),
        (1, 1, 4, [1, 0, 0, 0]),
        (0, 0, 4, [0, 0, 0, 0]),
    ],
)
def test_prefix_labels_cover_full_reject_middle_variable_and_censored(
    accepted, verified, width, expected
):
    actual = build_prefix_mask(accepted, verified, width)

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.uint8


def _collector(tmp_path, *, max_rows=20, shard_rows=20, width=3, slots=4):
    return DSparkCalibrationCollector(
        tmp_path,
        max_rows=max_rows,
        num_speculative_tokens=width,
        max_num_slots=slots,
        dp_rank=2,
        shard_rows=shard_rows,
    )


def _observe(
    collector,
    *,
    slots,
    verified,
    accepted,
    prefilling=None,
):
    slots = np.asarray(slots, dtype=np.int64)
    verified = np.asarray(verified, dtype=np.int64)
    accepted = np.asarray(accepted, dtype=np.int64)
    cu_num_logits = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(verified + 1))
    )
    if prefilling is None:
        prefilling = np.zeros(slots.size, dtype=np.bool_)
    return collector.observe(
        idx_mapping=slots,
        cu_num_logits=cu_num_logits,
        is_prefilling=np.asarray(prefilling, dtype=np.bool_),
        num_sampled=accepted + 1,
        num_rejected=verified - accepted,
        num_bonus_tokens=1,
    )


def test_reordered_idx_mapping_joins_by_slot_and_consumes_once(tmp_path):
    collector = _collector(tmp_path)
    collector.add_request(0)
    collector.add_request(2)
    proposal_slots = np.array([2, 0])
    proposal_mask = _observe(
        collector,
        slots=proposal_slots,
        verified=[0, 0],
        accepted=[0, 0],
    )
    raw = np.array([[20.0, 21.0, 22.0], [0.0, 1.0, 2.0]], dtype=np.float32)
    collector.record_proposal(
        raw_logits=raw,
        idx_mapping=proposal_slots,
        proposal_mask=proposal_mask,
    )

    _observe(
        collector,
        slots=[0, 2],
        verified=[3, 2],
        accepted=[1, 2],
    )
    _observe(
        collector,
        slots=[0, 2],
        verified=[3, 2],
        accepted=[1, 2],
    )
    collector.close()

    [shard] = sorted((tmp_path / "dp-0002").glob("*.npz"))
    with np.load(shard) as data:
        np.testing.assert_array_equal(data["raw_logits"], raw[[1, 0]])
        np.testing.assert_array_equal(data["verified_lengths"], [3, 2])
        np.testing.assert_array_equal(data["accepted_counts"], [1, 2])
        ordinals = data["request_ordinal"]
        assert ordinals[0] > 0
        assert ordinals[1] == ordinals[0] + 1


def test_slot_reuse_invalidates_pending_proposal(tmp_path):
    collector = _collector(tmp_path)
    collector.add_request(1)
    proposal_mask = _observe(collector, slots=[1], verified=[0], accepted=[0])
    collector.record_proposal(
        raw_logits=np.ones((1, 3), dtype=np.float32),
        idx_mapping=np.array([1]),
        proposal_mask=proposal_mask,
    )

    collector.add_request(1)
    _observe(collector, slots=[1], verified=[3], accepted=[3])
    collector.close()

    assert collector.total_rows == 0
    assert not list((tmp_path / "dp-0002").glob("*.npz"))


def test_pending_slot_absent_from_next_batch_is_invalidated(tmp_path):
    collector = _collector(tmp_path)
    collector.add_request(0)
    collector.add_request(1)
    proposal_mask = _observe(collector, slots=[0, 1], verified=[0, 0], accepted=[0, 0])
    collector.record_proposal(
        raw_logits=np.ones((2, 3), dtype=np.float32),
        idx_mapping=np.array([0, 1]),
        proposal_mask=proposal_mask,
    )

    _observe(collector, slots=[0], verified=[3], accepted=[3])
    assert collector.pending_valid.tolist() == [False, False, False, False]
    _observe(collector, slots=[1], verified=[3], accepted=[3])
    collector.close()

    assert collector.total_rows == 1


def test_prefill_zero_sample_and_invariant_failure_drop_whole_rows(tmp_path):
    collector = _collector(tmp_path)
    for slot in range(3):
        collector.add_request(slot)
    proposal_mask = _observe(
        collector, slots=[0, 1, 2], verified=[0, 0, 0], accepted=[0, 0, 0]
    )
    collector.record_proposal(
        raw_logits=np.ones((3, 3), dtype=np.float32),
        idx_mapping=np.array([0, 1, 2]),
        proposal_mask=proposal_mask,
    )

    collector.observe(
        idx_mapping=np.array([0, 1, 2]),
        cu_num_logits=np.array([0, 4, 8, 12]),
        is_prefilling=np.array([True, False, False]),
        num_sampled=np.array([4, 0, 2]),
        num_rejected=np.array([0, 0, 0]),
        num_bonus_tokens=1,
    )
    collector.close()

    assert collector.total_rows == 0


def test_hard_cap_shards_schema_dtypes_and_private_modes(tmp_path):
    collector = _collector(tmp_path, max_rows=5, shard_rows=2, width=3, slots=1)
    collector.add_request(0)
    for index in range(7):
        proposal_mask = _observe(
            collector, slots=[0], verified=[3], accepted=[index % 4]
        )
        collector.record_proposal(
            raw_logits=np.full((1, 3), index, dtype=np.float32),
            idx_mapping=np.array([0]),
            proposal_mask=proposal_mask,
        )
    collector.close()

    output_dir = tmp_path / "dp-0002"
    shards = sorted(output_dir.glob("*.npz"))
    assert [path.name for path in shards] == [
        "capture-dp0002-shard000000.npz",
        "capture-dp0002-shard000001.npz",
        "capture-dp0002-shard000002.npz",
    ]
    assert collector.total_rows == 5
    assert stat.S_IMODE(tmp_path.stat().st_mode) == 0o700
    assert stat.S_IMODE(output_dir.stat().st_mode) == 0o700

    row_counts = []
    for shard in shards:
        assert stat.S_IMODE(shard.stat().st_mode) == 0o600
        with np.load(shard) as data:
            assert frozenset(data.files) == CAPTURE_SCHEMA
            row_counts.append(data["raw_logits"].shape[0])
            assert data["raw_logits"].dtype == np.float32
            assert data["raw_logits"].shape[1:] == (3,)
            assert data["prefix_mask"].dtype == np.uint8
            assert data["verified_lengths"].dtype == np.uint8
            assert data["accepted_counts"].dtype == np.uint8
            assert data["request_ordinal"].dtype == np.uint64
            assert data["proposal_seq"].dtype == np.uint32
            assert data["engine_step"].dtype == np.uint64
            forbidden = {
                "req_id",
                "token_ids",
                "prompt",
                "response",
                "hidden_states",
                "vocab_logits",
            }
            assert forbidden.isdisjoint(data.files)
    assert row_counts == [2, 2, 1]


def test_writer_election_is_one_tp_rank_on_last_pp_stage():
    last_stage = [
        is_capture_writer(is_last_pp_rank=True, tp_rank=rank, tp_world_size=4)
        for rank in range(4)
    ]
    earlier_stage = [
        is_capture_writer(is_last_pp_rank=False, tp_rank=rank, tp_world_size=4)
        for rank in range(4)
    ]

    assert sum(last_stage) == 1
    assert last_stage[0]
    assert not any(earlier_stage)
