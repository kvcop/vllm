# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the offline ARC/LRU admission simulator.

Pins the measured hit rates of
``benchmarks/kv_offload/simulate_arc_admission.py`` on its three synthetic
trace families so that any change to the cache policies, the admission
filter, or the modeled call sequence shows up as a test failure instead of
a silent behavior shift. The numbers describe the synthetic traces only --
they are not claims about production hit rates.
"""

import json
from pathlib import Path

import pytest

from benchmarks.kv_offload.simulate_arc_admission import (
    CACHE_NUM_BLOCKS,
    TRACE_FAMILIES,
    RequestAccessTraceError,
    growing_agent_sessions,
    load_request_access_trace,
    one_shot_echo_scans,
    report_request_access_trace,
    returning_long_session,
    run_all,
)

HOT_SET = list(range(8))

# (hit_rate, hits, stores, evictions, stores_skipped, store_batches_refused)
# per trace family and cache configuration, measured on 2026-08-31.
BASELINE = {
    "growing_agent_sessions": {
        "lru+thr0": (0.0678, 244, 2412, 2380, 0, 0),
        "arc+thr0": (0.1278, 460, 1640, 1608, 0, 30),
        "lru+thr2": (0.1056, 380, 20, 0, 3200, 0),
        "arc+thr2": (0.1056, 380, 20, 0, 3200, 0),
    },
    "one_shot_echo_scans": {
        "lru+thr0": (0.3194, 184, 392, 360, 0, 0),
        "arc+thr0": (0.3194, 184, 392, 360, 0, 0),
        "lru+thr2": (0.2083, 120, 8, 0, 448, 0),
        "arc+thr2": (0.2083, 120, 8, 0, 448, 0),
    },
    "returning_long_session": {
        "lru+thr0": (0.2941, 80, 192, 160, 0, 0),
        "arc+thr0": (0.2721, 74, 174, 142, 0, 1),
        "lru+thr2": (0.1103, 30, 6, 0, 236, 0),
        "arc+thr2": (0.1103, 30, 6, 0, 236, 0),
    },
}


def _metrics(report: dict, trace: str, config: str) -> tuple:
    metrics = report[trace]["configs"][config]
    return (
        metrics["hit_rate"],
        metrics["hits"],
        metrics["stores"],
        metrics["evictions"],
        metrics["stores_skipped"],
        metrics["store_batches_refused"],
    )


def test_traces_match_their_definitions():
    # growing agent sessions: every turn re-reads the whole growing context
    trace = growing_agent_sessions()
    assert len(trace) == 40
    assert trace[0] == HOT_SET + [8, 9, 10, 11]
    assert trace[-1] == list(range(8 + 40 * 4))

    # one-shot Echo scans: hot prefix plus request-unique scan blocks
    trace = one_shot_echo_scans()
    assert len(trace) == 24
    assert trace[0] == HOT_SET + list(range(8, 24))
    assert trace[1] == HOT_SET + list(range(24, 40))
    scan_blocks = [block for req in trace for block in req[8:]]
    assert len(scan_blocks) == len(set(scan_blocks))  # never re-referenced

    # returning long session: 3 session passes around one-shot floods
    trace = returning_long_session()
    assert len(trace) == 16
    session = list(range(100, 124))
    assert trace[0] == HOT_SET + session
    flood_blocks = [b for req in trace for b in req if b >= 1000]
    assert len(flood_blocks) == len(set(flood_blocks))
    # floods and short-agent bodies never collide with the session blocks
    assert not set(flood_blocks) & set(session)


def test_replay_is_deterministic():
    assert run_all() == run_all()


def test_hit_rates_match_recorded_baseline():
    report = run_all()
    for trace, configs in BASELINE.items():
        for config, expected in configs.items():
            assert _metrics(report, trace, config) == expected, (trace, config)


def test_admission_filter_eliminates_one_shot_pollution():
    """threshold=2 stores only the repeatedly-seen hot prefix on scans."""
    report = run_all()
    scan = report["one_shot_echo_scans"]["configs"]
    for policy in ("lru", "arc"):
        unfiltered, filtered = scan[f"{policy}+thr0"], scan[f"{policy}+thr2"]
        # every scan block is stored and evicted under threshold 0
        assert unfiltered["stores"] == 392
        assert unfiltered["evictions"] == 360
        # under threshold 2 only the 8 hot-prefix blocks are ever stored
        assert filtered["stores"] == 8
        assert filtered["evictions"] == 0
        assert filtered["stores_skipped"] == 448
        # the hot prefix takes several requests to be admitted, so the hit
        # rate is lower than the unfiltered run on this short trace
        assert filtered["hit_rate"] < unfiltered["hit_rate"]


def test_arc_advantage_comes_from_refusing_batch_evictions():
    """ARC only beats LRU on growing sessions, and only because evict()
    refuses batches once the adaptive target demands T2 victims that do not
    exist -- the refusal preserves the adapted T1 working set. With the
    refusal counted as zero for LRU, the whole gap is attributable to it.
    """
    report = run_all()
    growing = report["growing_agent_sessions"]["configs"]
    assert growing["arc+thr0"]["hits"] > growing["lru+thr0"]["hits"]
    assert growing["arc+thr0"]["store_batches_refused"] == 30
    assert growing["lru+thr0"]["store_batches_refused"] == 0


def test_threshold_2_makes_the_policy_choice_irrelevant_here():
    """With admission filtering nothing is ever evicted on these traces, so
    LRU and ARC behave identically."""
    report = run_all()
    for trace in TRACE_FAMILIES:
        configs = report[trace]["configs"]
        assert _metrics(report, trace, "lru+thr2") == _metrics(
            report, trace, "arc+thr2"
        )
        assert configs["lru+thr2"]["evictions"] == 0


def test_threshold_2_serializes_admission_along_the_lookup_frontier():
    """The maximal-prefix lookup breaks at the first miss, so a returning
    long session is admitted one block per pass: threshold=2 costs most of
    the hit rate on that family even though nothing is evicted."""
    report = run_all()
    session = report["returning_long_session"]["configs"]
    assert session["lru+thr2"]["stores"] == 6  # 6 hot-prefix/frontier blocks
    assert session["lru+thr2"]["hits"] < session["lru+thr0"]["hits"]
    assert CACHE_NUM_BLOCKS == 32


def _access_row(
    *,
    seq: int,
    event_idx: int,
    request_seq: int,
    group_idx: int,
    hashes: list[int],
    engine_id: str = "engine-a",
    data_parallel_rank: int = 0,
    run_index: int = 0,
    lookup_performed: bool = True,
    group_count: int = 1,
) -> dict:
    return {
        "seq": seq,
        "event_idx": event_idx,
        "run_index": run_index,
        "ts": float(seq),
        "kind": "request_access",
        "schema_version": 2,
        "engine_id": engine_id,
        "data_parallel_rank": data_parallel_rank,
        "request_seq": request_seq,
        "pass_index": 0,
        "lookup_performed": lookup_performed,
        "group_count": group_count,
        "group_idx": group_idx,
        "terminal_block_hashes": hashes,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(f"{json.dumps(row)}\n" for row in rows))
    return path


def _residency_row(kind: str, *, block_hash: int) -> dict:
    row = {
        "seq": 9,
        "event_idx": 0,
        "run_index": 0,
        "ts": 9.0,
        "kind": kind,
        "medium": "CPU",
        "group_idx": 0,
        "block_hashes": [block_hash],
    }
    if kind == "stored":
        row.update({"block_size": 16, "num_tokens": 16})
    return row


def test_request_access_trace_restores_publisher_and_request_order(tmp_path: Path):
    high_hash = 901_234_567_890_123
    rows = [
        _residency_row("stored", block_hash=high_hash + 100),
        # Recovered publisher frames are deliberately written out of order.
        _access_row(
            seq=11,
            event_idx=0,
            request_seq=1,
            group_idx=0,
            hashes=[high_hash, high_hash + 2, high_hash + 3],
        ),
        _access_row(
            seq=10,
            event_idx=0,
            request_seq=0,
            group_idx=0,
            hashes=[high_hash, high_hash + 1],
        ),
        # A second publisher may reuse seq/event_idx; validation is per stream.
        _access_row(
            seq=10,
            event_idx=0,
            request_seq=0,
            group_idx=0,
            hashes=[high_hash + 4],
            engine_id="engine-b",
            data_parallel_rank=1,
        ),
        _residency_row("removed", block_hash=high_hash + 100),
    ]
    trace = load_request_access_trace(_write_jsonl(tmp_path / "trace.jsonl", rows))

    assert len(trace.streams) == 2
    first = trace.streams[0]
    assert first.group_indices == (0,)
    assert tuple(request.blocks for request in first.requests) == (
        ((0, high_hash), (0, high_hash + 1)),
        ((0, high_hash), (0, high_hash + 2), (0, high_hash + 3)),
    )
    assert tuple(request.blocks for request in trace.streams[1].requests) == (
        ((0, high_hash + 4),),
    )

    summary = report_request_access_trace(trace)
    assert summary["requests"] == 3
    assert summary["block_accesses"] == 6
    assert summary["blocks_touched"] == 6
    assert summary["group_counts"] == [1, 1]
    # Opaque hashes are accepted only as replay keys, never copied to a report.
    serialized = json.dumps(summary)
    assert str(high_hash) not in serialized
    assert "engine-a" not in serialized
    assert "engine-b" not in serialized


def test_request_access_trace_rejects_incomplete_group_set(tmp_path: Path):
    rows = [
        _access_row(
            seq=1,
            event_idx=0,
            request_seq=0,
            group_idx=0,
            hashes=[10],
            group_count=2,
        ),
        _access_row(
            seq=1,
            event_idx=1,
            request_seq=0,
            group_idx=1,
            hashes=[20],
            group_count=2,
        ),
        _access_row(
            seq=2,
            event_idx=0,
            request_seq=1,
            group_idx=0,
            hashes=[10],
            group_count=2,
        ),
    ]

    with pytest.raises(RequestAccessTraceError, match="incomplete/inconsistent"):
        load_request_access_trace(_write_jsonl(tmp_path / "trace.jsonl", rows))


def test_request_access_trace_rejects_uniformly_missing_last_group(tmp_path: Path):
    rows = [
        _access_row(
            seq=1,
            event_idx=0,
            request_seq=0,
            group_idx=0,
            hashes=[10],
            group_count=2,
        ),
        _access_row(
            seq=2,
            event_idx=0,
            request_seq=1,
            group_idx=0,
            hashes=[10],
            group_count=2,
        ),
    ]

    with pytest.raises(RequestAccessTraceError, match="complete, ordered"):
        load_request_access_trace(_write_jsonl(tmp_path / "trace.jsonl", rows))


def test_request_access_trace_rejects_complete_multi_group_replay(tmp_path: Path):
    rows = [
        _access_row(
            seq=1,
            event_idx=0,
            request_seq=0,
            group_idx=0,
            hashes=[10],
            group_count=2,
        ),
        _access_row(
            seq=1,
            event_idx=1,
            request_seq=0,
            group_idx=1,
            hashes=[20],
            group_count=2,
        ),
    ]

    with pytest.raises(RequestAccessTraceError, match="only single-group"):
        load_request_access_trace(_write_jsonl(tmp_path / "trace.jsonl", rows))


def test_request_access_trace_rejects_inconsistent_group_count(tmp_path: Path):
    rows = [
        _access_row(
            seq=1,
            event_idx=0,
            request_seq=0,
            group_idx=0,
            hashes=[10],
            group_count=2,
        ),
        _access_row(
            seq=1,
            event_idx=1,
            request_seq=0,
            group_idx=1,
            hashes=[20],
            group_count=3,
        ),
    ]

    with pytest.raises(RequestAccessTraceError, match="group_count must be consistent"):
        load_request_access_trace(_write_jsonl(tmp_path / "trace.jsonl", rows))


@pytest.mark.parametrize("group_count", [0, -1, True])
def test_request_access_trace_rejects_invalid_group_count(
    tmp_path: Path, group_count: int
):
    rows = [
        _access_row(
            seq=1,
            event_idx=0,
            request_seq=0,
            group_idx=0,
            hashes=[10],
            group_count=group_count,
        ),
    ]

    with pytest.raises(RequestAccessTraceError, match="group_count must be a positive"):
        load_request_access_trace(_write_jsonl(tmp_path / "trace.jsonl", rows))


def test_request_access_trace_rejects_request_sequence_gap(tmp_path: Path):
    rows = [
        _access_row(
            seq=1,
            event_idx=0,
            request_seq=0,
            group_idx=0,
            hashes=[10],
        ),
        _access_row(seq=2, event_idx=0, request_seq=2, group_idx=0, hashes=[20]),
    ]

    with pytest.raises(RequestAccessTraceError, match="gap-free"):
        load_request_access_trace(_write_jsonl(tmp_path / "trace.jsonl", rows))


def test_request_access_trace_rejects_incomplete_prefix_of_request_sequence(
    tmp_path: Path,
):
    rows = [
        _access_row(seq=2, event_idx=0, request_seq=1, group_idx=0, hashes=[20]),
    ]

    with pytest.raises(RequestAccessTraceError, match="start at zero"):
        load_request_access_trace(_write_jsonl(tmp_path / "trace.jsonl", rows))


def test_touch_only_request_skips_lookup_accounting_but_still_updates_policy(
    tmp_path: Path,
):
    rows = [
        _access_row(seq=1, event_idx=0, request_seq=0, group_idx=0, hashes=[10]),
        _access_row(
            seq=2,
            event_idx=0,
            request_seq=1,
            group_idx=0,
            hashes=[10],
            lookup_performed=False,
        ),
        _access_row(seq=3, event_idx=0, request_seq=2, group_idx=0, hashes=[10]),
    ]
    trace = load_request_access_trace(_write_jsonl(tmp_path / "trace.jsonl", rows))

    assert [request.lookup_performed for request in trace.streams[0].requests] == [
        True,
        False,
        True,
    ]
    summary = report_request_access_trace(trace)
    assert summary["block_accesses"] == 2
    assert summary["blocks_touched"] == 3
    assert summary["configs"]["lru+thr0"]["hits"] == 1


def test_request_access_trace_rejects_mixed_lookup_semantics_within_request(
    tmp_path: Path,
):
    rows = [
        _access_row(
            seq=1,
            event_idx=0,
            request_seq=0,
            group_idx=0,
            hashes=[10],
            group_count=2,
        ),
        _access_row(
            seq=1,
            event_idx=1,
            request_seq=0,
            group_idx=1,
            hashes=[20],
            lookup_performed=False,
            group_count=2,
        ),
    ]

    with pytest.raises(RequestAccessTraceError, match="agree across all groups"):
        load_request_access_trace(_write_jsonl(tmp_path / "trace.jsonl", rows))


@pytest.mark.parametrize("kind", ["sequence_reset", "cleared", "decode_error"])
def test_request_access_trace_rejects_boundaries_and_decode_errors(
    tmp_path: Path, kind: str
):
    rows = [
        _access_row(seq=1, event_idx=0, request_seq=0, group_idx=0, hashes=[10]),
        {"kind": kind},
    ]

    with pytest.raises(RequestAccessTraceError, match="forbidden"):
        load_request_access_trace(_write_jsonl(tmp_path / "trace.jsonl", rows))


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("schema_version", 1, "unsupported request-access schema"),
        ("pass_index", 1, "unsupported request-access pass"),
        ("lookup_performed", 1, "lookup_performed must be bool"),
        ("request_id", "private-id", "privacy-sensitive"),
        ("token_ids", [1, 2, 3], "privacy-sensitive"),
    ],
)
def test_request_access_trace_rejects_schema_pass_and_private_fields(
    tmp_path: Path, field_name: str, value: object, message: str
):
    row = _access_row(seq=1, event_idx=0, request_seq=0, group_idx=0, hashes=[10])
    row[field_name] = value

    with pytest.raises(RequestAccessTraceError, match=message):
        load_request_access_trace(_write_jsonl(tmp_path / "trace.jsonl", [row]))


def test_request_access_trace_rejects_duplicate_publisher_position(tmp_path: Path):
    rows = [
        _access_row(
            seq=1,
            event_idx=0,
            request_seq=0,
            group_idx=0,
            hashes=[10],
            group_count=2,
        ),
        _access_row(
            seq=1,
            event_idx=0,
            request_seq=0,
            group_idx=1,
            hashes=[20],
            group_count=2,
        ),
    ]

    with pytest.raises(RequestAccessTraceError, match="duplicate publisher position"):
        load_request_access_trace(_write_jsonl(tmp_path / "trace.jsonl", rows))


@pytest.mark.parametrize("kind", ["stored", "removed"])
@pytest.mark.parametrize("field_name", ["request_id", "token_ids", "unknown"])
def test_request_access_trace_rejects_extra_residency_fields(
    tmp_path: Path, kind: str, field_name: str
):
    residency = _residency_row(kind, block_hash=123)
    residency[field_name] = "private" if field_name != "token_ids" else [1, 2]
    rows = [
        residency,
        _access_row(seq=1, event_idx=0, request_seq=0, group_idx=0, hashes=[10]),
    ]

    with pytest.raises(RequestAccessTraceError, match="privacy-sensitive"):
        load_request_access_trace(_write_jsonl(tmp_path / "trace.jsonl", rows))
