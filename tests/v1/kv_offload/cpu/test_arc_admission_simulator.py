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

from benchmarks.kv_offload.simulate_arc_admission import (
    CACHE_NUM_BLOCKS,
    TRACE_FAMILIES,
    growing_agent_sessions,
    one_shot_echo_scans,
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
