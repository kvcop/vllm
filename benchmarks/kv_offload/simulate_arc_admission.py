# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic offline cache-policy simulator for CPU KV offloading.

Replays synthetic block-access traces through the real
``CPUOffloadingManager`` (no GPU, no server) and reports hit rates for
``lru``/``arc`` with and without the ``store_threshold=2`` admission filter.
It models the production call sequence of
``OffloadingConnectorScheduler`` for each request:

1. maximal-prefix ``lookup()`` per block, breaking at the first ``MISS``
   (``_maximal_prefix_lookup``), so the ``store_threshold`` lookup counter
   only ever sees the contiguous hit prefix plus one frontier block;
2. ``touch()`` over all of the request's keys (``_touch``);
3. ``prepare_store()``/``complete_store()`` for blocks not currently in the
   cache, chunked to at most ``num_blocks`` per call (production stores in
   bounded batches; the manager cannot evict write-pending blocks).

Deliberate simplifications (documented, not claimed as production behavior):
no load pinning (``prepare_load``/``complete_load`` are skipped -- in a
sequential replay they pin and immediately unpin, so only the touch above
matters), stores complete synchronously in submission order, and each
request performs a single lookup walk from position 0 (production re-walks
from the advancing ``num_locally_computed_tokens`` frontier once per
scheduler pass, which lets a request's own blocks accumulate one extra
counter hit each; omitting that makes the ``store_threshold=2`` hit rates
here a slight underestimate, and leaves the threshold-0 runs unchanged).

The traces are fully deterministic (no RNG). Run:

    .venv/bin/python -m benchmarks.kv_offload.simulate_arc_admission
"""

import argparse
import json
from collections.abc import Iterator
from dataclasses import dataclass, field

from vllm.v1.kv_offload.base import LookupResult, ReqContext, make_offload_key
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

HOT_SET_SIZE = 8
"""Blocks shared by every "agent" request (system prompt / tool prefix)."""

CACHE_NUM_BLOCKS = 32
"""Simulated CPU tier capacity in blocks."""


def _ctx() -> ReqContext:
    return ReqContext(req_id="", kv_transfer_params=None)


@dataclass(frozen=True)
class SimConfig:
    """One cache configuration under comparison."""

    cache_policy: str = "lru"
    store_threshold: int = 0
    num_blocks: int = CACHE_NUM_BLOCKS
    max_tracker_size: int = 64_000


@dataclass
class SimResult:
    """Counters gathered over one (trace, config) replay."""

    accesses: int = 0
    hits: int = 0
    stores: int = 0
    store_batches_refused: int = 0
    store_batches_filtered: int = 0
    evictions: int = 0
    stores_skipped: int = 0
    admitted_blocks_peak: int = 0
    hit_rate_history: list[float] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        return self.hits / self.accesses if self.accesses else 0.0


def _chunked(keys: list[int], size: int) -> Iterator[list[int]]:
    for start in range(0, len(keys), size):
        yield keys[start : start + size]


def run_trace(trace: list[list[int]], config: SimConfig) -> SimResult:
    """Replay ``trace`` through a real CPUOffloadingManager.

    Args:
        trace: Requests as lists of integer block ids, in arrival order.
        config: Policy and admission-filter configuration.

    Returns:
        Counters over the whole replay.
    """
    manager = CPUOffloadingManager(
        num_blocks=config.num_blocks,
        cache_policy=config.cache_policy,
        store_threshold=config.store_threshold,
        max_tracker_size=config.max_tracker_size,
    )
    ctx = _ctx()
    result = SimResult()
    stored: set[int] = set()

    for request in trace:
        keys = [make_offload_key(str(block).encode(), 0) for block in request]

        # 1. maximal-prefix lookup: break at the first MISS, exactly like
        #    _maximal_prefix_lookup in the offloading scheduler.
        for key in keys:
            match manager.lookup(key, ctx):
                case LookupResult.HIT | LookupResult.HIT_PENDING:
                    result.hits += 1
                case LookupResult.MISS:
                    break
        result.accesses += len(keys)

        # 2. touch every key of the request (production _touch).
        manager.touch(keys, ctx)

        # 3. store blocks not currently in the cache, chunked to capacity.
        to_store = [block for block in request if block not in stored]
        for chunk in _chunked(to_store, config.num_blocks):
            output = manager.prepare_store(
                [make_offload_key(str(block).encode(), 0) for block in chunk], ctx
            )
            if output is None:
                # eviction could not be satisfied (soft failure: the
                # scheduler retries these chunks on a later batch)
                result.store_batches_refused += 1
                continue
            if not output.keys_to_store:
                # every key was filtered by the admission threshold
                result.store_batches_filtered += 1
                continue
            manager.complete_store(output.keys_to_store, ctx)
            result.stores += len(output.keys_to_store)
            stored.update(int(key[:-4]) for key in output.keys_to_store)
            for key in output.evicted_keys:
                stored.discard(int(key[:-4]))
            result.evictions += len(output.evicted_keys)

        result.stores_skipped += manager.stores_skipped_in_current_batch
        manager.stores_skipped_in_current_batch = 0
        result.admitted_blocks_peak = max(result.admitted_blocks_peak, len(stored))
        result.hit_rate_history.append(result.hit_rate)

    return result


def growing_agent_sessions(
    turns: int = 40,
    hot_set_size: int = HOT_SET_SIZE,
    new_blocks_per_turn: int = 4,
) -> list[list[int]]:
    """One agent conversation that grows every turn.

    Request ``t`` (1-based) re-reads the full growing context: the shared
    hot prefix ``[0, hot_set_size)`` plus conversation blocks
    ``[hot_set_size, hot_set_size + t * new_blocks_per_turn)``. Every block
    of the session is re-referenced by every later turn until the context
    outgrows the cache, after which only the most recent ``num_blocks``
    blocks can ever hit.
    """
    trace = []
    for turn in range(1, turns + 1):
        context_len = hot_set_size + turn * new_blocks_per_turn
        trace.append(list(range(context_len)))
    return trace


def one_shot_echo_scans(
    requests: int = 24,
    hot_set_size: int = HOT_SET_SIZE,
    scan_blocks: int = 16,
) -> list[list[int]]:
    """Short-lived Echo-style scans sharing only the hot prefix.

    Request ``i`` (0-based) reads the hot prefix ``[0, hot_set_size)`` plus
    ``scan_blocks`` fresh blocks unique to that request, never referenced
    again. Models service-isolated one-shot traffic (per-request prefix
    exclusion): the scan bodies are pure cache pollution.
    """
    trace = []
    for i in range(requests):
        base = hot_set_size + i * scan_blocks
        trace.append(list(range(hot_set_size)) + list(range(base, base + scan_blocks)))
    return trace


def returning_long_session(
    session_blocks: int = 24,
    hot_set_size: int = HOT_SET_SIZE,
    short_body_blocks: int = 8,
    flood_blocks: int = 8,
) -> list[list[int]]:
    """A long session that returns after one-shot floods evict parts of it.

    Blocks: ``H = [0, hot_set_size)`` is the shared hot prefix; the long
    session ``A = [100, 100 + session_blocks)`` fits in the cache beside H;
    short agent requests ``S_i = H + fresh8`` keep touching H while it is
    cached; flood requests read ``flood_blocks`` fresh one-shot blocks with
    no hot prefix (service-isolated Echo traffic).

    Trace (16 requests):
    ``[H+A] S S F S F S [H+A] F F S S [H+A] S S``. The two single floods
    evict a quarter of the cache each; the double flood plus two short
    requests before pass 3 flush everything the policy has not pinned.
    """
    hot = list(range(hot_set_size))
    session = list(range(100, 100 + session_blocks))

    def short(i: int) -> list[int]:
        base = 500 + i * short_body_blocks
        return hot + list(range(base, base + short_body_blocks))

    def flood(i: int) -> list[int]:
        base = 1000 + i * flood_blocks
        return list(range(base, base + flood_blocks))

    return [
        hot + session,  # pass 1: admits H and the whole session
        short(0),
        short(1),
        flood(0),
        short(2),
        flood(1),
        short(3),
        hot + session,  # pass 2: how much of the session survived?
        flood(2),
        flood(3),  # double flood
        short(4),
        short(5),
        hot + session,  # pass 3: hot prefix gone for LRU, T2-pinned for ARC
        short(6),
        short(7),
        short(8),
    ]


TRACE_FAMILIES = {
    "growing_agent_sessions": growing_agent_sessions,
    "one_shot_echo_scans": one_shot_echo_scans,
    "returning_long_session": returning_long_session,
}


def run_all(configs: dict[str, SimConfig] | None = None) -> dict:
    """Replay every trace family under every configuration.

    Args:
        configs: Mapping of configuration label to SimConfig. Defaults to
            the four lru/arc x threshold combinations under comparison.

    Returns:
        Nested dict ``trace -> config -> metrics`` ready for ``json.dumps``.
    """
    if configs is None:
        configs = {
            "lru+thr0": SimConfig(cache_policy="lru", store_threshold=0),
            "arc+thr0": SimConfig(cache_policy="arc", store_threshold=0),
            "lru+thr2": SimConfig(cache_policy="lru", store_threshold=2),
            "arc+thr2": SimConfig(cache_policy="arc", store_threshold=2),
        }
    report: dict = {}
    for trace_name, builder in TRACE_FAMILIES.items():
        trace = builder()
        num_accesses = sum(len(request) for request in trace)
        report[trace_name] = {
            "requests": len(trace),
            "block_accesses": num_accesses,
            "configs": {},
        }
        for label, config in configs.items():
            result = run_trace(trace, config)
            report[trace_name]["configs"][label] = {
                "hit_rate": round(result.hit_rate, 4),
                "hits": result.hits,
                "stores": result.stores,
                "evictions": result.evictions,
                "stores_skipped": result.stores_skipped,
                "store_batches_refused": result.store_batches_refused,
                "store_batches_filtered": result.store_batches_filtered,
                "admitted_blocks_peak": result.admitted_blocks_peak,
            }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", help="Optional path to write the report as JSON.")
    args = parser.parse_args()

    report = run_all()
    for trace_name, trace_report in report.items():
        print(
            f"\n=== {trace_name} ({trace_report['requests']} requests, "
            f"{trace_report['block_accesses']} block accesses) ==="
        )
        header = (
            f"{'config':<10} {'hit_rate':>8} {'hits':>6} {'stores':>7} "
            f"{'evictions':>9} {'skipped':>8} {'refused':>8}"
        )
        print(header)
        print("-" * len(header))
        for label, metrics in trace_report["configs"].items():
            print(
                f"{label:<10} {metrics['hit_rate']:>8.4f} {metrics['hits']:>6} "
                f"{metrics['stores']:>7} {metrics['evictions']:>9} "
                f"{metrics['stores_skipped']:>8} "
                f"{metrics['store_batches_refused']:>8}"
            )
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nReport written to {args.json}")


if __name__ == "__main__":
    main()
