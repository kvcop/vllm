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

An opt-in live observer trace can replace the synthetic families:

    .venv/bin/python -m benchmarks.kv_offload.simulate_arc_admission \
        --request-access-jsonl trace.jsonl --json report.json

Live input is fail-closed: privacy-bearing/unknown fields, control boundaries,
decode-error markers, request-sequence gaps, incomplete KV groups, and unknown
schemas abort replay. ``lookup_performed=false`` skips maximal-prefix lookup
and hit accounting but preserves the request's policy touch and store phases.
The current consumer accepts only ``group_count=1``: hybrid multi-group replay
needs per-group chunking/convergence metadata that schema v2 does not expose.
"""

import argparse
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias, cast

from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadKey,
    ReqContext,
    make_offload_key,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

HOT_SET_SIZE = 8
"""Blocks shared by every "agent" request (system prompt / tool prefix)."""

CACHE_NUM_BLOCKS = 32
"""Simulated CPU tier capacity in blocks."""

TraceBlock: TypeAlias = int | tuple[int, int]
"""Synthetic id, or ``(group_idx, opaque terminal hash)`` from live events."""

StreamIdentity: TypeAlias = tuple[int, str, int]
"""``(run_index, engine_id, data_parallel_rank)`` of one scheduler stream."""


class RequestAccessTraceError(ValueError):
    """A request-access JSONL trace is incomplete, unsafe, or ambiguous."""

    @classmethod
    def at_line(cls, line_number: int, message: str) -> "RequestAccessTraceError":
        return cls(f"line {line_number}: {message}")


@dataclass(frozen=True)
class TraceRequest:
    """One replay request, including whether external lookup participated."""

    blocks: tuple[TraceBlock, ...]
    lookup_performed: bool = True


@dataclass(frozen=True)
class RequestAccessStream:
    """Validated accesses from one scheduler stream in request order."""

    identity: StreamIdentity
    group_indices: tuple[int, ...]
    requests: tuple[TraceRequest, ...]


@dataclass(frozen=True)
class RequestAccessTrace:
    """Fail-closed live trace ready for independent per-stream replay."""

    streams: tuple[RequestAccessStream, ...]

    @property
    def request_count(self) -> int:
        return sum(len(stream.requests) for stream in self.streams)

    @property
    def block_access_count(self) -> int:
        return sum(
            len(request.blocks)
            for stream in self.streams
            for request in stream.requests
            if request.lookup_performed
        )

    @property
    def touched_block_count(self) -> int:
        return sum(
            len(request.blocks)
            for stream in self.streams
            for request in stream.requests
        )


@dataclass(frozen=True)
class _RequestAccessRow:
    """One validated row before publisher-order reconstruction."""

    line_number: int
    seq: int
    event_idx: int
    run_index: int
    engine_id: str
    data_parallel_rank: int
    request_seq: int
    lookup_performed: bool
    group_count: int
    group_idx: int
    terminal_block_hashes: tuple[int, ...]

    @property
    def stream_identity(self) -> StreamIdentity:
        return (self.run_index, self.engine_id, self.data_parallel_rank)

    @property
    def publisher_position(self) -> tuple[int, int]:
        return (self.seq, self.event_idx)


_REQUEST_ACCESS_FIELDS = {
    "seq",
    "event_idx",
    "run_index",
    "ts",
    "kind",
    "schema_version",
    "engine_id",
    "data_parallel_rank",
    "request_seq",
    "pass_index",
    "lookup_performed",
    "group_count",
    "group_idx",
    "terminal_block_hashes",
}
_REJECTED_TRACE_KINDS = {"sequence_reset", "cleared", "decode_error"}
_IGNORED_RESIDENCY_KINDS = {"stored", "removed"}
_RESIDENCY_FIELDS = {
    "seq",
    "event_idx",
    "run_index",
    "ts",
    "kind",
    "medium",
    "group_idx",
    "block_hashes",
}
_STORED_FIELDS = _RESIDENCY_FIELDS | {"block_size", "num_tokens"}


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


def _chunked(keys: list[OffloadKey], size: int) -> Iterator[list[OffloadKey]]:
    for start in range(0, len(keys), size):
        yield keys[start : start + size]


def _offload_key(block: TraceBlock) -> OffloadKey:
    if isinstance(block, tuple):
        group_idx, terminal_hash = block
        return make_offload_key(str(terminal_hash).encode(), group_idx)
    return make_offload_key(str(block).encode(), 0)


def run_trace(
    trace: Sequence[Sequence[TraceBlock] | TraceRequest], config: SimConfig
) -> SimResult:
    """Replay ``trace`` through a real CPUOffloadingManager.

    Args:
        trace: Requests as block ids in arrival order. Live request-access
            keys are represented by ``(group_idx, terminal_hash)`` pairs and
            carry a request-wide external-lookup participation flag.
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
    stored: set[OffloadKey] = set()

    for request in trace:
        blocks: Sequence[TraceBlock]
        if isinstance(request, TraceRequest):
            blocks = request.blocks
            lookup_performed = request.lookup_performed
        else:
            blocks = request
            lookup_performed = True
        keys = [_offload_key(block) for block in blocks]

        # 1. maximal-prefix lookup: break at the first MISS, exactly like
        #    _maximal_prefix_lookup in the offloading scheduler.
        if lookup_performed:
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
        to_store = [key for key in keys if key not in stored]
        for chunk in _chunked(to_store, config.num_blocks):
            output = manager.prepare_store(chunk, ctx)
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
            stored.update(output.keys_to_store)
            for key in output.evicted_keys:
                stored.discard(key)
            result.evictions += len(output.evicted_keys)

        result.stores_skipped += manager.stores_skipped_in_current_batch
        manager.stores_skipped_in_current_batch = 0
        result.admitted_blocks_peak = max(result.admitted_blocks_peak, len(stored))
        result.hit_rate_history.append(result.hit_rate)

    return result


def _require_trace(condition: bool, line_number: int, message: str) -> None:
    if not condition:
        raise RequestAccessTraceError.at_line(line_number, message)


def _required_int(row: dict[str, Any], field_name: str, line_number: int) -> int:
    value = row.get(field_name)
    _require_trace(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        line_number,
        f"{field_name} must be a non-negative int",
    )
    return cast(int, value)


def _required_positive_int(
    row: dict[str, Any], field_name: str, line_number: int
) -> int:
    value = row.get(field_name)
    _require_trace(
        isinstance(value, int) and not isinstance(value, bool) and value > 0,
        line_number,
        f"{field_name} must be a positive int",
    )
    return cast(int, value)


def _parse_request_access_row(
    row: dict[str, Any], line_number: int
) -> _RequestAccessRow:
    unexpected = sorted(set(row) - _REQUEST_ACCESS_FIELDS)
    missing = sorted(_REQUEST_ACCESS_FIELDS - set(row))
    _require_trace(
        not unexpected,
        line_number,
        f"unexpected or privacy-sensitive fields: {unexpected}",
    )
    _require_trace(not missing, line_number, f"missing fields: {missing}")
    _require_trace(
        row.get("schema_version") == 2,
        line_number,
        "unsupported request-access schema",
    )
    _require_trace(
        row.get("pass_index") == 0,
        line_number,
        "unsupported request-access pass",
    )
    lookup_performed = row.get("lookup_performed")
    _require_trace(
        isinstance(lookup_performed, bool),
        line_number,
        "lookup_performed must be bool",
    )
    engine_id = row.get("engine_id")
    _require_trace(
        isinstance(engine_id, str) and bool(engine_id),
        line_number,
        "engine_id must be a non-empty string",
    )
    timestamp = row.get("ts")
    _require_trace(
        isinstance(timestamp, int | float) and not isinstance(timestamp, bool),
        line_number,
        "ts must be numeric",
    )
    raw_hashes = row.get("terminal_block_hashes")
    _require_trace(
        isinstance(raw_hashes, list),
        line_number,
        "terminal_block_hashes must be a list",
    )
    hashes = cast(list[Any], raw_hashes)
    _require_trace(
        all(isinstance(value, int) and not isinstance(value, bool) for value in hashes),
        line_number,
        "terminal_block_hashes must contain integers",
    )
    return _RequestAccessRow(
        line_number=line_number,
        seq=_required_int(row, "seq", line_number),
        event_idx=_required_int(row, "event_idx", line_number),
        run_index=_required_int(row, "run_index", line_number),
        engine_id=cast(str, engine_id),
        data_parallel_rank=_required_int(row, "data_parallel_rank", line_number),
        request_seq=_required_int(row, "request_seq", line_number),
        lookup_performed=cast(bool, lookup_performed),
        group_count=_required_positive_int(row, "group_count", line_number),
        group_idx=_required_int(row, "group_idx", line_number),
        terminal_block_hashes=tuple(cast(list[int], hashes)),
    )


def _validate_residency_row(row: dict[str, Any], line_number: int) -> None:
    """Validate known collector residency rows before intentionally ignoring them."""
    kind = cast(str, row["kind"])
    expected_fields = _STORED_FIELDS if kind == "stored" else _RESIDENCY_FIELDS
    unexpected = sorted(set(row) - expected_fields)
    missing = sorted(expected_fields - set(row))
    _require_trace(
        not unexpected,
        line_number,
        f"unexpected or privacy-sensitive fields: {unexpected}",
    )
    _require_trace(not missing, line_number, f"missing fields: {missing}")
    _required_int(row, "seq", line_number)
    _required_int(row, "event_idx", line_number)
    _required_int(row, "run_index", line_number)
    timestamp = row.get("ts")
    _require_trace(
        isinstance(timestamp, int | float) and not isinstance(timestamp, bool),
        line_number,
        "ts must be numeric",
    )
    medium = row.get("medium")
    _require_trace(
        isinstance(medium, str) and bool(medium),
        line_number,
        "medium must be a non-empty string",
    )
    group_idx = row.get("group_idx")
    _require_trace(
        group_idx is None
        or (
            isinstance(group_idx, int)
            and not isinstance(group_idx, bool)
            and group_idx >= 0
        ),
        line_number,
        "group_idx must be null or a non-negative int",
    )
    raw_hashes = row.get("block_hashes")
    _require_trace(
        isinstance(raw_hashes, list), line_number, "block_hashes must be a list"
    )
    hashes = cast(list[Any], raw_hashes)
    _require_trace(
        all(isinstance(value, int) and not isinstance(value, bool) for value in hashes),
        line_number,
        "block_hashes must contain integers",
    )
    if kind == "stored":
        # Placeholder events for groups the tracker cannot fully describe use
        # block_size=0; it is a legitimate collector row, not missing data.
        _required_int(row, "block_size", line_number)
        _required_int(row, "num_tokens", line_number)


def _parse_jsonl(path: Path) -> list[_RequestAccessRow]:
    access_rows: list[_RequestAccessRow] = []
    with path.open(encoding="utf-8") as source:
        for line_number, raw in enumerate(source, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RequestAccessTraceError.at_line(
                    line_number, f"invalid JSON: {exc.msg}"
                ) from exc
            _require_trace(
                isinstance(row, dict), line_number, "row must be a JSON object"
            )
            row = cast(dict[str, Any], row)
            kind = row.get("kind")
            _require_trace(isinstance(kind, str), line_number, "kind must be a string")
            if kind in _REJECTED_TRACE_KINDS:
                raise RequestAccessTraceError.at_line(
                    line_number, f"trace contains forbidden {kind!r} boundary"
                )
            if kind in _IGNORED_RESIDENCY_KINDS:
                _validate_residency_row(row, line_number)
                continue
            _require_trace(
                kind == "request_access",
                line_number,
                f"unknown trace row kind: {kind!r}",
            )
            access_rows.append(_parse_request_access_row(row, line_number))
    if not access_rows:
        raise RequestAccessTraceError("trace contains no request_access rows")
    return access_rows


def _build_request_access_streams(
    rows: list[_RequestAccessRow],
) -> tuple[RequestAccessStream, ...]:
    rows_by_stream: dict[StreamIdentity, list[_RequestAccessRow]] = {}
    stream_order: list[StreamIdentity] = []
    for row in rows:
        if row.stream_identity not in rows_by_stream:
            rows_by_stream[row.stream_identity] = []
            stream_order.append(row.stream_identity)
        rows_by_stream[row.stream_identity].append(row)

    streams: list[RequestAccessStream] = []
    for identity in stream_order:
        unsorted_rows = rows_by_stream[identity]
        seen_positions: dict[tuple[int, int], int] = {}
        for row in unsorted_rows:
            previous_line = seen_positions.setdefault(
                row.publisher_position, row.line_number
            )
            _require_trace(
                previous_line == row.line_number,
                row.line_number,
                f"duplicate publisher position (first seen at line {previous_line})",
            )
        stream_rows = sorted(unsorted_rows, key=lambda row: row.publisher_position)
        group_count = stream_rows[0].group_count
        expected_groups = tuple(range(group_count))
        for row in stream_rows:
            _require_trace(
                row.group_count == group_count,
                row.line_number,
                "group_count must be consistent within one stream",
            )
            _require_trace(
                row.group_idx < group_count,
                row.line_number,
                "group_idx must be smaller than group_count",
            )
        _require_trace(
            stream_rows[0].request_seq == 0,
            stream_rows[0].line_number,
            "request_seq must start at zero for a complete stream",
        )
        requests: list[TraceRequest] = []
        request_groups: list[tuple[int, ...]] = []
        request_lines: list[int] = []
        current_seq: int | None = None
        current_line = 0
        current_lookup_performed = True
        current_groups: list[int] = []
        current_keys: list[tuple[int, int]] = []

        for row in stream_rows:
            if current_seq is None or row.request_seq != current_seq:
                if current_seq is not None:
                    _require_trace(
                        row.request_seq == current_seq + 1,
                        row.line_number,
                        "request_seq must be gap-free and must not reset",
                    )
                    request_groups.append(tuple(current_groups))
                    requests.append(
                        TraceRequest(
                            blocks=tuple(current_keys),
                            lookup_performed=current_lookup_performed,
                        )
                    )
                    request_lines.append(current_line)
                current_seq = row.request_seq
                current_line = row.line_number
                current_lookup_performed = row.lookup_performed
                current_groups = []
                current_keys = []
            _require_trace(
                row.lookup_performed == current_lookup_performed,
                row.line_number,
                "lookup_performed must agree across all groups of one request",
            )
            _require_trace(
                row.group_idx not in current_groups,
                row.line_number,
                "duplicate group_idx in one request",
            )
            current_groups.append(row.group_idx)
            current_keys.extend(
                (row.group_idx, terminal_hash)
                for terminal_hash in row.terminal_block_hashes
            )
        request_groups.append(tuple(current_groups))
        requests.append(
            TraceRequest(
                blocks=tuple(current_keys),
                lookup_performed=current_lookup_performed,
            )
        )
        request_lines.append(current_line)

        canonical_groups = request_groups[0]
        _require_trace(
            canonical_groups == expected_groups,
            stream_rows[0].line_number,
            "KV group indices must be complete, ordered, and zero-based",
        )
        for index, groups in enumerate(request_groups):
            _require_trace(
                groups == canonical_groups,
                request_lines[index],
                "request group set or group order is incomplete/inconsistent",
            )
        _require_trace(
            group_count == 1,
            stream_rows[0].line_number,
            "request-access replay supports only single-group traces",
        )
        streams.append(
            RequestAccessStream(
                identity=identity,
                group_indices=canonical_groups,
                requests=tuple(requests),
            )
        )
    return tuple(streams)


def load_request_access_trace(path: Path) -> RequestAccessTrace:
    """Load an observer JSONL trace, rejecting unsafe or incomplete input.

    Replayed frames may appear late in the file, so access rows are restored to
    publisher order by ``(run_index, seq, event_idx)``. Request sequences are
    then checked independently for each run/engine/DP stream.
    """
    return RequestAccessTrace(streams=_build_request_access_streams(_parse_jsonl(path)))


def _merge_result(target: SimResult, source: SimResult) -> None:
    target.accesses += source.accesses
    target.hits += source.hits
    target.stores += source.stores
    target.store_batches_refused += source.store_batches_refused
    target.store_batches_filtered += source.store_batches_filtered
    target.evictions += source.evictions
    target.stores_skipped += source.stores_skipped
    # Streams have independent managers; the sum is their concurrent footprint.
    target.admitted_blocks_peak += source.admitted_blocks_peak


def run_request_access_trace(trace: RequestAccessTrace, config: SimConfig) -> SimResult:
    """Replay each engine/DP stream through an independent manager."""
    result = SimResult()
    for stream in trace.streams:
        stream_result = run_trace(list(stream.requests), config)
        _merge_result(result, stream_result)
    return result


def report_request_access_trace(
    trace: RequestAccessTrace,
    configs: dict[str, SimConfig] | None = None,
) -> dict[str, Any]:
    """Build a hash-free LRU/ARC comparison for a validated live trace."""
    if configs is None:
        configs = {
            "lru+thr0": SimConfig(cache_policy="lru", store_threshold=0),
            "arc+thr0": SimConfig(cache_policy="arc", store_threshold=0),
            "lru+thr2": SimConfig(cache_policy="lru", store_threshold=2),
            "arc+thr2": SimConfig(cache_policy="arc", store_threshold=2),
        }
    report: dict[str, Any] = {
        "trace_contract": "request-access-v2",
        "streams": len(trace.streams),
        "requests": trace.request_count,
        "block_accesses": trace.block_access_count,
        "blocks_touched": trace.touched_block_count,
        "group_counts": sorted(len(stream.group_indices) for stream in trace.streams),
        "configs": {},
    }
    for label, config in configs.items():
        result = run_request_access_trace(trace, config)
        report["configs"][label] = {
            "hit_rate": round(result.hit_rate, 4),
            "hits": result.hits,
            "stores": result.stores,
            "evictions": result.evictions,
            "stores_skipped": result.stores_skipped,
            "store_batches_refused": result.store_batches_refused,
            "store_batches_filtered": result.store_batches_filtered,
            "admitted_blocks_peak_sum": result.admitted_blocks_peak,
        }
    return report


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
    parser.add_argument(
        "--request-access-jsonl",
        type=Path,
        help=(
            "Replay a fail-closed request_access observer trace instead of "
            "synthetic traces."
        ),
    )
    args = parser.parse_args()

    if args.request_access_jsonl:
        trace = load_request_access_trace(args.request_access_jsonl)
        report = {"observed_request_access": report_request_access_trace(trace)}
    else:
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
