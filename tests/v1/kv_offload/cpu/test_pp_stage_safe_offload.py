# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage-safety tests for the shared CPU offload region under pipeline
parallelism, where per-stage KV group views are unequal (Qwen3.8 hybrid
projection: (8,8,8,8) on stage 0 vs (8,8,8,9) on stage 1).

The region-level tests simulate the unequal views directly with per-rank
slot-width tables; the spec-level tests verify that CPUOffloadingSpec sizes
one shared layout from the registered table, refuses to attach on any
mismatch, and stays byte-compatible for uniform (TP-only) tables.
"""

import contextlib
import mmap
import os
import threading
import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest

from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingGroupConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.factory import OffloadingSpecFactory

PAGE_SIZE = mmap.PAGESIZE


def _cleanup_file(path: str) -> None:
    """Best-effort file removal for test teardown."""
    with contextlib.suppress(FileNotFoundError):
        os.unlink(path)


@pytest.fixture
def iid():
    """Fresh instance ID for each test."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Region-level: heterogeneous per-stage slot layout
# ---------------------------------------------------------------------------

# Two PP stages x two TP ranks. Stage 1 carries one extra full-attention
# layer, so its per-rank chunk width is wider than stage 0's. The widths are
# deliberately page-unaligned in sum, exercising the row-tail padding.
STAGE0_WIDTH = 2 * PAGE_SIZE
STAGE1_WIDTH = 2 * PAGE_SIZE + PAGE_SIZE // 2 + 64
HETERO_SLOTS = [STAGE0_WIDTH, STAGE0_WIDTH, STAGE1_WIDTH, STAGE1_WIDTH]
HETERO_ROW_STRIDE = -(-sum(HETERO_SLOTS) // PAGE_SIZE) * PAGE_SIZE  # round_up
ROW_PADDING = HETERO_ROW_STRIDE - sum(HETERO_SLOTS)
assert ROW_PADDING > 0
NUM_BLOCKS = 3


def _make_hetero_regions(iid: str) -> list[SharedOffloadRegion]:
    """One region per rank, all sharing the registered per-rank slot table."""
    return [
        SharedOffloadRegion(
            engine_id=iid,
            num_blocks=NUM_BLOCKS,
            rank=rank,
            kv_bytes_per_block=HETERO_ROW_STRIDE,
            cpu_page_size=HETERO_SLOTS[rank],
            slot_page_sizes=HETERO_SLOTS,
        )
        for rank in range(4)
    ]


def _slot_bounds(rank: int) -> tuple[int, int]:
    start = sum(HETERO_SLOTS[:rank])
    return start, start + HETERO_SLOTS[rank]


@pytest.fixture
def hetero_regions(iid):
    regions = _make_hetero_regions(iid)
    try:
        yield regions
    finally:
        for r in regions:
            r.cleanup()
        _cleanup_file(regions[0].mmap_path)


def test_hetero_layout_row_stride_and_total(iid):
    regions = _make_hetero_regions(iid)
    try:
        for r in regions:
            assert r.row_stride == HETERO_ROW_STRIDE
            assert r.total_size_bytes == NUM_BLOCKS * HETERO_ROW_STRIDE
        assert len({r._creator for r in regions}) == 2  # exactly one creator
        assert os.path.getsize(regions[0].mmap_path) == NUM_BLOCKS * HETERO_ROW_STRIDE
    finally:
        for r in regions:
            r.cleanup()
        _cleanup_file(regions[0].mmap_path)


def test_hetero_stage_slot_round_trip_no_cross_stage_aliasing(hetero_regions):
    """Each rank stores its own bytes for every block row and reads them back
    unchanged; no rank's write ever lands inside another rank's slot."""
    views = []
    for rank, r in enumerate(hetero_regions):
        # Two canonical tensors per rank, together filling that rank's slot.
        half = HETERO_SLOTS[rank] // 2
        views.append((r.create_next_view(half), r.create_next_view(half)))

    for rank, (ta, tb) in enumerate(views):
        ta[:, :] = 10 + rank
        tb[:, :] = 20 + rank

    raw = memoryview(hetero_regions[0].mmap_obj)
    for blk in range(NUM_BLOCKS):
        row = blk * HETERO_ROW_STRIDE
        for rank in range(4):
            start, end = _slot_bounds(rank)
            slot = bytes(raw[row + start : row + end])
            half = HETERO_SLOTS[rank] // 2
            assert all(b == 10 + rank for b in slot[:half]), (
                f"block {blk} rank {rank}: first half corrupted"
            )
            assert all(b == 20 + rank for b in slot[half:]), (
                f"block {blk} rank {rank}: second half corrupted"
            )
        # Row tail padding must stay untouched between the last slot and the
        # next row (HETERO_ROW_STRIDE is the padded sum of slot widths).
        pad = bytes(raw[row + sum(HETERO_SLOTS) : row + HETERO_ROW_STRIDE])
        assert len(pad) == ROW_PADDING
        assert all(b == 0 for b in pad)

    # Load side: each stage re-reads exactly its own slots for one block.
    for rank, (ta, tb) in enumerate(views):
        assert all(b == 10 + rank for b in bytes(ta[1].tolist()))
        assert all(b == 20 + rank for b in bytes(tb[1].tolist()))
    del raw, views


def test_hetero_uniform_table_matches_legacy_byte_layout(iid):
    """A table of equal widths must reproduce the legacy uniform layout:
    same stride, same slot offsets, same total size."""
    legacy = SharedOffloadRegion(
        engine_id=iid,
        num_blocks=2,
        rank=1,
        kv_bytes_per_block=4 * PAGE_SIZE,
        cpu_page_size=PAGE_SIZE,
    )
    try:
        assert legacy.row_stride == 4 * PAGE_SIZE
        assert legacy._worker_offset == PAGE_SIZE
        assert legacy._worker_area_end == 2 * PAGE_SIZE
    finally:
        legacy.cleanup()
        _cleanup_file(legacy.mmap_path)

    tabled = SharedOffloadRegion(
        engine_id=iid,
        num_blocks=2,
        rank=1,
        kv_bytes_per_block=4 * PAGE_SIZE,
        cpu_page_size=PAGE_SIZE,
        slot_page_sizes=[PAGE_SIZE] * 4,
    )
    try:
        assert tabled.row_stride == 4 * PAGE_SIZE
        assert tabled._worker_offset == PAGE_SIZE
        assert tabled._worker_area_end == 2 * PAGE_SIZE
        assert tabled.total_size_bytes == 2 * 4 * PAGE_SIZE
    finally:
        tabled.cleanup()
        _cleanup_file(tabled.mmap_path)


def test_region_refuses_rank_outside_registered_table(iid):
    with pytest.raises(AssertionError, match="no slot in the registered"):
        SharedOffloadRegion(
            engine_id=iid,
            num_blocks=1,
            rank=4,
            kv_bytes_per_block=HETERO_ROW_STRIDE,
            cpu_page_size=STAGE0_WIDTH,
            slot_page_sizes=HETERO_SLOTS,
        )
    _cleanup_file(f"/dev/shm/vllm_offload_{iid}.mmap")


def test_region_refuses_stale_or_foreign_sized_file(iid):
    """A leftover file whose size matches a different layout must be refused
    instead of mapped with a foreign stride (silent cross-stage aliasing)."""
    path = f"/dev/shm/vllm_offload_{iid}.mmap"
    stale_size = NUM_BLOCKS * HETERO_ROW_STRIDE + PAGE_SIZE
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    os.ftruncate(fd, stale_size)
    os.close(fd)
    try:
        with pytest.raises(RuntimeError, match="refusing to attach"):
            SharedOffloadRegion(
                engine_id=iid,
                num_blocks=NUM_BLOCKS,
                rank=0,
                kv_bytes_per_block=HETERO_ROW_STRIDE,
                cpu_page_size=STAGE0_WIDTH,
                slot_page_sizes=HETERO_SLOTS,
            )
    finally:
        _cleanup_file(path)


def test_region_refuses_peer_with_smaller_registered_table(iid):
    """A worker holding a different (smaller) table than the creator must
    fail closed: the opener sees a file larger than its layout and refuses
    to attach instead of mapping a foreign stride."""
    regions = _make_hetero_regions(iid)
    try:
        with pytest.raises(RuntimeError, match="refusing to attach"):
            SharedOffloadRegion(
                engine_id=iid,
                num_blocks=NUM_BLOCKS,
                rank=0,
                kv_bytes_per_block=4 * PAGE_SIZE,
                cpu_page_size=PAGE_SIZE,
                slot_page_sizes=[PAGE_SIZE] * 4,
            )
    finally:
        for r in regions:
            r.cleanup()
        _cleanup_file(regions[0].mmap_path)


def test_region_with_larger_registered_table_times_out(iid, monkeypatch):
    """A worker whose table demands more bytes than the creator truncated
    cannot attach either: it waits for the file to grow and times out —
    the second documented failure mode of unequal shared views."""
    import vllm.v1.kv_offload.cpu.shared_offload_region as region_module

    def _fail_fast(fd: int, expected_size: int, timeout: float = 30.0) -> None:
        raise TimeoutError(f"mmap file never reached {expected_size} bytes")

    regions = _make_hetero_regions(iid)
    # Patch only after the base regions exist, so their own rendezvous uses
    # the real size wait.
    monkeypatch.setattr(region_module, "_wait_for_file_size", _fail_fast)
    try:
        with pytest.raises(TimeoutError):
            SharedOffloadRegion(
                engine_id=iid,
                num_blocks=NUM_BLOCKS,
                rank=0,
                # One extra page per rank: a different (larger) registration.
                kv_bytes_per_block=HETERO_ROW_STRIDE + 4 * PAGE_SIZE,
                cpu_page_size=STAGE0_WIDTH + PAGE_SIZE,
                slot_page_sizes=[w + PAGE_SIZE for w in HETERO_SLOTS],
            )
    finally:
        for r in regions:
            r.cleanup()
        _cleanup_file(regions[0].mmap_path)


# ---------------------------------------------------------------------------
# Region lifecycle: unlink-after-rendezvous (#52596 semantics) per region
# ---------------------------------------------------------------------------


def test_barrier_unlinks_path_after_every_worker_mapped(iid):
    """With a barrier, the creator unlinks the path once all workers mapped;
    mappings stay usable and shared afterwards."""
    path = f"/dev/shm/vllm_offload_{iid}.mmap"
    rendezvous = threading.Barrier(4)
    regions: list[SharedOffloadRegion | None] = [None] * 4

    def make(rank: int) -> None:
        regions[rank] = SharedOffloadRegion(
            engine_id=iid,
            num_blocks=2,
            rank=rank,
            kv_bytes_per_block=HETERO_ROW_STRIDE,
            cpu_page_size=HETERO_SLOTS[rank],
            slot_page_sizes=HETERO_SLOTS,
            barrier=rendezvous.wait,
        )

    threads = [threading.Thread(target=make, args=(r,)) for r in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    try:
        assert not os.path.exists(path), "path must be unlinked after rendezvous"
        views = [
            r.create_next_view(HETERO_SLOTS[i])
            for i, r in enumerate(regions)  # type: ignore[union-attr]
        ]
        # MAP_SHARED survives the unlink: writes stay visible across ranks.
        views[0][:, :] = 0x5A
        raw = memoryview(regions[0].mmap_obj)  # type: ignore[union-attr]
        assert raw[0] == 0x5A
        rank1_start = HETERO_SLOTS[0]
        rank1_end = rank1_start + HETERO_SLOTS[1]
        assert all(b == 0x00 for b in raw[rank1_start:rank1_end])
        del raw, views
    finally:
        for r in regions:
            if r is not None:
                r.cleanup()


def test_barrier_failure_releases_region_and_file(iid):
    """If the rendezvous barrier raises, the worker closes its mapping and
    the creator removes the file instead of leaving it behind."""
    path = f"/dev/shm/vllm_offload_{iid}.mmap"

    def failing_barrier() -> None:
        raise RuntimeError("rendezvous lost a worker")

    with pytest.raises(RuntimeError, match="rendezvous lost a worker"):
        SharedOffloadRegion(
            engine_id=iid,
            num_blocks=2,
            rank=0,
            kv_bytes_per_block=HETERO_ROW_STRIDE,
            cpu_page_size=STAGE0_WIDTH,
            slot_page_sizes=HETERO_SLOTS,
            barrier=failing_barrier,
        )
    # The failing worker closed its mapping and the creator removed the file.
    assert not os.path.exists(path)
    _cleanup_file(path)


def test_barrier_failure_opener_leaves_creators_file(iid):
    """A non-creator whose barrier fails must not unlink the path: only the
    creator owns the unlink, so late/open peers keep a consistent file."""
    path = f"/dev/shm/vllm_offload_{iid}.mmap"

    def failing_barrier() -> None:
        raise RuntimeError("rendezvous lost a worker")

    creator = SharedOffloadRegion(
        engine_id=iid,
        num_blocks=2,
        rank=0,
        kv_bytes_per_block=HETERO_ROW_STRIDE,
        cpu_page_size=STAGE0_WIDTH,
        slot_page_sizes=HETERO_SLOTS,
    )
    try:
        with pytest.raises(RuntimeError, match="rendezvous lost a worker"):
            SharedOffloadRegion(
                engine_id=iid,
                num_blocks=2,
                rank=1,
                kv_bytes_per_block=HETERO_ROW_STRIDE,
                cpu_page_size=STAGE0_WIDTH,
                slot_page_sizes=HETERO_SLOTS,
                barrier=failing_barrier,
            )
        assert os.path.exists(path), "non-creator must not remove the file"
    finally:
        creator.cleanup()
        _cleanup_file(path)


# ---------------------------------------------------------------------------
# Spec-level: sizing, refusal and byte-compatibility
# ---------------------------------------------------------------------------


def _make_config(
    *,
    rank: int = 0,
    worker_kv_bytes_per_block: int,
    by_rank: tuple[int, ...] | None,
    world_size: int = 4,
    tp_size: int = 2,
    pp_size: int = 2,
    cpu_bytes_to_use: int = 27 * PAGE_SIZE,
) -> OffloadingConfig:
    return OffloadingConfig(
        groups=(OffloadingGroupConfig(16, ("layer",)),),
        worker_kv_bytes_per_block=worker_kv_bytes_per_block,
        enable_kv_cache_events=False,
        extra_config={"cpu_bytes_to_use": cpu_bytes_to_use},
        engine_id="pp-stage-safe-test",
        model=OffloadingModelConfig(name="test-model", dtype="float16"),
        cache=OffloadingCacheConfig(tokens_per_hash=16, blocks_per_chunk=1),
        parallel=OffloadingParallelConfig(
            rank=rank,
            world_size=world_size,
            tp_size=tp_size,
            pp_size=pp_size,
            pcp_size=1,
            dcp_size=1,
            data_parallel_index=0,
            is_parallelism_agnostic=False,
            worker_kv_bytes_per_block_by_rank=by_rank,
        ),
    )


def _create_spec(**kwargs: Any) -> CPUOffloadingSpec:
    spec = OffloadingSpecFactory.create_spec(_make_config(**kwargs))
    assert isinstance(spec, CPUOffloadingSpec)
    return spec


def _stage_table() -> tuple[int, ...]:
    stage0 = PAGE_SIZE
    stage1 = PAGE_SIZE + PAGE_SIZE // 2  # one extra full-attention layer
    return (stage0, stage0, stage1, stage1)


def test_spec_sizes_shared_tier_from_registered_table_for_all_ranks():
    table = _stage_table()
    row = sum(table)
    specs = [
        _create_spec(rank=r, worker_kv_bytes_per_block=table[r], by_rank=table)
        for r in range(4)
    ]
    for r, spec in enumerate(specs):
        assert spec.num_blocks == (27 * PAGE_SIZE) // row
        assert spec.kv_bytes_per_chunk == row
        assert spec.cpu_page_size_per_worker == table[r]
    # The scheduler (rank 0) and every worker agree on one block-ID capacity,
    # so the manager can never allocate an ID that overflows another stage.
    assert len({spec.num_blocks for spec in specs}) == 1


def test_spec_refuses_local_view_drift_against_registered_slot():
    table = _stage_table()
    with pytest.raises(Exception, match="registered per-worker layout slot"):
        _create_spec(rank=2, worker_kv_bytes_per_block=table[0], by_rank=table)


def test_spec_refuses_pp_without_registered_table():
    with pytest.raises(Exception, match="registered per-worker KV layout"):
        _create_spec(rank=0, worker_kv_bytes_per_block=PAGE_SIZE, by_rank=None)


def test_spec_create_worker_passes_slot_table_only_for_hetero_layouts(monkeypatch):
    import vllm.v1.kv_offload.cpu.spec as cpu_spec_module

    monkeypatch.setattr(cpu_spec_module.current_platform, "is_cuda_alike", lambda: True)
    region_calls: list[dict[str, Any]] = []

    def fake_region_ctor(**kwargs):
        region_calls.append(kwargs)
        return MagicMock()

    monkeypatch.setattr(cpu_spec_module, "SharedOffloadRegion", fake_region_ctor)
    monkeypatch.setattr(cpu_spec_module, "CPUOffloadingWorker", MagicMock())

    # Unequal stage views: the explicit per-rank slot table must reach the
    # region, and every rank must agree on num_blocks and the table itself.
    table = _stage_table()
    for rank in (0, 3):
        monkeypatch.setattr(
            cpu_spec_module.torch.accelerator,
            "current_device_index",
            lambda rank=rank: rank,
        )
        spec = _create_spec(
            rank=rank, worker_kv_bytes_per_block=table[rank], by_rank=table
        )
        spec.create_worker(MagicMock())
    assert len(region_calls) == 2
    for call in region_calls:
        assert call["slot_page_sizes"] == list(table)
        assert call["num_blocks"] == (27 * PAGE_SIZE) // sum(table)
        assert call["barrier"] is cpu_spec_module._all_workers_barrier
    assert region_calls[0]["rank"] == 0
    assert region_calls[1]["rank"] == 3

    # Uniform table (TP-only): byte-compatible legacy form, no slot table.
    region_calls.clear()
    monkeypatch.setattr(
        cpu_spec_module.torch.accelerator, "current_device_index", lambda: 7
    )
    spec = _create_spec(
        rank=3,
        worker_kv_bytes_per_block=PAGE_SIZE,
        by_rank=(PAGE_SIZE,) * 4,
        world_size=4,
    )
    spec.create_worker(MagicMock())
    (call,) = region_calls
    assert call["slot_page_sizes"] is None
    assert call["rank"] == 3  # 7 % 4
    assert call["kv_bytes_per_block"] == 4 * PAGE_SIZE
    assert call["cpu_page_size"] == PAGE_SIZE


def test_spec_create_worker_refuses_device_slot_mismatch(monkeypatch):
    """The device-derived slot must match the registered layout too: if the
    local view belongs to another stage's slot, attach is refused."""
    import vllm.v1.kv_offload.cpu.spec as cpu_spec_module

    monkeypatch.setattr(cpu_spec_module.current_platform, "is_cuda_alike", lambda: True)
    monkeypatch.setattr(cpu_spec_module, "SharedOffloadRegion", MagicMock())
    monkeypatch.setattr(cpu_spec_module, "CPUOffloadingWorker", MagicMock())
    # Registered rank 0 is a stage-0 slot, but the device index says slot 0
    # while the local bytes are stage-1's: mismatch must refuse.
    monkeypatch.setattr(
        cpu_spec_module.torch.accelerator, "current_device_index", lambda: 0
    )
    table = _stage_table()
    spec = _create_spec(rank=3, worker_kv_bytes_per_block=table[3], by_rank=table)
    with pytest.raises(Exception, match="refusing to attach"):
        spec.create_worker(MagicMock())


# ---------------------------------------------------------------------------
# Connector config boundary: registered-layout validation
# ---------------------------------------------------------------------------


def _kv_cache_config(num_blocks: int) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[],
    )


def _parallel_config(tp: int, pp: int):
    # Real ParallelConfig validates against physical GPUs; the resolver only
    # reads the parallel sizes, so a plain namespace keeps the test CPU-only.
    from types import SimpleNamespace

    return SimpleNamespace(
        world_size=tp * pp,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
    )


class _FakeTensor:
    def __init__(self, size: int):
        self.size = size


def _config_with_bytes(cfg: KVCacheConfig, total: int, blocks: int) -> KVCacheConfig:
    cfg.num_blocks = blocks
    cfg.kv_cache_tensors = [_FakeTensor(total)]  # type: ignore[assignment]
    return cfg


def test_resolve_registered_worker_kv_bytes_validates_shape():
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
        _resolve_registered_worker_kv_bytes,
    )
    from vllm.v1.core.kv_cache_utils import compute_per_worker_kv_bytes_per_block

    def make(total: int) -> KVCacheConfig:
        return _config_with_bytes(_kv_cache_config(8), total, 8)

    # PP2 x TP2: stage 0 stores 2 pages per block, stage 1 stores 2.5.
    configs = [make(16 * PAGE_SIZE)] * 2 + [make(20 * PAGE_SIZE)] * 2
    parallel = _parallel_config(tp=2, pp=2)
    stage1_width = 20 * PAGE_SIZE // 8
    # Production flow: the engine core computes and registers the table
    # before any consumer resolves it.
    registered = compute_per_worker_kv_bytes_per_block(configs)
    assert registered == [
        2 * PAGE_SIZE,
        2 * PAGE_SIZE,
        stage1_width,
        stage1_width,
    ]
    configs[0].per_worker_kv_bytes_per_block = registered
    resolved = _resolve_registered_worker_kv_bytes(configs[0], parallel)
    assert resolved == (
        2 * PAGE_SIZE,
        2 * PAGE_SIZE,
        stage1_width,
        stage1_width,
    )

    # Missing registration under PP fails closed.
    empty = _kv_cache_config(8)
    with pytest.raises(ValueError, match="requires the registered"):
        _resolve_registered_worker_kv_bytes(empty, parallel)

    # Wrong entry count fails closed.
    short = _kv_cache_config(8)
    short.per_worker_kv_bytes_per_block = [2 * PAGE_SIZE] * 3
    with pytest.raises(ValueError, match="entries"):
        _resolve_registered_worker_kv_bytes(short, parallel)

    # TP peers inside one stage disagreeing fails closed.
    drifting = _kv_cache_config(8)
    drifting.per_worker_kv_bytes_per_block = [2 * PAGE_SIZE, 3 * PAGE_SIZE] * 2
    with pytest.raises(ValueError, match="inside PP stage 0"):
        _resolve_registered_worker_kv_bytes(drifting, parallel)

    # TP-only without a registration stays on the legacy path (None).
    tp_only = _parallel_config(tp=4, pp=1)
    assert _resolve_registered_worker_kv_bytes(empty, tp_only) is None


# ---------------------------------------------------------------------------
# Engine-core attach: per-worker footprints land on every config
# ---------------------------------------------------------------------------


def test_compute_per_worker_kv_bytes_per_block():
    from vllm.v1.core.kv_cache_utils import (
        compute_per_worker_kv_bytes_per_block,
        generate_scheduler_kv_cache_config,
    )

    def make(total: int) -> KVCacheConfig:
        return _config_with_bytes(_kv_cache_config(8), total, 8)

    # Homogeneous (TP-only): None, so the previous config shape is kept.
    assert compute_per_worker_kv_bytes_per_block([make(16 * PAGE_SIZE)] * 4) is None
    # Single worker: None.
    assert compute_per_worker_kv_bytes_per_block([make(16 * PAGE_SIZE)]) is None

    stage0 = [make(16 * PAGE_SIZE) for _ in range(2)]
    stage1 = [make(20 * PAGE_SIZE) for _ in range(2)]
    configs = stage0 + stage1
    per_worker = compute_per_worker_kv_bytes_per_block(configs)
    stage1_width = 20 * PAGE_SIZE // 8
    assert per_worker == [
        2 * PAGE_SIZE,
        2 * PAGE_SIZE,
        stage1_width,
        stage1_width,
    ]

    # get_kv_cache_configs registers the table on every worker's config; the
    # scheduler view is a deepcopy of worker 0 and must carry it too.
    for cfg in configs:
        cfg.per_worker_kv_bytes_per_block = per_worker
    scheduler_cfg = generate_scheduler_kv_cache_config(configs)
    assert scheduler_cfg.per_worker_kv_bytes_per_block == per_worker
    assert stage0[0].per_worker_kv_bytes_per_block == per_worker
    assert stage1[1].per_worker_kv_bytes_per_block == per_worker
