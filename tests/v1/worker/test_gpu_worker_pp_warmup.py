# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the PP communication warmup in Worker.determine_available_memory.

The PP relay exercises the PP device group's NCCL communicator for the first
time inside the first execute_model, which runs after KV cache sizing. The
warmup must run inside determine_available_memory, before the memory-
profiling window, so the communicator's GPU buffers land inside the profiled
non-KV memory instead of silently eating into the KV cache budget.
"""

from types import SimpleNamespace

import pytest

from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.worker import gpu_worker
from vllm.v1.worker.gpu_worker import Worker

pytestmark = [
    pytest.mark.cpu_test,
    pytest.mark.skip_global_cleanup,
]


class _FakePPGroup:
    def __init__(self, world_size: int, rank_in_group: int) -> None:
        self.world_size = world_size
        self.rank_in_group = rank_in_group
        self.calls: list[tuple[str, int]] = []

    def send_tensor_dict(self, tensor_dict, dst=None, **kwargs) -> None:
        assert dst is not None
        self.calls.append(("send", dst))

    def recv_tensor_dict(self, src=None, **kwargs) -> None:
        assert src is not None
        self.calls.append(("recv", src))


def _make_worker(kv_cache_memory_bytes: int | None = None) -> Worker:
    """The minimal Worker surface determine_available_memory touches."""
    worker = object.__new__(Worker)
    worker.vllm_config = SimpleNamespace()
    worker.parallel_config = SimpleNamespace(_api_process_count=1)
    worker.cache_config = SimpleNamespace(
        kv_cache_memory_bytes=kv_cache_memory_bytes,
        gpu_memory_utilization=0.9,
    )
    worker.model_config = SimpleNamespace(multimodal_config=None)
    worker.init_snapshot = SimpleNamespace(free_memory=10 * GiB_bytes)
    worker.requested_memory = 9 * GiB_bytes
    worker.device = "cpu"
    return worker


def _patch_profiling_collaborators(monkeypatch, events: list[str]) -> None:
    profile_result = SimpleNamespace(
        after_profile=SimpleNamespace(free_memory=8 * GiB_bytes),
        total_consumed=1 * GiB_bytes,
        transient_peak_headroom=GiB_bytes // 2,
        non_kv_cache_memory=2 * GiB_bytes,
    )

    class _FakeWindow:
        def __enter__(self):
            events.append("profile_window")
            return profile_result

        def __exit__(self, *args) -> None:
            return None

    monkeypatch.setattr(
        gpu_worker,
        "memory_profiling",
        lambda baseline_snapshot, weights_memory=0: _FakeWindow(),
    )
    monkeypatch.setattr(
        gpu_worker,
        "maybe_apply_startup_plan",
        lambda worker: events.append("startup_plan"),
    )
    monkeypatch.setattr(
        gpu_worker,
        "reserve_mm_ipc_gpu_memory",
        lambda memory, *args, **kwargs: memory,
    )
    monkeypatch.setattr(
        gpu_worker,
        "current_platform",
        SimpleNamespace(is_cuda_alike=lambda: False),
    )


def _make_runner(worker: Worker, events: list[str]) -> None:
    worker.model_runner = SimpleNamespace(
        model_memory_usage=1 * GiB_bytes,
        profile_run=lambda: events.append("profile_run"),
    )


def test_warmup_precedes_memory_profiling_window(monkeypatch) -> None:
    """The PP warmup must run before the profiled window opens, so its
    allocations are part of the measured non-KV memory."""
    events: list[str] = []
    pp_group = _FakePPGroup(world_size=2, rank_in_group=0)
    monkeypatch.setattr(gpu_worker, "get_pp_group", lambda: pp_group)
    _patch_profiling_collaborators(monkeypatch, events)
    worker = _make_worker()
    _make_runner(worker, events)

    def recording_warmup() -> None:
        events.append("pp_warmup")
        Worker._warmup_pp_communication(worker)

    worker._warmup_pp_communication = recording_warmup  # type: ignore[method-assign]

    available = Worker.determine_available_memory(worker)

    assert available == 9 * GiB_bytes - 2 * GiB_bytes
    assert events == [
        "startup_plan",
        "pp_warmup",
        "profile_window",
        "profile_run",
    ]
    assert pp_group.calls == [("send", 1)]


def test_warmup_skipped_with_explicit_kv_cache_memory(monkeypatch) -> None:
    """Manual KV sizing never profiles memory, so it never warms up either."""
    events: list[str] = []
    pp_group = _FakePPGroup(world_size=2, rank_in_group=0)
    monkeypatch.setattr(gpu_worker, "get_pp_group", lambda: pp_group)
    _patch_profiling_collaborators(monkeypatch, events)
    worker = _make_worker(kv_cache_memory_bytes=5 * GiB_bytes)
    _make_runner(worker, events)

    available = Worker.determine_available_memory(worker)

    assert available == 5 * GiB_bytes
    assert events == ["startup_plan", "profile_run"]
    assert pp_group.calls == []


@pytest.mark.parametrize(
    ("rank_in_group", "expected_calls"),
    [
        (0, [("send", 1)]),
        (1, [("recv", 0), ("send", 2)]),
        (2, [("recv", 1)]),
    ],
)
def test_warmup_pairs_each_stage_boundary(
    monkeypatch, rank_in_group: int, expected_calls: list[tuple[str, int]]
) -> None:
    """Every adjacent PP stage pair exchanges one tensor, stage by stage."""
    pp_group = _FakePPGroup(world_size=3, rank_in_group=rank_in_group)
    monkeypatch.setattr(gpu_worker, "get_pp_group", lambda: pp_group)
    worker = _make_worker()

    Worker._warmup_pp_communication(worker)

    assert pp_group.calls == expected_calls


def test_warmup_noop_for_single_stage(monkeypatch) -> None:
    pp_group = _FakePPGroup(world_size=1, rank_in_group=0)
    monkeypatch.setattr(gpu_worker, "get_pp_group", lambda: pp_group)
    worker = _make_worker()

    Worker._warmup_pp_communication(worker)

    assert pp_group.calls == []
