# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in per-stage timing for the pipeline-parallel relay.

Enabled with ``VLLM_PP_STAGE_TIMING=1``; every stage then logs, once per
``VLLM_PP_STAGE_TIMING_INTERVAL`` steps, how its step time splits between
model compute, the inter-stage receive and the inter-stage send. The point
is to tell "this stage genuinely computes more" apart from "this stage sits
early in the pipeline and simply waits less", which raw GPU-busy percentages
cannot distinguish.

Two clocks are used, deliberately, because the relay has two different kinds
of region:

* **Host clock** (``time.perf_counter``) for regions that block the Python
  thread. ``GroupCoordinator.isend_tensor_dict`` /
  ``irecv_tensor_dict`` open with ``send_object`` / ``recv_object``, which are
  synchronous ``torch.distributed`` calls on the *CPU* (gloo) group. Those
  really do stall the host, so wall clock is the correct instrument and CUDA
  events would not see them at all.

* **CUDA events** for regions whose work is asynchronous on the device. A
  wall-clock timer around ``model_runner.execute_model`` or around a NCCL
  handle ``wait()`` measures kernel *launch* time, not execution: both only
  enqueue onto a stream and return. Recording ``torch.cuda.Event`` pairs on
  the current stream instead measures the device timeline, and because
  ``Handle.wait()`` for NCCL inserts a real stream dependency, the gap
  between the two events includes the stall while the peer's data lands.
  Event pairs are resolved in batches at flush time, so no per-step
  ``synchronize()`` is injected into the hot path.

When the flag is unset the worker holds :data:`NULL_PP_STAGE_TIMER`, whose
regions are a single shared ``nullcontext`` and whose ``end_step`` is a
no-op: no events are created, no samples are stored and nothing is logged.
"""

import os
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext

import torch

from vllm.logger import init_logger
from vllm.utils.gpu_sync_debug import gpu_sync_allowed

logger = init_logger(__name__)

# Regions recorded per step, in the order they occur inside
# `Worker.execute_model`. Kept explicit so the log line has a stable shape
# even for stages that never record some of them (rank 0 never receives,
# the last rank never sends).
HOST_REGIONS = ("recv_post", "send_post")
DEVICE_REGIONS = ("prev_send_drain", "exec", "recv_wait", "sample")
ALL_REGIONS = (
    "exec",
    "recv_wait",
    "sample",
    "recv_post",
    "send_post",
    "prev_send_drain",
)

_NULL_CTX = nullcontext()


def _percentile(sorted_values: list[float], q: float) -> float:
    """Nearest-rank percentile of an already sorted, non-empty list."""
    idx = int(round(q * (len(sorted_values) - 1)))
    return sorted_values[idx]


class _NullStageTimer:
    """The disabled timer. Every entry point is a constant-time no-op."""

    enabled = False

    def host_region(self, name: str) -> nullcontext[None]:
        return _NULL_CTX

    def device_region(self, name: str) -> nullcontext[None]:
        return _NULL_CTX

    def step(self) -> nullcontext[None]:
        return _NULL_CTX

    def end_step(self) -> None:
        return None


NULL_PP_STAGE_TIMER = _NullStageTimer()


class PPStageTimer:
    """Collects per-step relay timings for one pipeline stage."""

    enabled = True

    def __init__(
        self,
        pp_rank: int,
        pp_world_size: int,
        *,
        log_interval: int = 100,
        use_cuda_events: bool = True,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if log_interval < 1:
            raise ValueError(f"log_interval must be >= 1, got {log_interval}")
        self.pp_rank = pp_rank
        self.pp_world_size = pp_world_size
        self.log_interval = log_interval
        # Fall back to the host clock only where no CUDA-event clock exists
        # (CPU-only builds and unit tests). The log line names the clock that
        # was actually used so a host-clocked sample can never be mistaken for
        # a valid measurement of an async device region.
        self.use_cuda_events = use_cuda_events
        self._clock = clock
        self._samples: dict[str, list[float]] = defaultdict(list)
        self._pending: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self._event_pool: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._steps = 0

    @property
    def device_clock_name(self) -> str:
        return "cuda_event" if self.use_cuda_events else "host"

    @contextmanager
    def host_region(self, name: str) -> Iterator[None]:
        """Time a region that blocks the calling thread."""
        start = self._clock()
        try:
            yield
        finally:
            self._samples[name].append((self._clock() - start) * 1000.0)

    @contextmanager
    def device_region(self, name: str) -> Iterator[None]:
        """Time a region whose work is asynchronous on the device."""
        if not self.use_cuda_events:
            with self.host_region(name):
                yield
            return
        start, end = self._acquire_event_pair()
        start.record()
        try:
            yield
        finally:
            end.record()
            self._pending.append((name, start, end))

    @contextmanager
    def step(self) -> Iterator[None]:
        """Wrap one ``execute_model`` call, flushing on the configured cadence."""
        try:
            yield
        finally:
            self.end_step()

    def end_step(self) -> None:
        self._steps += 1
        if self._steps % self.log_interval == 0:
            self.flush()

    def _acquire_event_pair(self) -> tuple[torch.cuda.Event, torch.cuda.Event]:
        if self._event_pool:
            return self._event_pool.pop()
        return (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )

    def _resolve_pending(self) -> None:
        """Read back the device timeline for the buffered event pairs.

        Called only at flush time. The oldest pairs are long complete by then,
        so ``synchronize()`` costs approximately nothing; doing it here rather
        than per region keeps the measured steps free of injected syncs.
        """
        # This sync is the deliberate cost of the flush; tell VLLM_GPU_SYNC_CHECK
        # so enabling both diagnostics does not raise a false alarm.
        with gpu_sync_allowed():
            for name, start, end in self._pending:
                end.synchronize()
                self._samples[name].append(start.elapsed_time(end))
                self._event_pool.append((start, end))
        self._pending.clear()

    def summary(self) -> dict[str, dict[str, float]]:
        """Mean/p50/p90/max per region over the samples collected so far."""
        self._resolve_pending()
        out: dict[str, dict[str, float]] = {}
        for name in ALL_REGIONS:
            values = self._samples.get(name)
            if not values:
                continue
            ordered = sorted(values)
            out[name] = {
                "n": float(len(ordered)),
                "mean": sum(ordered) / len(ordered),
                "p50": _percentile(ordered, 0.5),
                "p90": _percentile(ordered, 0.9),
                "max": ordered[-1],
            }
        return out

    def flush(self) -> str | None:
        """Log one summary line for this stage and reset the accumulators."""
        summary = self.summary()
        if not summary:
            return None
        parts = [
            f"{name}={summary[name]['mean']:.2f}/{summary[name]['p90']:.2f}"
            for name in ALL_REGIONS
            if name in summary
        ]
        # `exec` brackets the whole model_runner call, and on a non-first stage
        # the lazy `wait_for_comm` happens inside it. Subtracting the recv wait
        # is what separates real compute from pipeline position; `sample` is
        # added back because sampling and drafting are last-rank-only device
        # work that `execute_model` never sees.
        exec_mean = summary.get("exec", {}).get("mean", 0.0)
        recv_wait_mean = summary.get("recv_wait", {}).get("mean", 0.0)
        sample_mean = summary.get("sample", {}).get("mean", 0.0)
        parts.append(f"compute={exec_mean - recv_wait_mean + sample_mean:.2f}")
        line = (
            f"PP stage timing rank={self.pp_rank}/{self.pp_world_size} "
            f"steps={self._steps} clock={self.device_clock_name} "
            f"ms(mean/p90): {' '.join(parts)}"
        )
        logger.info(line)
        self._samples.clear()
        return line


# What call sites annotate: either the live timer or the shared no-op.
PPStageTimerLike = PPStageTimer | _NullStageTimer


def maybe_make_pp_stage_timer(
    pp_rank: int,
    pp_world_size: int,
) -> PPStageTimerLike:
    """Return a live timer when the flag is on and PP is actually in use."""
    if not bool(int(os.getenv("VLLM_PP_STAGE_TIMING", "0"))) or pp_world_size <= 1:
        return NULL_PP_STAGE_TIMER
    timer = PPStageTimer(
        pp_rank,
        pp_world_size,
        log_interval=int(os.getenv("VLLM_PP_STAGE_TIMING_INTERVAL", "100")),
        use_cuda_events=torch.cuda.is_available(),
    )
    logger.info(
        "PP stage timing enabled for rank %d/%d (interval=%d steps, clock=%s)",
        pp_rank,
        pp_world_size,
        timer.log_interval,
        timer.device_clock_name,
    )
    return timer
