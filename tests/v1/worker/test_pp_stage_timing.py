# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the opt-in per-stage pipeline-parallel timing.

The instrumentation exists to separate "this stage genuinely computes more"
from "this stage is early in the pipeline and simply waits less", so the two
properties worth pinning on CPU are (a) it is a real no-op when
`VLLM_PP_STAGE_TIMING` is unset, and (b) the aggregation it reports when on
is arithmetically what the log line claims.
"""

import pytest

from vllm.v1.worker import gpu_worker, pp_stage_timing
from vllm.v1.worker.gpu_worker import AsyncIntermediateTensors
from vllm.v1.worker.pp_stage_timing import (
    ALL_REGIONS,
    NULL_PP_STAGE_TIMER,
    PPStageTimer,
    maybe_make_pp_stage_timer,
)

pytestmark = [
    pytest.mark.cpu_test,
    pytest.mark.skip_global_cleanup,
]


class _FakeHandle:
    def __init__(self) -> None:
        self.waited = False

    def wait(self) -> None:
        self.waited = True


def _host_timer(**kwargs) -> PPStageTimer:
    """A timer forced onto the host clock so it runs without an accelerator."""
    kwargs.setdefault("log_interval", 100)
    return PPStageTimer(0, 2, use_cuda_events=False, **kwargs)


def _fake_clock(ticks: list[float]):
    """A deterministic clock yielding the given values, in seconds."""
    it = iter(ticks)
    return lambda: next(it)


# --------------------------------------------------------------------------
# Disabled path: nothing is measured, nothing is allocated.
# --------------------------------------------------------------------------


def test_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_PP_STAGE_TIMING", raising=False)
    assert maybe_make_pp_stage_timer(0, 2) is NULL_PP_STAGE_TIMER
    assert maybe_make_pp_stage_timer(1, 2) is NULL_PP_STAGE_TIMER


def test_disabled_when_pp_is_not_in_use(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_PP_STAGE_TIMING", "1")
    assert maybe_make_pp_stage_timer(0, 1) is NULL_PP_STAGE_TIMER


def test_enabled_flag_builds_a_real_timer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_PP_STAGE_TIMING", "1")
    monkeypatch.setenv("VLLM_PP_STAGE_TIMING_INTERVAL", "7")
    timer = maybe_make_pp_stage_timer(1, 2)
    assert isinstance(timer, PPStageTimer)
    assert (timer.pp_rank, timer.pp_world_size, timer.log_interval) == (1, 2, 7)


def test_null_timer_regions_allocate_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The disabled regions must not build events, contexts or samples."""

    def _explode(*args, **kwargs):
        raise AssertionError("the disabled timer must not create CUDA events")

    monkeypatch.setattr(pp_stage_timing.torch.cuda, "Event", _explode)

    # All four entry points hand back the same preallocated context object.
    contexts = [
        NULL_PP_STAGE_TIMER.host_region("recv_post"),
        NULL_PP_STAGE_TIMER.device_region("exec"),
        NULL_PP_STAGE_TIMER.step(),
    ]
    assert all(ctx is pp_stage_timing._NULL_CTX for ctx in contexts)

    with (
        NULL_PP_STAGE_TIMER.step(),
        NULL_PP_STAGE_TIMER.device_region("exec"),
        NULL_PP_STAGE_TIMER.host_region("recv_post"),
    ):
        pass
    assert NULL_PP_STAGE_TIMER.end_step() is None
    assert not NULL_PP_STAGE_TIMER.enabled


def test_async_intermediate_tensors_defaults_to_the_null_timer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The relay's default construction must stay uninstrumented."""

    def _explode(*args, **kwargs):
        raise AssertionError("the disabled timer must not create CUDA events")

    monkeypatch.setattr(pp_stage_timing.torch.cuda, "Event", _explode)

    handle = _FakeHandle()
    tensors = AsyncIntermediateTensors({}, comm_handles=[handle])
    assert tensors._timer is NULL_PP_STAGE_TIMER
    tensors.wait_for_comm()
    assert handle.waited


def test_worker_starts_with_the_null_timer() -> None:
    """`Worker.__init__` must not leave the attribute unset for `execute_model`."""
    assert gpu_worker.NULL_PP_STAGE_TIMER is NULL_PP_STAGE_TIMER


# --------------------------------------------------------------------------
# Enabled path: the numbers are what the log line says they are.
# --------------------------------------------------------------------------


def test_host_region_records_milliseconds() -> None:
    timer = _host_timer(clock=_fake_clock([1.0, 1.004, 10.0, 10.006]))
    with timer.host_region("recv_post"):
        pass
    with timer.host_region("recv_post"):
        pass

    summary = timer.summary()["recv_post"]
    assert summary["n"] == 2
    assert summary["mean"] == pytest.approx(5.0)
    assert summary["max"] == pytest.approx(6.0)


def test_device_region_without_cuda_uses_the_host_clock() -> None:
    """No accelerator means no CUDA-event clock; the log must say so."""
    timer = _host_timer(clock=_fake_clock([0.0, 0.002]))
    assert timer.device_clock_name == "host"
    with timer.device_region("exec"):
        pass
    assert timer.summary()["exec"]["mean"] == pytest.approx(2.0)


def test_recv_wait_is_recorded_through_the_relay() -> None:
    """`wait_for_comm` is the only place the recv stall can be observed."""
    timer = _host_timer(clock=_fake_clock([0.0, 0.003]))
    handle = _FakeHandle()
    postprocessed: list[str] = []
    tensors = AsyncIntermediateTensors(
        {},
        comm_handles=[handle],
        comm_postprocess=[lambda: postprocessed.append("done")],
        timer=timer,
    )

    # Touching `.tensors` triggers the lazy wait, exactly as the model does.
    assert tensors.tensors == {}
    assert handle.waited
    assert postprocessed == ["done"]
    assert timer.summary()["recv_wait"]["mean"] == pytest.approx(3.0)

    # The wait is idempotent and must not record a second sample.
    tensors.wait_for_comm()
    assert timer.summary()["recv_wait"]["n"] == 1


def test_compute_is_exec_minus_recv_wait_plus_sample() -> None:
    """The whole point of the split: position-independent per-stage compute.

    `exec` contains the recv stall on a non-first stage, and `sample` holds
    the sampling and speculative drafting that only the last stage runs and
    that `execute_model` never sees.
    """
    timer = _host_timer(
        log_interval=1,
        # exec spans 20 ms, of which 8 ms is the recv stall nested inside it;
        # sampling adds a further 5 ms outside execute_model.
        clock=_fake_clock([0.0, 0.001, 0.009, 0.020, 0.020, 0.025]),
    )
    with timer.device_region("exec"), timer.device_region("recv_wait"):
        pass
    with timer.device_region("sample"):
        pass
    line = timer.flush()
    assert line is not None
    assert "exec=20.00" in line
    assert "recv_wait=8.00" in line
    assert "sample=5.00" in line
    assert "compute=17.00" in line
    assert "rank=0/2" in line
    assert "clock=host" in line


def test_flush_happens_on_the_configured_cadence(caplog) -> None:
    timer = _host_timer(log_interval=3, clock=_fake_clock([0.0, 0.001] * 3))
    held = []
    for _ in range(3):
        with timer.step(), timer.device_region("exec"):
            pass
        held.append(len(timer._samples.get("exec") or ()))

    # Steps 1 and 2 accumulate; step 3 flushes and clears the accumulator.
    assert held == [1, 2, 0]
    assert timer._steps == 3


def test_flush_is_a_noop_without_samples() -> None:
    assert _host_timer().flush() is None


def test_regions_absent_on_this_stage_are_omitted() -> None:
    """Rank 0 never receives and the last rank never sends."""
    timer = _host_timer(clock=_fake_clock([0.0, 0.001]))
    with timer.device_region("exec"):
        pass
    summary = timer.summary()
    assert set(summary) == {"exec"}
    line = timer.flush()
    assert line is not None
    for absent in set(ALL_REGIONS) - {"exec"}:
        assert f"{absent}=" not in line


def test_event_pairs_are_pooled() -> None:
    """Long runs must not leak one event pair per step."""

    class _FakeEvent:
        created = 0

        def __init__(self, enable_timing: bool = False) -> None:
            type(self).created += 1
            self.enable_timing = enable_timing

        def record(self) -> None:
            pass

        def synchronize(self) -> None:
            pass

        def elapsed_time(self, other) -> float:
            return 1.0

    timer = PPStageTimer(0, 2, log_interval=2, use_cuda_events=True)
    original = pp_stage_timing.torch.cuda.Event
    pp_stage_timing.torch.cuda.Event = _FakeEvent  # type: ignore[misc]
    try:
        for _ in range(6):
            with timer.step(), timer.device_region("exec"):
                pass
    finally:
        pp_stage_timing.torch.cuda.Event = original  # type: ignore[misc]

    # One pair per unresolved step, so the pool fills to `log_interval` pairs
    # (2 steps x 2 events) and every later step reuses them.
    assert _FakeEvent.created == 4
    assert len(timer._event_pool) == 2
    assert timer._steps == 6


def test_invalid_interval_is_rejected() -> None:
    with pytest.raises(ValueError):
        PPStageTimer(0, 2, log_interval=0, use_cuda_events=False)
