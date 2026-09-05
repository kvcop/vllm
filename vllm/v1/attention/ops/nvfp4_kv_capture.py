# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional pre-cache K/V snapshot capture for the NVFP4-KV quality gate.

Environment-gated debug instrument for the E361 real-activation matrix. When
``VLLM_NVFP4KV_CAPTURE_DIR`` is set and the attention layer runs an
unquantized KV cache (``auto``/``bfloat16``), the Triton backend saves the
pre-cache key and value tensors of every attention layer to::

    <dir>/rank<r>/<layer_name>/call<i>.pt

Each file holds ``{"key", "value", "num_kv_heads", "head_dim", "kv_cache_dtype"}``
with bf16 CPU tensors of shape ``[num_tokens, num_kv_heads, head_dim]``.

Bounds: the first ``MAX_CALLS_PER_LAYER`` store calls per layer, or
``MAX_TOTAL_BYTES`` of saved tensor payload across the process, whichever
comes first. The hook is completely inert when the variable is unset (one
module-level flag check per call) and never raises into the serving path:
any failure prints one warning and disables the capture for the process.

The hook is CUDA-graph-capture aware (stand blocker, 05.09): during graph
capture it is a counted no-op (a ``Tensor.cpu()`` inside capture aborts the
capture with "Cannot copy between CPU and CUDA tensors"), and a capture-time
failure never disables the capture — only ``CONSECUTIVE_ERROR_LIMIT``
consecutive failures outside graph capture do. Snapshots are staged as
device-side clones and flushed to disk by the barrier's caller only when no
capture is in flight, so the flush ``Tensor.cpu()`` cannot abort a capture.
Each file additionally carries ``step`` (a monotonic per-process counter)
and ``wall_time_ns`` so the acceptance can count files written after API
readiness instead of any preexisting warmup file.
"""

from __future__ import annotations

import os
import time
import warnings
from collections.abc import Callable
from pathlib import Path

import torch

CAPTURE_ENV_VAR = "VLLM_NVFP4KV_CAPTURE_DIR"
MAX_CALLS_PER_LAYER = 64
MAX_TOTAL_BYTES = 4 * (1 << 30)
# Stand blocker 05.09: a .pt flush during CUDA graph capture aborts the
# capture; only repeated failures OUTSIDE capture may disable the hook.
CONSECUTIVE_ERROR_LIMIT = 8
MAX_PENDING_ITEMS = 4


class _PendingSnapshot:
    """One device-side staged K/V clone awaiting a non-capture flush."""

    __slots__ = ("layer_name", "key", "value", "call_index", "step")

    def __init__(
        self, layer_name: str, key: torch.Tensor, value: torch.Tensor, call_index: int, step: int
    ) -> None:
        self.layer_name = layer_name
        self.key = key
        self.value = value
        self.call_index = call_index
        self.step = step


class _CaptureState:
    """Process-local capture bookkeeping; one instance per TP worker."""

    __slots__ = (
        "root",
        "rank",
        "calls_by_layer",
        "total_bytes",
        "disabled",
        "pending",
        "step_counter",
        "capture_skips",
        "pending_drops",
        "consecutive_errors",
    )

    def __init__(self, root: Path, rank: int) -> None:
        self.root = root
        self.rank = rank
        self.calls_by_layer: dict[str, int] = {}
        self.total_bytes = 0
        self.disabled = False
        self.pending: list[_PendingSnapshot] = []
        self.step_counter = 0
        self.capture_skips = 0
        self.pending_drops = 0
        self.consecutive_errors = 0


def _is_cuda_graph_capturing(is_capturing: Callable[[], bool] | None = None) -> bool:
    """True while a CUDA graph capture is in flight on the current stream.

    Injectable for tests; the production check never raises into serving.
    """
    probe = is_capturing or torch.cuda.is_current_stream_capturing
    try:
        return bool(probe())
    except Exception:  # noqa: BLE001 - diagnostic instrument only.
        return False


_STATE: _CaptureState | None = None
_STATE_INITIALIZED = False


def _warn_safely(message: str) -> None:
    """Warns without ever raising.

    Under ``PYTHONWARNINGS=error`` (or a warnings-as-errors filter),
    ``warnings.warn`` itself raises; a diagnostic instrument must not turn
    that into a serving-path exception.
    """
    try:
        warnings.warn(message, stacklevel=3)
    except Exception:  # noqa: BLE001 - must never break serving.
        pass


def _resolve_rank() -> int:
    """Returns the tensor-parallel rank of this worker process.

    Real TP workers initialize the groups before model load, so the primary
    API is authoritative there. A dev/CPU harness without an initialized
    process group falls back to rank 0 instead of disabling the capture.
    """
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        return get_tensor_model_parallel_rank()
    except Exception:  # noqa: BLE001 - diagnostic instrument only.
        return 0


def _get_state() -> _CaptureState | None:
    """Lazily initializes the capture from the environment, once per process."""
    global _STATE, _STATE_INITIALIZED
    if _STATE_INITIALIZED:
        return _STATE
    _STATE_INITIALIZED = True
    raw = os.environ.get(CAPTURE_ENV_VAR)
    if not raw:
        return None
    try:
        state = _CaptureState(Path(raw), _resolve_rank())
        state.root.mkdir(parents=True, exist_ok=True)
        _STATE = state
    except Exception as error:  # noqa: BLE001 - must never break serving.
        _warn_safely(
            f"{CAPTURE_ENV_VAR} is set but the K/V capture cannot start: "
            f"{error!r}; the capture stays disabled."
        )
    return _STATE


def _stage_call(
    state: _CaptureState, layer_name: str, key: torch.Tensor, value: torch.Tensor
) -> None:
    """Stages one bounded snapshot as a device clone; the flush writes it.

    Never raises and never touches host memory: a ``Tensor.cpu()`` here would
    abort an in-flight CUDA graph capture (stand blocker, 05.09).
    """
    try:
        call_index = state.calls_by_layer.get(layer_name, 0)
        if call_index >= MAX_CALLS_PER_LAYER:
            return
        payload_bytes = key.numel() * key.element_size() + value.numel() * value.element_size()
        if state.total_bytes + payload_bytes > MAX_TOTAL_BYTES:
            state.disabled = True
            return
        if len(state.pending) >= MAX_PENDING_ITEMS:
            state.pending_drops += 1
            return
        state.step_counter += 1
        state.pending.append(
            _PendingSnapshot(
                layer_name,
                key.detach().clone(),
                value.detach().clone(),
                call_index,
                state.step_counter,
            )
        )
    except Exception as error:  # noqa: BLE001 - must never break serving.
        _count_failure(state, layer_name, error)


def _count_failure(state: _CaptureState, layer_name: str, error: BaseException) -> None:
    """Capture-time failures are counted, not fatal; only repeated
    non-capture failures disable the hook."""
    if _is_cuda_graph_capturing():
        state.capture_skips += 1
        return
    state.consecutive_errors += 1
    if state.consecutive_errors >= CONSECUTIVE_ERROR_LIMIT:
        state.disabled = True
        _warn_safely(
            f"NVFP4-KV K/V capture failed on layer {layer_name!r} "
            f"{state.consecutive_errors} times in a row outside CUDA graph "
            f"capture and is now disabled (last error: {error!r})"
        )


def _flush_pending(state: _CaptureState) -> None:
    """Writes staged snapshots to disk; called only when not capturing."""
    while state.pending:
        snapshot = state.pending.pop(0)
        layer_name = snapshot.layer_name
        call_index = snapshot.call_index
        try:
            key_cpu = snapshot.key.to(device="cpu", dtype=torch.bfloat16)
            value_cpu = snapshot.value.to(device="cpu", dtype=torch.bfloat16)
            layer_dir = state.root / f"rank{state.rank}" / layer_name
            layer_dir.mkdir(parents=True, exist_ok=True)
            target = layer_dir / f"call{call_index}.pt"
            # Atomic write (05.09 attempt 3): readers (stand preflight) must
            # never observe a half-written .pt. Payload goes to <name>.pt.tmp,
            # fsync makes the bytes durable, os.replace swaps atomically.
            tmp_path = layer_dir / f"call{call_index}.pt.tmp"
            try:
                with tmp_path.open("wb") as handle:
                    torch.save(
                        {
                            "key": key_cpu,
                            "value": value_cpu,
                            "num_kv_heads": snapshot.key.shape[1],
                            "head_dim": snapshot.key.shape[2],
                            "kv_cache_dtype": "bfloat16",
                            "step": snapshot.step,
                            "wall_time_ns": time.time_ns(),
                        },
                        handle,
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp_path, target)
            finally:
                # A crashed flush may leave the .tmp behind; acceptance
                # ignores *.tmp, so unlinking is best-effort hygiene only.
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
        except Exception as error:  # noqa: BLE001 - must never break serving.
            _count_failure(state, layer_name, error)
            continue
        state.consecutive_errors = 0
        state.calls_by_layer[layer_name] = call_index + 1
        state.total_bytes += (
            snapshot.key.numel() * snapshot.key.element_size()
            + snapshot.value.numel() * snapshot.value.element_size()
        )


def capture_kv_snapshot(
    layer_name: str | None, key: torch.Tensor, value: torch.Tensor
) -> None:
    """Stages one pre-cache K/V pair when the env gate is open.

    Called from the Triton backend's cache-update path. Cheap no-op unless
    ``VLLM_NVFP4KV_CAPTURE_DIR`` is set in the process environment, and a
    COUNTED no-op while a CUDA graph capture is in flight: the flush
    (``capture_kv_flush``) is what touches host memory, and only when no
    capture is running.
    """
    state = _get_state()
    if state is None or state.disabled:
        return
    if _is_cuda_graph_capturing():
        state.capture_skips += 1
        return
    if key.ndim != 3 or value.ndim != 3 or key.shape != value.shape:
        # Unexpected projection shapes are not activation evidence.
        return
    _stage_call(state, layer_name or "unknown-layer", key, value)


def capture_kv_flush() -> None:
    """Flushes staged snapshots to disk; a no-op during graph capture.

    Called by the Triton backend's capture barrier right after
    ``capture_kv_snapshot``; the stage/flush split keeps every host copy out
    of CUDA graph capture windows.
    """
    state = _get_state()
    if state is None or state.disabled:
        return
    if _is_cuda_graph_capturing():
        return
    _flush_pending(state)
