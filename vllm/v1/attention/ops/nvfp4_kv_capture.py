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

Saving synchronizes the stream (``Tensor.cpu()``); use this only on a
dedicated capture arm, not on a serving profile.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import torch

CAPTURE_ENV_VAR = "VLLM_NVFP4KV_CAPTURE_DIR"
MAX_CALLS_PER_LAYER = 64
MAX_TOTAL_BYTES = 4 * (1 << 30)


class _CaptureState:
    """Process-local capture bookkeeping; one instance per TP worker."""

    __slots__ = ("root", "rank", "calls_by_layer", "total_bytes", "disabled")

    def __init__(self, root: Path, rank: int) -> None:
        self.root = root
        self.rank = rank
        self.calls_by_layer: dict[str, int] = {}
        self.total_bytes = 0
        self.disabled = False


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


def _save_call(
    state: _CaptureState, layer_name: str, key: torch.Tensor, value: torch.Tensor
) -> None:
    """Writes one bounded bf16 snapshot for the layer, never raising."""
    call_index = state.calls_by_layer.get(layer_name, 0)
    if call_index >= MAX_CALLS_PER_LAYER:
        return
    payload_bytes = key.numel() * key.element_size() + value.numel() * value.element_size()
    if state.total_bytes + payload_bytes > MAX_TOTAL_BYTES:
        state.disabled = True
        return
    try:
        key_cpu = key.detach().to(device="cpu", dtype=torch.bfloat16)
        value_cpu = value.detach().to(device="cpu", dtype=torch.bfloat16)
        layer_dir = state.root / f"rank{state.rank}" / layer_name
        layer_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "key": key_cpu,
                "value": value_cpu,
                "num_kv_heads": key.shape[1],
                "head_dim": key.shape[2],
                "kv_cache_dtype": "bfloat16",
            },
            layer_dir / f"call{call_index}.pt",
        )
    except Exception as error:  # noqa: BLE001 - must never break serving.
        state.disabled = True
        _warn_safely(
            f"NVFP4-KV K/V capture failed on layer {layer_name!r} "
            f"(call {call_index}) and is now disabled: {error!r}"
        )
        return
    state.calls_by_layer[layer_name] = call_index + 1
    state.total_bytes += payload_bytes


def capture_kv_snapshot(
    layer_name: str | None, key: torch.Tensor, value: torch.Tensor
) -> None:
    """Captures one pre-cache K/V pair when the env gate is open.

    Called from the Triton backend's cache-update path. Cheap no-op unless
    ``VLLM_NVFP4KV_CAPTURE_DIR`` is set in the process environment.
    """
    state = _get_state()
    if state is None or state.disabled:
        return
    if key.ndim != 3 or value.ndim != 3 or key.shape != value.shape:
        # Unexpected projection shapes are not activation evidence.
        return
    _save_call(state, layer_name or "unknown-layer", key, value)
