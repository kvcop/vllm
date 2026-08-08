# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Payload-free confidence calibration capture for DSpark."""

from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


DEFAULT_CAPTURE_SHARD_ROWS = 4096

CAPTURE_SCHEMA = frozenset(
    {
        "raw_logits",
        "prefix_mask",
        "verified_lengths",
        "accepted_counts",
        "request_ordinal",
        "proposal_seq",
        "engine_step",
    }
)
CAPTURE_DTYPES = {
    "raw_logits": np.dtype(np.float32),
    "prefix_mask": np.dtype(np.uint8),
    "verified_lengths": np.dtype(np.uint8),
    "accepted_counts": np.dtype(np.uint8),
    "request_ordinal": np.dtype(np.uint64),
    "proposal_seq": np.dtype(np.uint32),
    "engine_step": np.dtype(np.uint64),
}


def build_prefix_mask(
    accepted_count: int, verified_length: int, num_speculative_tokens: int
) -> np.ndarray:
    """Build prefix-survival labels; positions after verification are censored."""
    if not 0 <= accepted_count <= verified_length <= num_speculative_tokens:
        raise ValueError(
            "Expected 0 <= accepted_count <= verified_length <= "
            f"num_speculative_tokens, got {accepted_count}, {verified_length}, "
            f"{num_speculative_tokens}."
        )
    result = np.zeros(num_speculative_tokens, dtype=np.uint8)
    result[:accepted_count] = 1
    return result


def is_capture_writer(
    *, is_last_pp_rank: bool, tp_rank: int, tp_world_size: int
) -> bool:
    """Elect TP rank zero on the last pipeline stage."""
    if tp_world_size <= 0 or not 0 <= tp_rank < tp_world_size:
        raise ValueError(
            f"Invalid tensor-parallel rank {tp_rank} for size {tp_world_size}."
        )
    return is_last_pp_rank and tp_rank == 0


def _as_numpy(value: np.ndarray | torch.Tensor, dtype: np.dtype) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(dtype, copy=False)
    return value.detach().to(device="cpu").numpy().astype(dtype, copy=False)


class DSparkCalibrationCollector:
    """Join DSpark proposals to next-step verification outcomes by slot."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        max_rows: int,
        num_speculative_tokens: int,
        max_num_slots: int,
        dp_rank: int,
        shard_rows: int = DEFAULT_CAPTURE_SHARD_ROWS,
    ) -> None:
        if max_rows <= 0:
            raise ValueError(f"max_rows must be positive, got {max_rows}.")
        if shard_rows <= 0:
            raise ValueError(f"shard_rows must be positive, got {shard_rows}.")
        if not 0 < num_speculative_tokens <= np.iinfo(np.uint8).max:
            raise ValueError(
                "num_speculative_tokens must fit a positive uint8 capture length."
            )
        if max_num_slots <= 0:
            raise ValueError("max_num_slots must be positive.")
        if dp_rank < 0:
            raise ValueError(f"dp_rank must be non-negative, got {dp_rank}.")

        self.max_rows = max_rows
        self.num_speculative_tokens = num_speculative_tokens
        self.max_num_slots = max_num_slots
        self.dp_rank = dp_rank
        self.shard_rows = shard_rows

        capture_root = Path(output_dir)
        capture_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        capture_root.chmod(0o700)
        self.output_dir = capture_root / f"dp-{dp_rank:04d}"
        self.output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.output_dir.chmod(0o700)

        self._current_request_ordinal = np.zeros(max_num_slots, dtype=np.uint64)
        self._pending_raw_logits = np.zeros(
            (max_num_slots, num_speculative_tokens), dtype=np.float32
        )
        self._pending_request_ordinal = np.zeros(max_num_slots, dtype=np.uint64)
        self._pending_proposal_seq = np.zeros(max_num_slots, dtype=np.uint32)
        self._pending_engine_step = np.zeros(max_num_slots, dtype=np.uint64)
        self._pending_valid = np.zeros(max_num_slots, dtype=np.bool_)

        (
            self._shard_index,
            self._total_rows,
            self._next_request_ordinal,
            self._next_proposal_seq,
            self._engine_step,
        ) = self._load_existing_capture_state()
        if self._total_rows > self.max_rows:
            raise ValueError(
                f"Existing capture has {self._total_rows} rows, above the configured "
                f"hard cap {self.max_rows}."
            )
        self._rows: list[tuple[np.ndarray, np.ndarray, int, int, int, int, int]] = []
        self._closed = False

    @property
    def total_rows(self) -> int:
        return self._total_rows

    @property
    def pending_valid(self) -> np.ndarray:
        """Return a copy for diagnostics and focused tests."""
        return self._pending_valid.copy()

    def _load_existing_capture_state(self) -> tuple[int, int, int, int, int]:
        prefix = f"capture-dp{self.dp_rank:04d}-shard"
        indices = []
        total_rows = 0
        # A collision-resistant random starting point keeps request ordinals
        # comparable across separately captured train/held-out runs while
        # preserving monotonicity within a run. It contains no request data.
        next_request_ordinal = secrets.randbits(62) + 1
        next_proposal_seq = 0
        engine_step = -1
        for path in sorted(self.output_dir.glob(f"{prefix}*.npz")):
            suffix = path.stem.removeprefix(prefix)
            if not suffix.isdigit():
                continue
            indices.append(int(suffix))
            with np.load(path, allow_pickle=False) as arrays:
                if frozenset(arrays.files) != CAPTURE_SCHEMA:
                    raise ValueError(f"Capture shard has an invalid schema: {path}.")
                rows = int(arrays["raw_logits"].shape[0])
                if any(
                    arrays[name].dtype != dtype
                    for name, dtype in CAPTURE_DTYPES.items()
                ):
                    raise ValueError(f"Capture shard has invalid dtypes: {path}.")
                if (
                    arrays["raw_logits"].shape != (rows, self.num_speculative_tokens)
                    or arrays["prefix_mask"].shape
                    != (rows, self.num_speculative_tokens)
                    or any(
                        arrays[name].shape != (rows,)
                        for name in CAPTURE_SCHEMA - {"raw_logits", "prefix_mask"}
                    )
                ):
                    raise ValueError(f"Capture shard has inconsistent rows: {path}.")
                total_rows += rows
                if rows:
                    next_request_ordinal = max(
                        next_request_ordinal,
                        int(arrays["request_ordinal"].max()) + 1,
                    )
                    next_proposal_seq = max(
                        next_proposal_seq,
                        int(arrays["proposal_seq"].max()) + 1,
                    )
                    engine_step = max(engine_step, int(arrays["engine_step"].max()))
        return (
            max(indices, default=-1) + 1,
            total_rows,
            next_request_ordinal,
            next_proposal_seq,
            engine_step,
        )

    def _validate_slots(self, slots: np.ndarray) -> None:
        if slots.ndim != 1:
            raise ValueError(
                f"slot indices must be one-dimensional, got {slots.shape}."
            )
        if np.any(slots < 0) or np.any(slots >= self.max_num_slots):
            raise ValueError(
                f"slot indices must be in [0, {self.max_num_slots}), got {slots}."
            )

    def add_request(self, slot: int) -> int:
        """Assign a fresh opaque ordinal and invalidate reused slot state."""
        if not 0 <= slot < self.max_num_slots:
            raise ValueError(f"slot must be in [0, {self.max_num_slots}), got {slot}.")
        if self._closed:
            raise RuntimeError("Cannot add a request to a closed collector.")
        if self._next_request_ordinal > np.iinfo(np.uint64).max:
            raise OverflowError("DSpark capture request ordinal exhausted uint64.")

        ordinal = self._next_request_ordinal
        self._next_request_ordinal += 1
        self._current_request_ordinal[slot] = ordinal
        self._pending_valid[slot] = False
        return ordinal

    def observe(
        self,
        *,
        idx_mapping: np.ndarray | torch.Tensor,
        cu_num_logits: np.ndarray | torch.Tensor,
        is_prefilling: np.ndarray,
        num_sampled: np.ndarray | torch.Tensor,
        num_rejected: np.ndarray | torch.Tensor,
        num_bonus_tokens: int,
    ) -> np.ndarray:
        """Consume pending proposals and return rows eligible for a new proposal."""
        if self._closed:
            raise RuntimeError("Cannot observe with a closed collector.")
        if num_bonus_tokens <= 0:
            raise ValueError(
                f"num_bonus_tokens must be positive, got {num_bonus_tokens}."
            )

        slots = _as_numpy(idx_mapping, np.int64)
        cu_logits = _as_numpy(cu_num_logits, np.int64)
        sampled = _as_numpy(num_sampled, np.int64)
        rejected = _as_numpy(num_rejected, np.int64)
        # This is deliberately conservative on the final prefill chunk: the
        # runner's prefill predicate can remain true for one step after the
        # sampler predicate becomes false. Dropping that proposal loses a row
        # but cannot create a mislabeled row.
        prefilling = np.asarray(is_prefilling, dtype=np.bool_)
        self._validate_slots(slots)
        num_reqs = slots.size
        if (
            cu_logits.shape != (num_reqs + 1,)
            or sampled.shape != (num_reqs,)
            or rejected.shape != (num_reqs,)
            or prefilling.shape != (num_reqs,)
        ):
            raise ValueError("Capture observation arrays do not describe one batch.")
        if np.any(np.diff(cu_logits) < 0):
            raise ValueError("cu_num_logits must be non-decreasing.")

        if self._engine_step >= np.iinfo(np.uint64).max:
            raise OverflowError("DSpark capture engine step exhausted uint64.")
        self._engine_step += 1
        present = np.zeros(self.max_num_slots, dtype=np.bool_)
        present[slots] = True
        self._pending_valid[~present] = False

        proposal_mask = (sampled > 0) & ~prefilling
        for row, slot_value in enumerate(slots):
            slot = int(slot_value)
            if not self._pending_valid[slot]:
                continue

            pending_ordinal = int(self._pending_request_ordinal[slot])
            self._pending_valid[slot] = False
            if pending_ordinal != int(self._current_request_ordinal[slot]):
                continue

            num_logits = int(cu_logits[row + 1] - cu_logits[row])
            verified_length = num_logits - num_bonus_tokens
            if (
                sampled[row] == 0
                or prefilling[row]
                or verified_length <= 0
                or verified_length > self.num_speculative_tokens
            ):
                continue

            accepted_count = int(
                np.clip(
                    sampled[row] - num_bonus_tokens,
                    0,
                    verified_length,
                )
            )
            if rejected[row] < 0 or accepted_count != verified_length - rejected[row]:
                continue

            self._append_row(
                raw_logits=self._pending_raw_logits[slot],
                prefix_mask=build_prefix_mask(
                    accepted_count,
                    verified_length,
                    self.num_speculative_tokens,
                ),
                verified_length=verified_length,
                accepted_count=accepted_count,
                request_ordinal=pending_ordinal,
                proposal_seq=int(self._pending_proposal_seq[slot]),
                engine_step=int(self._pending_engine_step[slot]),
            )

        return proposal_mask

    def record_proposal(
        self,
        *,
        raw_logits: np.ndarray | torch.Tensor,
        idx_mapping: np.ndarray | torch.Tensor,
        proposal_mask: np.ndarray,
    ) -> None:
        """Record raw confidence logits for the proposal just produced."""
        if self._closed:
            raise RuntimeError("Cannot record a proposal with a closed collector.")
        if self._total_rows >= self.max_rows:
            return

        slots = _as_numpy(idx_mapping, np.int64)
        logits = _as_numpy(raw_logits, np.float32)
        mask = np.asarray(proposal_mask, dtype=np.bool_)
        self._validate_slots(slots)
        expected_shape = (slots.size, self.num_speculative_tokens)
        if logits.shape != expected_shape or mask.shape != (slots.size,):
            raise ValueError(
                f"Expected raw logits {expected_shape} and mask {(slots.size,)}, got "
                f"{logits.shape} and {mask.shape}."
            )

        for row, slot_value in enumerate(slots):
            if not mask[row]:
                continue
            if self._next_proposal_seq > np.iinfo(np.uint32).max:
                raise OverflowError(
                    "DSpark capture proposal sequence exhausted uint32."
                )
            slot = int(slot_value)
            ordinal = int(self._current_request_ordinal[slot])
            if ordinal == 0:
                continue

            self._pending_raw_logits[slot] = logits[row]
            self._pending_request_ordinal[slot] = ordinal
            self._pending_proposal_seq[slot] = self._next_proposal_seq
            self._pending_engine_step[slot] = self._engine_step
            self._pending_valid[slot] = True
            self._next_proposal_seq += 1

    def _append_row(
        self,
        *,
        raw_logits: np.ndarray,
        prefix_mask: np.ndarray,
        verified_length: int,
        accepted_count: int,
        request_ordinal: int,
        proposal_seq: int,
        engine_step: int,
    ) -> None:
        if self._total_rows >= self.max_rows:
            return
        self._rows.append(
            (
                raw_logits.astype(np.float32, copy=True),
                prefix_mask.astype(np.uint8, copy=True),
                verified_length,
                accepted_count,
                request_ordinal,
                proposal_seq,
                engine_step,
            )
        )
        self._total_rows += 1
        if len(self._rows) >= self.shard_rows or self._total_rows >= self.max_rows:
            self.flush()
        if self._total_rows >= self.max_rows:
            self._pending_valid.fill(False)

    def flush(self) -> None:
        """Atomically persist the current fixed-schema shard."""
        if not self._rows:
            return
        rows = self._rows
        arrays = {
            "raw_logits": np.stack([row[0] for row in rows]).astype(
                np.float32, copy=False
            ),
            "prefix_mask": np.stack([row[1] for row in rows]).astype(
                np.uint8, copy=False
            ),
            "verified_lengths": np.asarray([row[2] for row in rows], dtype=np.uint8),
            "accepted_counts": np.asarray([row[3] for row in rows], dtype=np.uint8),
            "request_ordinal": np.asarray([row[4] for row in rows], dtype=np.uint64),
            "proposal_seq": np.asarray([row[5] for row in rows], dtype=np.uint32),
            "engine_step": np.asarray([row[6] for row in rows], dtype=np.uint64),
        }
        assert frozenset(arrays) == CAPTURE_SCHEMA

        filename = f"capture-dp{self.dp_rank:04d}-shard{self._shard_index:06d}.npz"
        final_path = self.output_dir / filename
        temp_path = final_path.with_suffix(".npz.tmp")
        if final_path.exists() or temp_path.exists():
            raise FileExistsError(f"Refusing to overwrite capture shard {final_path}.")

        fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "wb") as file:
                np.savez(file, **arrays)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temp_path, final_path)
            final_path.chmod(0o600)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise

        self._rows = []
        self._shard_index += 1

    def close(self) -> None:
        """Flush pending rows and reject subsequent use."""
        if self._closed:
            return
        self.flush()
        self._pending_valid.fill(False)
        self._closed = True
