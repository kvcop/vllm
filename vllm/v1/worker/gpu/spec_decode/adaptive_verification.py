# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adaptive verification for DSpark speculative decoding."""

from collections import defaultdict
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

import numpy as np
import torch

import vllm.envs as envs
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu.async_utils import StepTimingSample, stream
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu

logger = init_logger(__name__)
_PROFILE_REPLAYS = 5

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.states import RequestState


def get_adaptive_verification_cudagraph_mode(
    configured_mode: CUDAGraphMode | None,
    attention_support: AttentionCGSupport,
    enabled: bool,
    enforce_eager: bool = False,
) -> CUDAGraphMode | None:
    """Choose a CUDA graph mode that is safe for variable draft lengths."""
    if not enabled:
        return configured_mode
    if enforce_eager or configured_mode in (None, CUDAGraphMode.NONE):
        raise ValueError(
            "Adaptive verification requires startup cost profiling and cannot "
            "run with --enforce-eager or cudagraph_mode=NONE. Use PIECEWISE or "
            "a FULL CUDA graph mode."
        )
    if attention_support == AttentionCGSupport.ALWAYS:
        return CUDAGraphMode.FULL_AND_PIECEWISE
    return CUDAGraphMode.PIECEWISE


def get_adaptive_speculator_cudagraph_mode(
    target_mode: CUDAGraphMode,
    enabled: bool,
) -> CUDAGraphMode:
    """Keep the draft eligible for FULL graphs when the target falls back."""
    if enabled:
        return CUDAGraphMode.FULL_AND_PIECEWISE
    return target_mode


def build_exact_adaptive_layout(
    scheduled_tokens: np.ndarray,
    scheduled_drafts: np.ndarray,
    admitted_drafts: np.ndarray,
    num_bonus_tokens: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build exact per-request query lengths and cumulative logit offsets."""
    if not (scheduled_tokens.shape == scheduled_drafts.shape == admitted_drafts.shape):
        raise ValueError("Adaptive layout arrays must have the same shape.")
    if np.any(admitted_drafts < 0) or np.any(admitted_drafts > scheduled_drafts):
        raise ValueError("Adaptive draft capacities exceed the scheduled drafts.")
    exact_scheduled_tokens = (
        scheduled_tokens - scheduled_drafts + admitted_drafts
    ).astype(np.int32, copy=False)
    cu_num_logits = np.empty(len(admitted_drafts) + 1, dtype=np.int32)
    cu_num_logits[0] = 0
    np.cumsum(admitted_drafts + num_bonus_tokens, out=cu_num_logits[1:])
    return exact_scheduled_tokens, cu_num_logits


def _assign_draft_token_budget(
    confidence_probs: torch.Tensor,
    idx_mapping: torch.Tensor,
    capacities: torch.Tensor,
    draft_budget: int,
    num_steps: int,
) -> None:
    """Admit the globally best ``draft_budget`` current-confidence slots.

    Every (request, step) slot is scored by its survival probability, the running
    product of that request's per-position confidences, and the highest scores win.
    Survival only decreases along a request, so a global top-k always admits
    continuously along steps with a request.

    Stable step-major ordering makes exact ties prefer a shallower step, then
    the existing request order. Budget sizing intentionally uses the t-2 CPU
    snapshot, while this allocation uses the current GPU confidences.
    """
    survival = confidence_probs[idx_mapping].cumprod(dim=1)
    steps = torch.arange(num_steps, device=survival.device)
    # Out-of-range slots score -inf so they never outrank a real draft.
    out_of_range = steps[None, :] >= capacities[:, None]
    survival = survival.masked_fill(out_of_range, -float("inf"))
    flat = survival.transpose(0, 1).flatten()
    winners = torch.argsort(flat, descending=True, stable=True)[:draft_budget]
    admitted = torch.zeros_like(flat, dtype=torch.bool).index_fill_(0, winners, True)
    torch.sum(
        admitted.view(num_steps, -1),
        dim=0,
        dtype=capacities.dtype,
        out=capacities,
    )


_assign_draft_token_budget_compiled = torch.compile(
    _assign_draft_token_budget, dynamic=True
)


def build_cost_tables_from_curves(
    draft_curve: list[tuple[int, float]],
    verify_curve: list[tuple[int, float]],
    max_num_reqs: int,
    max_batch_tokens: int,
    cudagraph_limit: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build cost tables: graph-padded below the capture limit, smooth above.

    Args:
        cudagraph_limit: Largest cudagraph-captured size. At or below it,
            execution pads up to the next captured size, so cost is a step
            function. Above it there is no padding, so cost is continuous.
    """

    def build_table(limit: int, curve: list[tuple[int, float]]) -> np.ndarray:
        xs, ys = np.asarray(curve, dtype=np.float64).T
        if not np.isfinite(xs).all() or not np.isfinite(ys).all():
            raise ValueError("DSpark cost curves must contain only finite values.")
        ys = np.maximum.accumulate(ys)
        values = np.arange(limit + 1)
        # Execution pads to the next captured size, so cost is a step
        # function of the padded size: smooth interpolation would invent
        # marginal per-token costs that don't exist within a pad bucket.
        idx = np.searchsorted(xs, values, side="left")
        result = ys[np.minimum(idx, len(xs) - 1)]
        # Past the capture limit nothing pads, so cost really is continuous in
        # size; snapping to the next profiled point overestimates badly when
        # the profiled points are far apart. Interpolate only between points
        # that are themselves past the limit: crossing the limit loses
        # cudagraphs entirely, which is a genuine discontinuity, so the first
        # point above it must not be blended with the last one below.
        if cudagraph_limit:
            smooth = values > cudagraph_limit
            above = xs > cudagraph_limit
            if smooth.any() and above.any():
                result[smooth] = np.interp(values[smooth], xs[above], ys[above])
        if len(xs) > 1:
            after = values > xs[-1]
            slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
            result[after] = ys[-1] + slope * (values[after] - xs[-1])
        return result

    draft_table = np.maximum(build_table(max_num_reqs, draft_curve), 0.0)
    verify_table = np.maximum(build_table(max_batch_tokens, verify_curve), 1e-6)
    if not np.isfinite(draft_table).all() or not np.isfinite(verify_table).all():
        raise ValueError("DSpark generated cost tables must be finite.")
    return draft_table, verify_table


class AdaptiveVerificationManager:
    def __init__(
        self,
        req_states: "RequestState",
        num_bonus_tokens: int,
        max_total_logits: int,
    ):
        self.req_states = req_states
        self.num_speculative_steps = req_states.num_speculative_steps
        device = req_states.device
        self._copy_stream = torch.cuda.Stream(device)

        self.num_bonus_tokens = num_bonus_tokens
        # Bound adaptive verification to one rejection-sampler chunk.
        self._max_total_logits = max_total_logits
        self.cost_tables: tuple[np.ndarray, np.ndarray] | None = None
        # Largest cudagraph-captured token count; above it nothing pads.
        self._cudagraph_limit = 0
        self._batch_budget: tuple[dict[str, int], dict[str, int], int] | None = None
        max_num_reqs = req_states.max_num_reqs
        # Current per-slot confidences
        self._confidence_probs = torch.empty(
            (max_num_reqs, self.num_speculative_steps),
            dtype=torch.float32,
            device=device,
        )
        self._batch_draft_capacity = CpuGpuBuffer(
            max_num_reqs,
            dtype=torch.int32,
            device=device,
        )
        self._capacity_copy_event = torch.cuda.Event(blocking=True)
        # Two D2H slots preserve stale inputs for budget selection.
        self._stale_confidences = [
            CpuGpuBuffer(
                max_num_reqs,
                self.num_speculative_steps,
                dtype=torch.float32,
                device=device,
            )
            for _ in range(2)
        ]
        self._copy_events = [torch.cuda.Event(blocking=True) for _ in range(2)]
        self._pending_resets: list[int] = []
        self._stale_idx = 0
        for slot in self._stale_confidences:
            slot.np.fill(1.0)

    def add_request(self, req_idx: int) -> None:
        self._stale_confidences[self._stale_idx].np[req_idx].fill(1.0)
        self._pending_resets.append(req_idx)
        self._confidence_probs[req_idx].fill_(1.0)

    def batches_to_profile(self, capture_sizes: list[int]) -> Iterator[dict[str, int]]:
        """Dummy-run kwargs whose step timings seed the cost tables.

        Run these inside StepTimingCollector.collect(), then hand the block's
        timings to set_initial_cost_curves."""
        max_num_tokens = self.req_states.max_num_batched_tokens
        size = self._cudagraph_limit = capture_sizes[-1] if capture_sizes else 0
        # Also profile beyond the capture limit: real steps run there
        # (piecewise/eager) and linear extrapolation badly underestimates
        # them. These runs double as JIT warmup for the piecewise shapes.
        tail_sizes: set[int] = set()
        if size:
            tail_sizes.add(min(size + size // 2, max_num_tokens))
            while size < max_num_tokens:
                size = min(size * 2, max_num_tokens)
                tail_sizes.add(size)
            tail_sizes -= set(capture_sizes)
        for num_tokens in capture_sizes + sorted(tail_sizes):
            for _ in range(_PROFILE_REPLAYS):
                yield {
                    "num_tokens": num_tokens,
                    "context_len": envs.VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN,
                }

    def set_initial_cost_curves(self, samples: list[StepTimingSample]) -> None:
        def median_curve(
            points: Iterable[tuple[int, float]],
        ) -> list[tuple[int, float]]:
            grouped: defaultdict[int, list[float]] = defaultdict(list)
            for key, value in points:
                grouped[key].append(value)
            return [(k, float(np.median(v))) for k, v in sorted(grouped.items())]

        full_graph_samples = [sample for sample in samples if sample.full_cudagraph]
        if full_graph_samples:
            draft_samples = full_graph_samples
            draft_source = "full-cudagraph"
        else:
            # Adaptive varlen FULL graphs are unsafe for backends below ALWAYS
            # support (notably Qwen GDN). In PIECEWISE mode, use the captured
            # size range and exclude eager tail shapes whose request counts can
            # collide with graph-covered points.
            draft_samples = [
                sample
                for sample in samples
                if not self._cudagraph_limit
                or sample.num_target_tokens <= self._cudagraph_limit
            ]
            draft_source = "piecewise"
        draft_curve = median_curve(
            (sample.num_reqs, sample.drafter_ms) for sample in draft_samples
        )
        verify_curve = median_curve(
            (s.num_target_tokens, s.forward_ms) for s in samples
        )
        logger.info(
            "DSpark adaptive profile: context_len=%d, cudagraph_limit=%d, "
            "samples=%d, draft_source=%s, draft_points=%d, verify_points=%d",
            envs.VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN,
            self._cudagraph_limit,
            len(samples),
            draft_source,
            len(draft_curve),
            len(verify_curve),
        )
        self.set_cost_curves(draft_curve, verify_curve)

    def set_cost_curves(
        self,
        draft_curve: list[tuple[int, float]],
        verify_curve: list[tuple[int, float]],
    ) -> None:
        draft_curve, verify_curve = get_tp_group().broadcast_object(
            (draft_curve, verify_curve), src=0
        )
        if not draft_curve or not verify_curve:
            raise ValueError(
                "DSpark adaptive verification requires non-empty draft and "
                "verification cost curves."
            )
        self.cost_tables = build_cost_tables_from_curves(
            draft_curve,
            verify_curve,
            self.req_states.max_num_reqs,
            self.req_states.max_num_batched_tokens,
            self._cudagraph_limit,
        )
        logger.info(
            "DSpark adaptive cost tables ready: draft=%d entries, verify=%d entries",
            len(self.cost_tables[0]),
            len(self.cost_tables[1]),
        )
        logger.debug("DSpark cost tables: %s", self.cost_tables)

    def record_confidences(
        self,
        confidence_probs: torch.Tensor,
        input_batch: "InputBatch",
    ) -> None:
        """Publish this step's raw confidences for the ranking kernel and start
        copying them to the CPU, where a later step's budget reads them."""
        num_reqs = input_batch.num_reqs
        ready_idx = self._stale_idx ^ 1
        with gpu_sync_allowed():
            self._copy_events[ready_idx].synchronize()
        if self._pending_resets:
            self._stale_confidences[ready_idx].np[self._pending_resets] = 1.0
            self._pending_resets.clear()
        # Last step's copy has landed: budgets read it, this step overwrites the
        # slot they were reading before.
        self._stale_idx, write_idx = ready_idx, self._stale_idx

        self._confidence_probs[input_batch.idx_mapping] = confidence_probs[:num_reqs]
        write_slot = self._stale_confidences[write_idx]
        write_slot.gpu.copy_(self._confidence_probs)

        current_stream = torch.cuda.current_stream(self.req_states.device)
        self._copy_stream.wait_stream(current_stream)
        with stream(self._copy_stream, current_stream):
            write_slot.copy_to_cpu()
            self._copy_events[write_idx].record()

    def get_num_tokens(
        self,
        num_tokens_per_req: dict[str, int],
        draft_tokens: dict[str, list[int]],
    ) -> int:
        """Token count once the draft budget is trimmed to fit.

        Stashes the chosen budget in ``_batch_budget`` for the compaction and
        reallocation that follow in the same step.
        """
        assert self.cost_tables is not None
        req_ids = list(num_tokens_per_req)
        num_reqs = len(req_ids)
        scheduled_tokens = np.fromiter(
            num_tokens_per_req.values(), dtype=np.int32, count=num_reqs
        )
        scheduled_drafts = np.fromiter(
            (len(draft_tokens.get(req_id, ())) for req_id in req_ids),
            dtype=np.int32,
            count=num_reqs,
        )
        num_non_draft_tokens = scheduled_tokens - scheduled_drafts
        slots = np.fromiter(
            (self.req_states.req_id_to_index[req_id] for req_id in req_ids),
            dtype=np.int32,
            count=len(req_ids),
        )
        stale_confidences = self._stale_confidences[self._stale_idx].np[slots]
        survival_probability = np.cumprod(stale_confidences.astype(np.float64), axis=1)
        steps = np.arange(self.num_speculative_steps)
        valid = steps[None, :] < scheduled_drafts[:, None]
        scores = np.sort(survival_probability[valid])[::-1]
        num_non_draft_tokens_total = int(num_non_draft_tokens.sum())
        max_draft_budget = min(
            int(scheduled_drafts.sum()),
            max(0, self._max_total_logits - num_reqs * self.num_bonus_tokens),
        )
        scores = scores[:max_draft_budget]
        draft_cost_ms, verify_cost_ms = self.cost_tables
        num_sampling_requests = np.count_nonzero(
            self.req_states.num_computed_tokens_np[slots] + num_non_draft_tokens
            >= self.req_states.prefill_len.np[slots]
        )
        num_tokens_to_estimated_accepted_tokens = np.concatenate(
            ([num_sampling_requests], num_sampling_requests + np.cumsum(scores))
        )
        costs = (
            draft_cost_ms[len(req_ids)]
            + verify_cost_ms[
                num_non_draft_tokens_total : num_non_draft_tokens_total
                + max_draft_budget
                + 1
            ]
        )
        num_non_draft_tokens_per_req = {
            req_id: int(num_tokens)
            for req_id, num_tokens in zip(req_ids, num_non_draft_tokens, strict=True)
        }
        draft_budget = int(np.argmax(num_tokens_to_estimated_accepted_tokens / costs))
        if draft_budget < int(scheduled_drafts.sum()):
            logger.info_once(
                "DSpark adaptive verification reduced scheduled drafts: "
                "profile_context_len=%d, scheduled=%d, selected=%d",
                envs.VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN,
                int(scheduled_drafts.sum()),
                draft_budget,
            )
        num_drafts_per_req = {
            req_id: int(num_drafts)
            for req_id, num_drafts in zip(req_ids, scheduled_drafts, strict=True)
        }
        self._batch_budget = (
            num_drafts_per_req,
            num_non_draft_tokens_per_req,
            draft_budget,
        )
        return sum(num_non_draft_tokens_per_req.values()) + draft_budget

    def allocate_drafts(
        self, req_ids: list[str], idx_mapping: torch.Tensor
    ) -> np.ndarray:
        """Allocate the stale-sized budget with current GPU confidences.

        Returns the exact per-request capacities on CPU. The bounded D2H copy
        is deliberate: attention metadata must see the same query lengths as
        the device kernels before the forward starts.
        """
        batch_budget, self._batch_budget = self._batch_budget, None
        assert batch_budget is not None
        num_drafts_per_req, _, draft_budget = batch_budget
        num_reqs = len(req_ids)
        scheduled_drafts = np.fromiter(
            (num_drafts_per_req[req_id] for req_id in req_ids),
            dtype=np.int32,
            count=num_reqs,
        )
        capacities = self._batch_draft_capacity.gpu[:num_reqs]
        if draft_budget == 0:
            return np.zeros(num_reqs, dtype=np.int32)
        if draft_budget == int(scheduled_drafts.sum()):
            return scheduled_drafts

        async_copy_to_gpu(scheduled_drafts, out=capacities)
        _assign_draft_token_budget_compiled(
            self._confidence_probs,
            idx_mapping,
            capacities,
            draft_budget,
            self.num_speculative_steps,
        )
        self._batch_draft_capacity.copy_to_cpu(num_reqs)
        self._capacity_copy_event.record()
        with gpu_sync_allowed():
            self._capacity_copy_event.synchronize()
        admitted_drafts = self._batch_draft_capacity.np[:num_reqs].copy()
        if np.any(admitted_drafts > scheduled_drafts):
            raise RuntimeError(
                "Adaptive verification admitted more drafts than were scheduled."
            )
        if int(admitted_drafts.sum()) != draft_budget:
            raise RuntimeError(
                "Adaptive verification capacity does not match the selected budget."
            )
        return admitted_drafts
