# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import InputBatch


def _prepare_grammar_bitmask(
    input_batch: InputBatch,
    grammar_req_ids: list[str],
    grammar_bitmask: np.ndarray,
    mask_stride: int,
) -> tuple[list[int], np.ndarray]:
    """Match scheduler grammar rows to the final per-request logit layout.

    The scheduler produces a contiguous group of masks for every structured
    request using the full speculative window. Adaptive verification can later
    keep a shorter prefix for each request. Select that prefix (including the
    row that becomes its bonus-token mask) while retaining the original group
    sizes to find the next request's rows.
    """
    mapping: list[int] = []
    selected_rows: list[int] = []
    req_id_to_idx = {req_id: i for i, req_id in enumerate(input_batch.req_ids)}
    cu_num_logits = input_batch.cu_num_logits_np
    original_num_logits = input_batch.original_num_logits_per_req
    if original_num_logits is None:
        raise RuntimeError(
            "Structured-output mask compaction requires the original per-request "
            "logit layout."
        )
    source_offset = 0

    for grammar_req_id in grammar_req_ids:
        req_idx = req_id_to_idx[grammar_req_id]
        actual_count = int(cu_num_logits[req_idx + 1] - cu_num_logits[req_idx])
        original_count = int(original_num_logits[req_idx])
        if not 0 <= actual_count <= original_count:
            raise RuntimeError(
                "Structured-output logit layout exceeds its scheduler mask layout: "
                f"request={grammar_req_id!r}, actual={actual_count}, "
                f"original={original_count}."
            )

        selected_rows.extend(range(source_offset, source_offset + actual_count))
        mapping.extend(
            req_idx * mask_stride + position for position in range(actual_count)
        )
        source_offset += original_count

    if source_offset != grammar_bitmask.shape[0]:
        raise RuntimeError(
            "Structured-output mask rows do not match the scheduler logit layout: "
            f"masks={grammar_bitmask.shape[0]}, expected={source_offset}."
        )

    if len(selected_rows) != source_offset:
        grammar_bitmask = grammar_bitmask[np.asarray(selected_rows, dtype=np.intp)]
    return mapping, grammar_bitmask


class StructuredOutputsWorker:
    def __init__(
        self,
        max_num_logits: int,
        vocab_size: int,
        device: torch.device,
        mask_stride: int,
    ):
        self.logits_indices = torch.zeros(
            max_num_logits, dtype=torch.int32, device=device
        )
        self.grammar_bitmask = torch.zeros(
            (max_num_logits, cdiv(vocab_size, 32)), dtype=torch.int32, device=device
        )
        self.device = device
        self.copy_stream = torch.cuda.Stream()
        self.mask_stride = mask_stride

    def apply_grammar_bitmask(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        grammar_req_ids: list[str],
        grammar_bitmask: np.ndarray,
    ) -> None:
        if not grammar_req_ids:
            return

        mapping, grammar_bitmask = _prepare_grammar_bitmask(
            input_batch, grammar_req_ids, grammar_bitmask, self.mask_stride
        )
        if not mapping:
            return

        # Asynchronously copy the bitmask to GPU.
        with torch.cuda.stream(self.copy_stream):
            bitmask = async_copy_to_gpu(
                grammar_bitmask, out=self.grammar_bitmask[: grammar_bitmask.shape[0]]
            )

        # Asynchronously copy the mapping to GPU.
        with torch.cuda.stream(self.copy_stream):
            logits_indices = torch.tensor(
                mapping, dtype=torch.int32, device="cpu", pin_memory=True
            )
            logits_indices = self.logits_indices[: len(mapping)].copy_(
                logits_indices, non_blocking=True
            )

        # Ensure all async copies are complete before launching the kernel.
        current_stream = torch.cuda.current_stream()
        current_stream.wait_stream(self.copy_stream)

        num_masks = bitmask.shape[0]
        assert num_masks == len(mapping)
        vocab_size = logits.shape[-1]
        BLOCK_SIZE = 8192
        grid = (num_masks, triton.cdiv(vocab_size, BLOCK_SIZE))
        _apply_grammar_bitmask_kernel[grid](
            logits,
            logits.stride(0),
            logits_indices,
            input_batch.cu_num_logits,
            bitmask,
            bitmask.stride(0),
            vocab_size,
            MASK_STRIDE=self.mask_stride,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # Ensure the copy stream waits for the device tensors to finish being used
        # before it re-uses or deallocates them
        self.copy_stream.wait_stream(current_stream)


# Adapted from
# https://github.com/mlc-ai/xgrammar/blob/main/python/xgrammar/kernels/apply_token_bitmask_inplace_triton.py
@triton.jit
def _apply_grammar_bitmask_kernel(
    logits_ptr,
    logits_stride,
    logits_indices_ptr,
    cu_num_logits_ptr,
    bitmask_ptr,
    bitmask_stride,
    vocab_size,
    MASK_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    bitmask_idx = tl.program_id(0)
    mapping_idx = tl.load(logits_indices_ptr + bitmask_idx)
    req_idx = mapping_idx // MASK_STRIDE
    position_idx = mapping_idx % MASK_STRIDE
    logits_idx = tl.load(cu_num_logits_ptr + req_idx)
    num_req_logits = tl.load(cu_num_logits_ptr + req_idx + 1) - logits_idx
    logits_idx += position_idx
    position_is_active = position_idx < num_req_logits

    # Load the bitmask.
    block_id = tl.program_id(1)
    bitmask_offset = (block_id * BLOCK_SIZE) // 32 + tl.arange(0, BLOCK_SIZE // 32)
    packed_bitmask = tl.load(
        bitmask_ptr + bitmask_idx * bitmask_stride + bitmask_offset,
        mask=bitmask_offset < bitmask_stride,
    )
    # Unpack the bitmask.
    bitmask = ((packed_bitmask[:, None] >> (tl.arange(0, 32)[None, :])) & 1) == 0
    bitmask = bitmask.reshape(BLOCK_SIZE)

    # Apply the bitmask to the logits.
    block_offset = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(
        logits_ptr + logits_idx * logits_stride + block_offset,
        -float("inf"),
        mask=position_is_active & bitmask & (block_offset < vocab_size),
    )
