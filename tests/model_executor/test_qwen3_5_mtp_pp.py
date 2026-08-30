# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any, cast

import torch
from torch import nn

from vllm.model_executor.models.interfaces import supports_pp
from vllm.model_executor.models.qwen3_5_mtp import (
    Qwen3_5MTP,
    Qwen3_5MultiTokenPredictor,
)


class _Fusion(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_size = hidden_states.shape[-1] // 2
        return hidden_states[..., :hidden_size] + hidden_states[..., hidden_size:]


class _DraftLayer(nn.Module):
    use_attn_reduce_scatter_for_moe = False

    def forward(
        self,
        *,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del positions, residual
        return hidden_states + 1, torch.zeros_like(hidden_states)


class _FinalNorm(nn.Module):
    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        return hidden_states + residual, None


def test_qwen3_5_mtp_declares_pipeline_support():
    assert supports_pp(Qwen3_5MTP)


def test_qwen3_5_mtp_predictor_runs_whole_draft_on_owning_rank():
    predictor = Qwen3_5MultiTokenPredictor.__new__(Qwen3_5MultiTokenPredictor)
    nn.Module.__init__(predictor)
    predictor.config = cast(Any, SimpleNamespace(hidden_size=4))
    predictor.num_mtp_layers = 1
    predictor.embed_tokens = nn.Embedding(16, 4)
    predictor.pre_fc_norm_embedding = nn.Identity()
    predictor.pre_fc_norm_hidden = nn.Identity()
    predictor.fc = _Fusion()
    predictor.layers = nn.ModuleList([_DraftLayer()])
    predictor.norm = _FinalNorm()

    input_ids = torch.tensor([1, 2])
    hidden_states = torch.ones(2, 4)
    output = predictor.forward(
        input_ids=input_ids,
        positions=torch.tensor([0, 1]),
        hidden_states=hidden_states,
        intermediate_tensors=None,
    )

    assert output.shape == hidden_states.shape
    assert torch.isfinite(output).all()
