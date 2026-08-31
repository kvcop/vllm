# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Auxiliary hidden-state relay across pipeline stages.

EAGLE-3 / DFlash / DSpark drafters read hidden states tapped from several
target layers, but the drafter itself is built only on the last PP rank. These
tests cover the bookkeeping that carries the taps owned by earlier stages to
the stage that drafts.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    aux_hidden_state_pp_key,
    split_aux_hidden_state_pp_layers,
)
from vllm.model_executor.models.qwen3_next import Qwen3NextModel
from vllm.sequence import IntermediateTensors

NUM_LAYERS = 48
AUX_LAYERS = (2, 24, 45)


def _pp_group(rank: int, world_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        rank_in_group=rank,
        world_size=world_size,
        is_first_rank=rank == 0,
        is_last_rank=rank == world_size - 1,
    )


def _pp_bounds(rank: int, world_size: int) -> tuple[int, int]:
    """Even layer split, matching `get_pp_indices` for a divisible model."""
    per_stage = NUM_LAYERS // world_size
    return rank * per_stage, (rank + 1) * per_stage


# --------------------------------------------------------------------------
# Layer-id bookkeeping
# --------------------------------------------------------------------------


def test_split_pp1_keeps_everything_local():
    upstream, outgoing = split_aux_hidden_state_pp_layers(
        AUX_LAYERS, start_layer=0, end_layer=NUM_LAYERS, is_first_rank=True
    )
    assert upstream == ()
    assert outgoing == AUX_LAYERS


def test_split_pp2_first_stage_owns_the_low_taps():
    upstream, outgoing = split_aux_hidden_state_pp_layers(
        AUX_LAYERS, start_layer=0, end_layer=24, is_first_rank=True
    )
    assert upstream == ()
    # Layer 24 is the output *after* decoder layer 23, so stage 0 owns it.
    assert outgoing == (2, 24)


def test_split_pp2_last_stage_receives_the_low_taps():
    upstream, outgoing = split_aux_hidden_state_pp_layers(
        AUX_LAYERS, start_layer=24, end_layer=48, is_first_rank=False
    )
    assert upstream == (2, 24)
    assert outgoing == AUX_LAYERS


def test_split_embedding_tap_belongs_to_the_first_stage_only():
    _, outgoing_first = split_aux_hidden_state_pp_layers(
        (0, 2), start_layer=0, end_layer=24, is_first_rank=True
    )
    assert outgoing_first == (0, 2)

    upstream_last, _ = split_aux_hidden_state_pp_layers(
        (0, 2), start_layer=24, end_layer=48, is_first_rank=False
    )
    assert upstream_last == (0, 2)


@pytest.mark.parametrize("world_size", [2, 3, 4])
def test_handoff_is_exact_across_every_stage_boundary(world_size):
    """Stage k's outgoing set must equal stage k+1's upstream set.

    A mismatch is what would reach the drafter as a misaligned feature vector.
    """
    splits = []
    for rank in range(world_size):
        start, end = _pp_bounds(rank, world_size)
        splits.append(
            split_aux_hidden_state_pp_layers(
                AUX_LAYERS, start, end, is_first_rank=rank == 0
            )
        )
    for rank in range(world_size - 1):
        assert splits[rank][1] == splits[rank + 1][0]
    # The last stage ends up holding every configured tap, in ascending order.
    assert splits[-1][1] == tuple(sorted(AUX_LAYERS))


def test_taps_beyond_the_model_are_never_claimed():
    _, outgoing = split_aux_hidden_state_pp_layers(
        (2, 999), start_layer=24, end_layer=48, is_first_rank=False
    )
    assert outgoing == (2,)


def test_relay_key_is_addressed_by_global_layer_id():
    assert aux_hidden_state_pp_key(24) == "aux_hidden_states.24"
    assert aux_hidden_state_pp_key(0) != aux_hidden_state_pp_key(24)


# --------------------------------------------------------------------------
# Relay buffer keys
# --------------------------------------------------------------------------


def _bare_stack(start_layer: int, end_layer: int) -> Qwen3NextModel:
    stack = object.__new__(Qwen3NextModel)
    nn.Module.__init__(stack)
    stack.start_layer = start_layer
    stack.end_layer = end_layer
    stack._init_intermediate_tensors(8)
    return stack


def test_relay_buffers_gain_a_slot_per_upstream_tap(monkeypatch):
    stack = _bare_stack(24, 48)
    alias = stack.make_empty_intermediate_tensors
    monkeypatch.setattr(
        "vllm.model_executor.models.qwen3_next.get_pp_group",
        lambda: _pp_group(1, 2),
    )

    stack._set_aux_hidden_state_layers(AUX_LAYERS)

    assert stack.aux_hidden_state_pp_upstream_layers == (2, 24)
    assert stack.aux_hidden_state_pp_outgoing_layers == AUX_LAYERS
    # The wrappers alias the bound method captured before configuration; the
    # shared key list is mutated in place so they see the new slots too.
    tensors = alias(batch_size=4, dtype=torch.float32, device="cpu")
    assert list(tensors.tensors) == [
        "hidden_states",
        "residual",
        "aux_hidden_states.2",
        "aux_hidden_states.24",
    ]
    assert tensors["aux_hidden_states.2"].shape == (4, 8)


def test_relay_buffers_stay_bare_without_pipeline_parallelism(monkeypatch):
    stack = _bare_stack(0, NUM_LAYERS)
    monkeypatch.setattr(
        "vllm.model_executor.models.qwen3_next.get_pp_group",
        lambda: _pp_group(0, 1),
    )

    stack._set_aux_hidden_state_layers(AUX_LAYERS)

    assert stack.aux_hidden_state_pp_upstream_layers == ()
    assert stack.aux_hidden_state_pp_outgoing_layers == ()
    tensors = stack.make_empty_intermediate_tensors(
        batch_size=4, dtype=torch.float32, device="cpu"
    )
    assert list(tensors.tensors) == ["hidden_states", "residual"]


def test_reconfiguring_taps_does_not_leak_stale_slots(monkeypatch):
    stack = _bare_stack(24, 48)
    monkeypatch.setattr(
        "vllm.model_executor.models.qwen3_next.get_pp_group",
        lambda: _pp_group(1, 2),
    )

    stack._set_aux_hidden_state_layers(AUX_LAYERS)
    stack._set_aux_hidden_state_layers((24,))

    assert list(
        stack.make_empty_intermediate_tensors(
            batch_size=1, dtype=torch.float32, device="cpu"
        ).tensors
    ) == ["hidden_states", "residual", "aux_hidden_states.24"]


# --------------------------------------------------------------------------
# Capability gate
# --------------------------------------------------------------------------


class _StackWithoutRelay(EagleModelMixin):
    pass


def test_a_stack_without_a_relay_is_rejected_under_pp(monkeypatch):
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_pp_group",
        lambda: _pp_group(0, 2),
    )
    with pytest.raises(ValueError, match="does not relay auxiliary hidden states"):
        _StackWithoutRelay()._set_aux_hidden_state_layers(AUX_LAYERS)


def test_a_stack_without_a_relay_is_fine_without_pp(monkeypatch):
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_pp_group",
        lambda: _pp_group(0, 1),
    )
    stack = _StackWithoutRelay()
    stack._set_aux_hidden_state_layers(AUX_LAYERS)
    assert stack.aux_hidden_state_layers == AUX_LAYERS


def test_qwen3_next_declares_the_relay():
    assert Qwen3NextModel.supports_aux_hidden_state_pp_relay is True
    assert EagleModelMixin.supports_aux_hidden_state_pp_relay is False


# --------------------------------------------------------------------------
# Forward relay
# --------------------------------------------------------------------------


class _Layer(nn.Module):
    """Adds a constant so each layer's output is identifiable."""

    use_attn_reduce_scatter_for_moe = False

    def __init__(self, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx

    def forward(self, *, positions, hidden_states, residual):
        del positions
        return hidden_states + (self.layer_idx + 1), torch.zeros_like(hidden_states)


class _Norm(nn.Module):
    def forward(self, hidden_states, residual):
        return hidden_states + residual, None


class _Stage(Qwen3NextModel):
    """A Qwen3Next decoder stack with the layer bodies replaced.

    Everything under test -- the first-rank embedding tap, the upstream unpack,
    the outgoing pack and the last-rank ordering -- lives in the inherited
    `forward`, which this stub runs verbatim.
    """

    def __init__(self, start_layer: int, end_layer: int, hidden_size: int = 4) -> None:
        nn.Module.__init__(self)
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.layers = nn.ModuleList(_Layer(i) for i in range(NUM_LAYERS))
        self.norm = _Norm()
        self._init_intermediate_tensors(hidden_size)

    def embed_input_ids(self, input_ids):
        return input_ids.to(torch.float32).unsqueeze(-1).repeat(1, 4)


def _run_stage(monkeypatch, stage, rank, world_size, **kwargs):
    monkeypatch.setattr(
        "vllm.model_executor.models.qwen3_next.get_pp_group",
        lambda: _pp_group(rank, world_size),
    )
    return stage.forward(
        input_ids=torch.zeros(3, dtype=torch.long),
        positions=torch.arange(3),
        **kwargs,
    )


def test_forward_relays_every_tap_to_the_last_stage(monkeypatch):
    aux_layers = (0, 2, 24, 45)
    stage0 = _Stage(0, 24)
    stage1 = _Stage(24, 48)

    monkeypatch.setattr(
        "vllm.model_executor.models.qwen3_next.get_pp_group",
        lambda: _pp_group(0, 2),
    )
    stage0._set_aux_hidden_state_layers(aux_layers)
    monkeypatch.setattr(
        "vllm.model_executor.models.qwen3_next.get_pp_group",
        lambda: _pp_group(1, 2),
    )
    stage1._set_aux_hidden_state_layers(aux_layers)

    relayed = _run_stage(monkeypatch, stage0, 0, 2, intermediate_tensors=None)
    assert isinstance(relayed, IntermediateTensors)
    assert list(relayed.tensors) == [
        "hidden_states",
        "residual",
        "aux_hidden_states.0",
        "aux_hidden_states.2",
        "aux_hidden_states.24",
    ]

    hidden_states, aux_hidden_states = _run_stage(
        monkeypatch, stage1, 1, 2, intermediate_tensors=relayed
    )
    assert len(aux_hidden_states) == len(aux_layers)
    # Ascending tap order is what the drafter's `fc` is trained against.
    torch.testing.assert_close(aux_hidden_states[0], relayed["aux_hidden_states.0"])
    torch.testing.assert_close(aux_hidden_states[1], relayed["aux_hidden_states.2"])
    torch.testing.assert_close(aux_hidden_states[2], relayed["aux_hidden_states.24"])
    assert hidden_states.shape == (3, 4)


def test_forward_matches_the_single_stage_result(monkeypatch):
    """The relay must not change which activations the drafter sees."""
    aux_layers = (0, 2, 24, 45)

    whole = _Stage(0, NUM_LAYERS)
    monkeypatch.setattr(
        "vllm.model_executor.models.qwen3_next.get_pp_group",
        lambda: _pp_group(0, 1),
    )
    whole._set_aux_hidden_state_layers(aux_layers)
    reference_hidden, reference_aux = _run_stage(
        monkeypatch, whole, 0, 1, intermediate_tensors=None
    )

    stage0 = _Stage(0, 24)
    stage1 = _Stage(24, 48)
    monkeypatch.setattr(
        "vllm.model_executor.models.qwen3_next.get_pp_group",
        lambda: _pp_group(0, 2),
    )
    stage0._set_aux_hidden_state_layers(aux_layers)
    monkeypatch.setattr(
        "vllm.model_executor.models.qwen3_next.get_pp_group",
        lambda: _pp_group(1, 2),
    )
    stage1._set_aux_hidden_state_layers(aux_layers)

    relayed = _run_stage(monkeypatch, stage0, 0, 2, intermediate_tensors=None)
    split_hidden, split_aux = _run_stage(
        monkeypatch, stage1, 1, 2, intermediate_tensors=relayed
    )

    torch.testing.assert_close(split_hidden, reference_hidden)
    assert len(split_aux) == len(reference_aux)
    for got, want in zip(split_aux, reference_aux, strict=True):
        torch.testing.assert_close(got, want)


def test_forward_does_not_mistake_a_relayed_activation_for_the_embedding(monkeypatch):
    """Tap id 0 is the embedding output and exists only on the first stage."""
    stage1 = _Stage(24, 48)
    monkeypatch.setattr(
        "vllm.model_executor.models.qwen3_next.get_pp_group",
        lambda: _pp_group(1, 2),
    )
    stage1._set_aux_hidden_state_layers((0,))

    relayed = IntermediateTensors(
        {
            "hidden_states": torch.full((3, 4), 7.0),
            "residual": torch.zeros(3, 4),
            "aux_hidden_states.0": torch.full((3, 4), 1.0),
        }
    )
    _, aux_hidden_states = _run_stage(
        monkeypatch, stage1, 1, 2, intermediate_tensors=relayed
    )

    assert len(aux_hidden_states) == 1
    # The relayed embedding tap, not stage 1's inbound activation.
    torch.testing.assert_close(aux_hidden_states[0], torch.full((3, 4), 1.0))


def test_forward_without_taps_relays_only_hidden_states_and_residual(monkeypatch):
    stage0 = _Stage(0, 24)
    monkeypatch.setattr(
        "vllm.model_executor.models.qwen3_next.get_pp_group",
        lambda: _pp_group(0, 2),
    )
    stage0._set_aux_hidden_state_layers(())

    relayed = _run_stage(monkeypatch, stage0, 0, 2, intermediate_tensors=None)
    assert list(relayed.tensors) == ["hidden_states", "residual"]
