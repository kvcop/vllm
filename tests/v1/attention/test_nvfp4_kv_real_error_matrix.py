# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only unit tests for the NVFP4-KV real-activation error matrix.

Covers, on synthetic tensors and without any GPU:

1. the streaming cosine / relative-L2 accumulators against direct
   computation;
2. the NVFP4 and FP8 arms on tensors pinned to their representable grids
   (exact round-trips) and the FP8 saturation behaviour;
3. the capture module: env gating, file layout, per-layer call bound, and
   the never-raise contract;
4. ``analyze_capture_dir`` end-to-end on a synthetic two-rank capture
   directory, including the gate verdict and markdown rendering.

Run standalone or via pytest inside an env with torch and the fork's
``vllm.v1.attention.ops.triton_nvfp4_kv`` importable (the stand venv).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nvfp4_kv_real_error_matrix import (  # noqa: E402
    FP8_E4M3_MAX,
    RunningStats,
    analyze_capture_dir,
    fp8_e4m3_scale1,
    quantize_arm,
    write_markdown,
)
from vllm.v1.attention.ops import nvfp4_kv_capture  # noqa: E402
from vllm.v1.attention.ops.triton_nvfp4_kv import (  # noqa: E402
    nvfp4_dequantize_reference,
    nvfp4_quantize_reference,
)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def test_running_stats_match_direct_computation() -> None:
    """Chunked accumulation equals one-shot cosine / rel-L2."""
    torch.manual_seed(0)
    expected = torch.randn(7, 4, 256).bfloat16()
    actual = expected + torch.randn_like(expected) * 0.05
    stats = RunningStats()
    for start in range(0, 7, 3):
        stats.update(actual[start : start + 3], expected[start : start + 3])
    assert stats.cosine() == pytest.approx(_cos(actual, expected), abs=1e-5)
    assert stats.rel_l2() == pytest.approx(_rel_l2(actual, expected), abs=1e-5)
    assert stats.count == expected.numel()


def test_nvfp4_arm_exact_on_grid_tensor() -> None:
    """Values already on the NVFP4 grid round-trip bit-exactly."""
    torch.manual_seed(1)
    raw = torch.randn(64, 4, 256).bfloat16()
    on_grid = nvfp4_dequantize_reference(
        *nvfp4_quantize_reference(raw)
    ).to(torch.bfloat16)
    replayed = quantize_arm("nvfp4", on_grid)
    assert torch.equal(replayed, on_grid)
    stats = RunningStats()
    stats.update(replayed, on_grid)
    assert stats.cosine() == pytest.approx(1.0)
    assert stats.rel_l2() == pytest.approx(0.0, abs=1e-9)


def test_fp8_arm_exact_on_grid_and_saturates() -> None:
    """e4m3-grid values survive the scale-1.0 arm; overflow clamps to 448."""
    torch.manual_seed(2)
    base = torch.randn(64, 4, 256).bfloat16()
    on_grid = base.to(torch.float8_e4m3fn).to(torch.bfloat16)
    assert torch.equal(fp8_e4m3_scale1(on_grid), on_grid)
    hot = torch.tensor([1000.0, -1000.0, 448.0, -448.0]).bfloat16()
    assert torch.equal(
        fp8_e4m3_scale1(hot),
        torch.tensor([FP8_E4M3_MAX, -FP8_E4M3_MAX, 448.0, -448.0]).bfloat16(),
    )
    assert torch.isfinite(fp8_e4m3_scale1(hot).float()).all()


@pytest.fixture()
def capture_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Enables the capture hook against tmp_path and resets module state."""
    root = tmp_path / "capture"
    monkeypatch.setenv(nvfp4_kv_capture.CAPTURE_ENV_VAR, str(root))
    nvfp4_kv_capture._STATE = None
    nvfp4_kv_capture._STATE_INITIALIZED = False
    yield root
    nvfp4_kv_capture._STATE = None
    nvfp4_kv_capture._STATE_INITIALIZED = False


def test_capture_module_layout_and_call_bound(capture_env: Path) -> None:
    """Snapshots land under rank0/<layer>/call<i>.pt, bounded per layer."""
    torch.manual_seed(3)
    layer = "model.layers.3.self_attn"
    key = torch.randn(32, 4, 256).bfloat16()
    value = torch.randn(32, 4, 256).bfloat16()
    beyond = nvfp4_kv_capture.MAX_CALLS_PER_LAYER + 8
    for _ in range(beyond):
        nvfp4_kv_capture.capture_kv_snapshot(layer, key, value)
    layer_dir = capture_env / "rank0" / layer
    files = sorted(layer_dir.glob("call*.pt"))
    assert len(files) == nvfp4_kv_capture.MAX_CALLS_PER_LAYER
    snapshot = torch.load(files[0], map_location="cpu", weights_only=True)
    assert snapshot["key"].dtype == torch.bfloat16
    assert snapshot["key"].shape == (32, 4, 256)
    assert snapshot["value"].shape == (32, 4, 256)
    assert snapshot["kv_cache_dtype"] == "bfloat16"
    assert snapshot["key"].shape[1] == snapshot["num_kv_heads"] == 4
    assert snapshot["key"].shape[2] == snapshot["head_dim"] == 256


def test_capture_module_inert_without_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No state and no files when the env var is unset."""
    monkeypatch.delenv(nvfp4_kv_capture.CAPTURE_ENV_VAR, raising=False)
    nvfp4_kv_capture._STATE = None
    nvfp4_kv_capture._STATE_INITIALIZED = False
    nvfp4_kv_capture.capture_kv_snapshot(
        "some.layer", torch.randn(4, 4, 256).bfloat16(), torch.randn(4, 4, 256).bfloat16()
    )
    assert nvfp4_kv_capture._STATE is None
    assert not any(tmp_path.rglob("call*.pt"))
    nvfp4_kv_capture._STATE_INITIALIZED = False


def test_capture_module_never_raises(capture_env: Path) -> None:
    """Bad shapes are skipped; an unwritable root disables, not raises."""
    nvfp4_kv_capture.capture_kv_snapshot(
        "bad.shape", torch.randn(4, 256).bfloat16(), torch.randn(4, 256).bfloat16()
    )
    assert not (capture_env / "rank0" / "bad.shape").exists()
    blocker = capture_env / "blocker"
    blocker.write_text("not a directory")
    nvfp4_kv_capture._STATE.root = blocker
    nvfp4_kv_capture.capture_kv_snapshot(
        "blocked.layer",
        torch.randn(4, 4, 256).bfloat16(),
        torch.randn(4, 4, 256).bfloat16(),
    )
    state = nvfp4_kv_capture._STATE
    assert state is not None and state.disabled


def _write_capture_tree(
    root: Path, ranks: list[int], layers: list[str], calls: int, corrupt: bool
) -> None:
    """Writes a synthetic capture tree with grid-exact (or noisy) tensors."""
    torch.manual_seed(4)
    for rank in ranks:
        for layer in layers:
            layer_dir = root / f"rank{rank}" / layer
            layer_dir.mkdir(parents=True, exist_ok=True)
            for call in range(calls):
                key = nvfp4_dequantize_reference(
                    *nvfp4_quantize_reference(torch.randn(16, 1, 256))
                ).to(torch.bfloat16)
                value = nvfp4_dequantize_reference(
                    *nvfp4_quantize_reference(torch.randn(16, 1, 256))
                ).to(torch.bfloat16)
                if corrupt:
                    key = key + torch.randn_like(key) * 0.75
                    value = value + torch.randn_like(value) * 0.75
                torch.save(
                    {
                        "key": key.contiguous(),
                        "value": value.contiguous(),
                        "num_kv_heads": 1,
                        "head_dim": 256,
                        "kv_cache_dtype": "bfloat16",
                    },
                    layer_dir / f"call{call}.pt",
                )


def test_analyze_capture_dir_clean_pass_and_corrupt_fail(tmp_path: Path) -> None:
    """Grid-exact captures pass the gate; noisy ones fail with layer names."""
    layers = ["model.layers.0.self_attn", "model.layers.3.self_attn"]
    clean = tmp_path / "clean"
    _write_capture_tree(clean, ranks=[0, 1], layers=layers, calls=2, corrupt=False)
    report = analyze_capture_dir(clean)
    assert report["layer_count"] == 2
    assert report["ranks"] == ["rank0", "rank1"]
    assert len(report["layers"]) == 2
    assert report["layers"][0]["layer"] == "model.layers.0.self_attn"
    for layer in report["layers"]:
        assert layer["calls"] == 4  # 2 ranks x 2 calls
        assert layer["nvfp4_gate_pass"] is True
        assert layer["nvfp4_K_cos"] == pytest.approx(1.0, abs=1e-6)
        assert layer["nvfp4_K_rel_l2"] == pytest.approx(0.0, abs=1e-9)
        assert layer["fp8_K_cos"] == pytest.approx(1.0, abs=1e-6)
    assert report["aggregate"]["nvfp4_gate_pass"] is True
    assert report["aggregate"]["nvfp4_fp8_rel_l2_ratio"] == pytest.approx(0.0, abs=1e-9)
    assert report["aggregate"]["nvfp4_K_elements"] == 2 * 2 * 16 * 1 * 256

    corrupt = tmp_path / "corrupt"
    _write_capture_tree(corrupt, ranks=[0], layers=layers, calls=1, corrupt=True)
    failed = analyze_capture_dir(corrupt)
    assert failed["aggregate"]["nvfp4_gate_pass"] is False
    assert failed["aggregate"]["nvfp4_gate_failing_layers"] == layers
    for layer in failed["layers"]:
        assert layer["nvfp4_gate_pass"] is False

    md_path = tmp_path / "matrix.md"
    write_markdown(failed, md_path)
    text = md_path.read_text(encoding="utf-8")
    assert "real-activation error matrix" in text
    assert "| Layer |" in text
    assert "FAIL" in text


def test_analyze_capture_dir_respects_max_calls(tmp_path: Path) -> None:
    """The analysis can truncate calls for a quick pass."""
    root = tmp_path / "cap"
    _write_capture_tree(root, ranks=[0], layers=["model.layers.5.self_attn"], calls=4, corrupt=False)
    full = analyze_capture_dir(root)
    truncated = analyze_capture_dir(root, max_calls=1)
    assert full["layers"][0]["calls"] == 4
    assert truncated["layers"][0]["calls"] == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
