#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-activation error matrix for NVFP4-KV vs FP8-KV against bf16 references.

Loads a capture directory written by the ``VLLM_NVFP4KV_CAPTURE_DIR`` hook
(``vllm/v1/attention/ops/nvfp4_kv_capture.py``, bf16/auto KV arms), replays
on the *same* captured tensors:

- the port's reference NVFP4 quantize + dequantize chain
  (``nvfp4_quantize_reference`` / ``nvfp4_dequantize_reference``, bit-identical
  to what the Triton store kernel writes), and
- the pilot's FP8 KV arm: e4m3 with scale ``1.0`` (no checkpoint KV scales),
  modelled as a saturating cast to ±448, matching the ``satfinite`` PTX
  conversion the Triton FP8 store path performs,

and reports per layer / per K|V cosine similarity and relative L2 against the
bf16 reference, per-layer rows (aggregated across TP ranks and captured
calls), the all-layer aggregate, and the real-activation gate verdict
(``cos >= 0.995`` and ``rel-L2 <= 0.05``).

CPU-runnable (torch CPU is enough); run inside the stand venv or any env with
torch + the fork installed:

    python tests/v1/attention/nvfp4_kv_real_error_matrix.py \
        --capture-dir /home/user/.tmp/e361-nvfp4kv/capture \
        --out-json matrix.json --out-md matrix.md

Standalone CLI plus importable functions for the CPU unit tests.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import torch

from vllm.v1.attention.ops.triton_nvfp4_kv import (
    nvfp4_dequantize_reference,
    nvfp4_quantize_reference,
)

FP8_E4M3_MAX = 448.0
GATE_COS = 0.995
GATE_REL_L2 = 0.05
ARMS = ("nvfp4", "fp8")
TENSOR_KEYS = ("K", "V")


def fp8_e4m3_scale1(x: torch.Tensor) -> torch.Tensor:
    """Pilot FP8-KV reference arm: e4m3 with scale 1.0, saturating.

    The Triton FP8 store path converts through ``cvt.rn.satfinite.e4m3x2``,
    so out-of-range values clamp to ±448 rather than producing NaN/inf.
    """
    return x.clamp(min=-FP8_E4M3_MAX, max=FP8_E4M3_MAX).to(torch.float8_e4m3fn).to(
        x.dtype
    )


def quantize_arm(arm: str, x: torch.Tensor) -> torch.Tensor:
    """Applies one KV-cache quantization arm to a bf16 tensor."""
    if arm == "nvfp4":
        packed, scale = nvfp4_quantize_reference(x)
        return nvfp4_dequantize_reference(packed, scale).to(x.dtype)
    if arm == "fp8":
        return fp8_e4m3_scale1(x)
    raise ValueError(f"unknown arm: {arm}")


@dataclass
class RunningStats:
    """Streaming cosine / relative-L2 accumulators (float32)."""

    dot: float = 0.0
    norm_a2: float = 0.0
    norm_b2: float = 0.0
    diff2: float = 0.0
    count: int = 0

    def update(self, actual: torch.Tensor, expected: torch.Tensor) -> None:
        a = actual.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        b = expected.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        self.dot += torch.dot(a, b).item()
        self.norm_a2 += torch.dot(a, a).item()
        self.norm_b2 += torch.dot(b, b).item()
        d = a - b
        self.diff2 += torch.dot(d, d).item()
        self.count += b.numel()

    def cosine(self) -> float:
        denom = (self.norm_a2 * self.norm_b2) ** 0.5
        return self.dot / denom if denom > 0.0 else float("nan")

    def rel_l2(self) -> float:
        return (self.diff2 / self.norm_b2) ** 0.5 if self.norm_b2 > 0.0 else float("nan")


@dataclass
class LayerStats:
    """Per (rank, layer) stats for both arms and both tensors."""

    calls: int = 0
    stats: dict[str, dict[str, RunningStats]] = field(
        default_factory=lambda: {
            arm: {key: RunningStats() for key in TENSOR_KEYS} for arm in ARMS
        }
    )


def layer_sort_key(name: str) -> tuple:
    """Orders layer directories numerically by their model layer index."""
    numbers = [int(value) for value in re.findall(r"\d+", name)]
    return (tuple(numbers), name)


def iter_call_files(layer_dir: Path, max_calls: int | None) -> Iterator[Path]:
    """Yields call<i>.pt files in call order, optionally truncated."""
    files = sorted(
        layer_dir.glob("call*.pt"),
        key=lambda path: int(re.search(r"call(\d+)", path.stem).group(1)),
    )
    yield from files[:max_calls] if max_calls else files


def analyze_capture_dir(
    capture_dir: Path | str,
    *,
    max_calls: int | None = None,
    gate_cos: float = GATE_COS,
    gate_rel_l2: float = GATE_REL_L2,
) -> dict:
    """Builds the full error-matrix report for one capture directory."""
    root = Path(capture_dir)
    rank_dirs = sorted(
        (path for path in root.iterdir() if path.is_dir() and path.name.startswith("rank")),
        key=lambda path: int(re.search(r"rank(\d+)", path.name).group(1)),
    )
    if not rank_dirs:
        raise SystemExit(f"no rank*/ directories under {root}")

    per_rank_layer: dict[str, dict[str, LayerStats]] = {}
    layer_names: set[str] = set()
    for rank_dir in rank_dirs:
        rank = rank_dir.name
        for layer_dir in sorted(
            (p for p in rank_dir.iterdir() if p.is_dir()), key=lambda p: layer_sort_key(p.name)
        ):
            layer = per_rank_layer.setdefault(rank, {}).setdefault(
                layer_dir.name, LayerStats()
            )
            for call_file in iter_call_files(layer_dir, max_calls):
                snapshot = torch.load(call_file, map_location="cpu", weights_only=True)
                layer.calls += 1
                for arm in ARMS:
                    for key, tensor_key in (("K", "key"), ("V", "value")):
                        reference = snapshot[tensor_key]
                        layer.stats[arm][key].update(quantize_arm(arm, reference), reference)
            layer_names.add(layer_dir.name)

    ordered_layers = sorted(layer_names, key=layer_sort_key)
    report: dict = {
        "capture_dir": str(root),
        "gate": {"cos_min": gate_cos, "rel_l2_max": gate_rel_l2},
        "ranks": sorted(per_rank_layer, key=lambda name: int(name.removeprefix("rank"))),
        "layer_count": len(ordered_layers),
        "layers": [],
        "aggregate": {},
    }

    overall = {
        arm: {key: RunningStats() for key in TENSOR_KEYS} for arm in ARMS
    }
    for layer_name in ordered_layers:
        layer_report: dict = {"layer": layer_name, "calls": 0, "ranks": {}}
        merged = {arm: {key: RunningStats() for key in TENSOR_KEYS} for arm in ARMS}
        for rank, layers in sorted(per_rank_layer.items()):
            layer = layers.get(layer_name)
            if layer is None:
                continue
            layer_report["calls"] += layer.calls
            rank_row = {}
            for arm in ARMS:
                for key in TENSOR_KEYS:
                    stats = layer.stats[arm][key]
                    rank_row[f"{arm}_{key}_cos"] = round(stats.cosine(), 6)
                    rank_row[f"{arm}_{key}_rel_l2"] = round(stats.rel_l2(), 6)
            layer_report["ranks"][rank] = rank_row
        # Merge by replaying the numeric sums (RunningStats fields are additive).
        for rank, layers in per_rank_layer.items():
            layer = layers.get(layer_name)
            if layer is None:
                continue
            for arm in ARMS:
                for key in TENSOR_KEYS:
                    source = layer.stats[arm][key]
                    target = merged[arm][key]
                    target.dot += source.dot
                    target.norm_a2 += source.norm_a2
                    target.norm_b2 += source.norm_b2
                    target.diff2 += source.diff2
                    target.count += source.count
                    overall[arm][key].dot += source.dot
                    overall[arm][key].norm_a2 += source.norm_a2
                    overall[arm][key].norm_b2 += source.norm_b2
                    overall[arm][key].diff2 += source.diff2
                    overall[arm][key].count += source.count
        for arm in ARMS:
            for key in TENSOR_KEYS:
                stats = merged[arm][key]
                layer_report[f"{arm}_{key}_cos"] = round(stats.cosine(), 6)
                layer_report[f"{arm}_{key}_rel_l2"] = round(stats.rel_l2(), 6)
        layer_report["nvfp4_gate_pass"] = all(
            merged["nvfp4"][key].cosine() >= gate_cos
            and merged["nvfp4"][key].rel_l2() <= gate_rel_l2
            for key in TENSOR_KEYS
        )
        report["layers"].append(layer_report)

    for arm in ARMS:
        for key in TENSOR_KEYS:
            stats = overall[arm][key]
            report["aggregate"][f"{arm}_{key}_cos"] = round(stats.cosine(), 6)
            report["aggregate"][f"{arm}_{key}_rel_l2"] = round(stats.rel_l2(), 6)
            report["aggregate"][f"{arm}_{key}_elements"] = stats.count
    fp8_l2 = max(
        report["aggregate"][f"fp8_{key}_rel_l2"] for key in TENSOR_KEYS
    )
    nvfp4_l2 = max(
        report["aggregate"][f"nvfp4_{key}_rel_l2"] for key in TENSOR_KEYS
    )
    report["aggregate"]["nvfp4_fp8_rel_l2_ratio"] = (
        round(nvfp4_l2 / fp8_l2, 4) if fp8_l2 > 0 else None
    )
    report["aggregate"]["nvfp4_gate_pass"] = all(
        overall["nvfp4"][key].cosine() >= gate_cos
        and overall["nvfp4"][key].rel_l2() <= gate_rel_l2
        for key in TENSOR_KEYS
    )
    failed = [
        layer["layer"] for layer in report["layers"] if not layer["nvfp4_gate_pass"]
    ]
    report["aggregate"]["nvfp4_gate_failing_layers"] = failed
    return report


def write_markdown(report: dict, path: Path | str) -> None:
    """Renders the report as a per-layer + aggregate markdown table."""
    lines = [
        "# NVFP4-KV real-activation error matrix",
        "",
        f"Capture dir: `{report['capture_dir']}`",
        f"Layers captured: {report['layer_count']} "
        "(expected 16 full-attention layers for Qwen3.8-27B)",
        f"Gate: cos >= {report['gate']['cos_min']}, rel-L2 <= {report['gate']['rel_l2_max']}",
        "",
        "| Layer | nvfp4 K cos | nvfp4 K relL2 | nvfp4 V cos | nvfp4 V relL2 "
        "| fp8 K cos | fp8 K relL2 | fp8 V cos | fp8 V relL2 | gate |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for layer in report["layers"]:
        cells = [
            f"{layer[f'nvfp4_{key}_{metric}']:.4f}"
            for key in ("K", "V")
            for metric in ("cos", "rel_l2")
        ] + [
            f"{layer[f'fp8_{key}_{metric}']:.4f}"
            for key in ("K", "V")
            for metric in ("cos", "rel_l2")
        ]
        lines.append(
            "| " + " | ".join([layer["layer"]] + cells + ["pass" if layer["nvfp4_gate_pass"] else "FAIL"]) + " |"
        )
    aggregate = report["aggregate"]
    lines += [
        "",
        "## Aggregate (all captured layers)",
        "",
        f"- nvfp4: K cos {aggregate['nvfp4_K_cos']:.4f}, "
        f"K rel-L2 {aggregate['nvfp4_K_rel_l2']:.4f}; "
        f"V cos {aggregate['nvfp4_V_cos']:.4f}, "
        f"V rel-L2 {aggregate['nvfp4_V_rel_l2']:.4f}",
        f"- fp8:   K cos {aggregate['fp8_K_cos']:.4f}, "
        f"K rel-L2 {aggregate['fp8_K_rel_l2']:.4f}; "
        f"V cos {aggregate['fp8_V_cos']:.4f}, "
        f"V rel-L2 {aggregate['fp8_V_rel_l2']:.4f}",
        f"- nvfp4/fp8 rel-L2 ratio (worst of K/V): {aggregate['nvfp4_fp8_rel_l2_ratio']}",
        f"- gate verdict: {'PASS' if aggregate['nvfp4_gate_pass'] else 'FAIL'}",
    ]
    if aggregate["nvfp4_gate_failing_layers"]:
        lines.append(
            f"- failing layers ({len(aggregate['nvfp4_gate_failing_layers'])}): "
            + ", ".join(aggregate["nvfp4_gate_failing_layers"])
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Parses CLI arguments, analyzes, and writes JSON + markdown outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--max-calls", type=int, default=None)
    parser.add_argument("--gate-cos", type=float, default=GATE_COS)
    parser.add_argument("--gate-rel-l2", type=float, default=GATE_REL_L2)
    args = parser.parse_args()
    report = analyze_capture_dir(
        args.capture_dir,
        max_calls=args.max_calls,
        gate_cos=args.gate_cos,
        gate_rel_l2=args.gate_rel_l2,
    )
    payload = json.dumps(report, indent=2)
    if args.out_json:
        args.out_json.write_text(payload + "\n", encoding="utf-8")
    if args.out_md:
        write_markdown(report, args.out_md)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
