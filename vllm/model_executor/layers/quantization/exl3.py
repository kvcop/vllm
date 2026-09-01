# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 (trellis/QTIP-family) weight-only quantization backend.

Loads checkpoints produced by turboderp/exllamav3 (format v1.4.x) into
standard vLLM linear layers for the qwen3_5 family. The dequantization math
and the CUDA kernels come from the ``exllamav3`` Python package, which must be
installed with its compiled extension (``exllamav3.ext``) available.

Storage layout per quantized linear ``{prefix}`` in the safetensors shards:

- ``{prefix}.trellis``  int16 ``(in_features // 16, out_features // 16, 16K)``
  where ``K`` is the per-layer trellis depth (``bits_per_weight`` in the
  checkpoint manifest). Element ``(i, j)`` of the first two dims covers the
  16x16 weight block ``(16i..16i+16, 16j..16j+16)``; the last dim is a dense
  K-bit-per-weight bitstream.
- ``{prefix}.suh``      fp16 ``(in_features,)``  input-side Hadamard sign
  vectors *and* scales (arbitrary values, not just +-1).
- ``{prefix}.svh``      fp16 ``(out_features,)`` output-side sign vectors and
  scales.
- ``{prefix}.mul1``     int32 scalar, records the mul1 codebook multiplier the
  encoder used. The upstream kernels hardcode 0x83DCD12D; loading refuses the
  tensor if it differs, instead of decoding silently wrong values.

The reconstructed weight (in ``exllamav3``'s ``(in, out)`` orientation) is

    W = had_128_n(reconstruct(trellis) * suh) * svh
    reconstruct(trellis) = had_128_k(core), core = exllamav3_ext.reconstruct

with ``had_128`` the 128-point Sylvester Hadamard applied blockwise along the
given dimension, followed by the sign/scale vector. vLLM convention is
``(out, in)``, so apply() computes ``x @ W.T``.

Forward paths (mirroring exllamav3's LinearEXL3):

- small batches (<= VLLM_EXL3_DECODE_ROWS, default 144 rows): the fused
  ``exllamav3_ext.BC_LinearEXL3`` GEMV-style kernels.
- larger batches: slab-streamed dequant (``reconstruct_slice``) plus a
  standard torch GEMM per output slab, one slab in flight, so no bf16 copy of
  the whole layer is ever materialized.
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    LinearMethodBase,
    MergedColumnParallelLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm.model_executor.parameter import Parameter

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization import QuantizationMethods

logger = init_logger(__name__)

# Multiplier baked into the upstream mul1 codebook kernels
# (exl3_lib/quantize.py: codebook_mul1_mult). Checkpoints recording a different
# value cannot be decoded by those kernels and are refused loudly.
CODEBOOK_MUL1_MULT = 0x83DCD12D

# Upstream AUTO_RECONSTRUCT_THRESHOLD: rows at or below this use the fused
# batch-collect kernels instead of materializing dequantized slabs.
DEFAULT_DECODE_ROWS = 144
# Upstream MAX_RECONSTRUCT_SLICE_N: output columns dequantized per slab.
DEFAULT_PREFILL_SLAB = 32768

# vLLM fused module leaf -> exllamav3 checkpoint part names. Every part must be
# EXL3-quantized with the same K for the fusion to be valid; enforced at load.
FUSED_TO_PARTS: dict[str, tuple[str, ...]] = {
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
    "in_proj_qkvz": ("in_proj_qkv", "in_proj_z"),
}

_ext: Any = None
_ext_checked = False


def _get_exllamav3_ext():
    """Import the compiled exllamav3 extension once, or fail loudly."""
    global _ext, _ext_checked
    if not _ext_checked:
        _ext_checked = True
        try:
            from exllamav3.ext import exllamav3_ext

            _ext = exllamav3_ext
        except Exception as e:  # pragma: no cover - environment dependent
            raise ImportError(
                "The exl3 quantization backend requires the upstream "
                "exllamav3 Python package with its compiled CUDA extension "
                "(pip install exllamav3 and import exllamav3 once to JIT "
                "build). The dequantize-to-bf16-only fallback is not "
                "implemented yet, so no kernel-free path exists."
            ) from e
    return _ext


_had_cache: dict[torch.device, torch.Tensor] = {}


def _get_had_128(device: torch.device) -> torch.Tensor:
    """Sylvester Hadamard H_128 / sqrt(128) as fp32, cached per device."""
    had = _had_cache.get(device)
    if had is None:
        h = torch.ones(1, 1, dtype=torch.float16)
        while h.shape[0] < 128:
            h = torch.cat([torch.cat([h, h]), torch.cat([h, -h])], dim=1)
        had = (h.float() / math.sqrt(128)).to(device)
        _had_cache[device] = had
    return had


def dequant_weight(
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    K: int,
    mul1: bool,
) -> torch.Tensor:
    """Dequantize one EXL3 linear to fp16 in (in, out) orientation."""
    ext = _get_exllamav3_ext()
    in_features = trellis.shape[0] * 16
    out_features = trellis.shape[1] * 16
    had = _get_had_128(trellis.device)
    w = torch.empty(in_features, out_features, dtype=torch.half, device=trellis.device)
    ext.reconstruct(w, trellis, K, False, mul1)
    w = had @ w.float().view(-1, 128, out_features)
    w = w.view(in_features, out_features) * suh.float().unsqueeze(1)
    w = (w.view(in_features, -1, 128) @ had).view(in_features, out_features)
    w = w * svh.float().unsqueeze(0)
    return w.half()


def dequant_weight_slice(
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    K: int,
    mul1: bool,
    n_start: int,
    n_end: int,
    out: torch.Tensor,
) -> None:
    """Dequantize output columns [n_start, n_end) of an EXL3 linear.

    Writes the ``(in_features, n_end - n_start)`` fp16 slab into ``out``.
    ``n_start`` must be 128-aligned (kernel requirement).
    """
    ext = _get_exllamav3_ext()
    in_features = trellis.shape[0] * 16
    width = n_end - n_start
    ext.reconstruct_slice(out, trellis, K, False, mul1, n_start)
    had = _get_had_128(trellis.device)
    w = had @ out.float().view(-1, 128, width)
    w = w.view(in_features, width) * suh.float().unsqueeze(1)
    w = (w.view(in_features, -1, 128) @ had).view(in_features, width)
    out.copy_((w * svh.float()[n_start:n_end].unsqueeze(0)).half())


def _as_int_shards(shard_id: Any) -> list[int] | None:
    """Normalize a vLLM shard id into fused-piece indices."""
    if shard_id is None or isinstance(shard_id, bool):
        return None
    if isinstance(shard_id, int):
        return [shard_id]
    if isinstance(shard_id, tuple) and all(isinstance(s, int) for s in shard_id):
        return list(range(shard_id[0], shard_id[-1] + 1))
    return None


class Exl3Config(QuantizationConfig):
    """Quantization config for EXL3 (exllamav3 trellis) checkpoints."""

    def __init__(self, full_config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.full_config: dict[str, Any] = full_config or {}
        self.tensor_storage: dict[str, dict[str, Any]] = self.full_config.get(
            "tensor_storage", {}
        )
        if self.full_config and not self.tensor_storage:
            logger.warning(
                "exl3 quant config has no tensor_storage manifest; layers "
                "cannot resolve their trellis depth K. Pass --quantization "
                "exl3 so vLLM reads quantization_config.json, or the loader "
                "will fail per layer."
            )

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "exl3"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        # The upstream kernels compile for sm_75+; only sm_89 has been
        # exercised by this fork so far.
        return 75

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["quantization_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Exl3Config:
        return cls(full_config=config)

    def _candidate_parts(self, prefix: str) -> tuple[str, ...] | None:
        """Resolve which checkpoint parts feed the vLLM layer at ``prefix``.

        Returns the manifest keys of the EXL3 pieces (one for plain linears,
        several for fused ones), or None if this layer is not EXL3-quantized.
        """
        leaf = prefix.split(".")[-1]
        part_names = FUSED_TO_PARTS.get(leaf, (leaf,))

        def candidates(name: str) -> list[str]:
            # vLLM text models use model.layers..., the checkpoint (and the
            # exllamav3 manifest) uses model.language_model.layers...
            cands = [name]
            if name.startswith("model."):
                cands.append(name.replace("model.", "model.language_model.", 1))
            return cands

        found: list[str] = []
        for part in part_names:
            base = prefix[: -len(leaf)] + part
            entry = None
            for cand in candidates(base):
                e = self.tensor_storage.get(cand)
                if e is not None:
                    entry = (cand, e)
                    break
            if entry is None or entry[1].get("quant_format") != "exl3":
                return None
            found.append(entry[0])
        return tuple(found) or None

    def get_quant_method(
        self, layer: nn.Module, prefix: str
    ) -> LinearMethodBase | None:
        is_linear = isinstance(layer, LinearBase) or (
            hasattr(layer, "input_size") and hasattr(layer, "output_size")
        )
        if is_linear:
            parts = self._candidate_parts(prefix)
            if parts is not None:
                return Exl3LinearMethod(self, prefix, parts)
            return UnquantizedLinearMethod()
        if type(layer).__name__ in (
            "VocabParallelEmbedding",
            "ParallelLMHead",
        ):
            parts = self._candidate_parts(prefix)
            if parts is not None:
                raise NotImplementedError(
                    f"{prefix}: EXL3-quantized lm_head/embedding is not "
                    "supported yet; the trellis decode path for a 248k-row "
                    "vocab projection is tracked as the next work item."
                )
        return None


class Exl3LinearMethod(LinearMethodBase):
    """Weight loader and forward paths for one EXL3 linear layer."""

    def __init__(
        self,
        quant_config: Exl3Config,
        prefix: str,
        manifest_parts: tuple[str, ...],
    ) -> None:
        self.quant_config = quant_config
        self.prefix = prefix
        self.manifest_parts = manifest_parts
        self.decode_rows = int(
            os.environ.get("VLLM_EXL3_DECODE_ROWS", DEFAULT_DECODE_ROWS)
        )
        self.prefill_slab = int(
            os.environ.get("VLLM_EXL3_PREFILL_SLAB", DEFAULT_PREFILL_SLAB)
        )

    # -- geometry -----------------------------------------------------------

    def _layer_geometry(self, layer: LinearBase) -> dict[str, Any]:
        """Full and local (TP) in/out sizes of the vLLM linear.

        Duck-typed on purpose: layer classes are detected by attributes
        (``output_sizes`` for merged-column, ``input_size_per_partition`` for
        row-parallel) so the loader also works against lightweight stand-ins
        in unit tests.
        """
        full_in = layer.input_size
        full_out = layer.output_size
        merged_sizes = getattr(layer, "output_sizes", None)
        is_row = isinstance(layer, RowParallelLinear) or (
            not isinstance(layer, (ColumnParallelLinear, MergedColumnParallelLinear))
            and getattr(layer, "input_size_per_partition", None) is not None
        )
        if merged_sizes is not None:
            piece_full = list(merged_sizes)
            piece_local = [s // layer.tp_size for s in piece_full]
        elif is_row:
            piece_full = None
            piece_local = None
        else:
            piece_full = [full_out]
            piece_local = [full_out // layer.tp_size]
        local_in = full_in // layer.tp_size if piece_local is None else full_in
        local_out = (
            full_out // layer.tp_size if piece_local is None else sum(piece_local)
        )
        return {
            "full_in": full_in,
            "full_out": full_out,
            "local_in": local_in,
            "local_out": local_out,
            "piece_full": piece_full,
            "piece_local": piece_local,
            "is_row": piece_local is None,
        }

    def _manifest_K(self) -> int:
        Ks = set()
        for part in self.manifest_parts:
            entry = self.quant_config.tensor_storage[part]
            K = entry.get("bits_per_weight")
            if not isinstance(K, int) or not 1 <= K <= 8:
                raise ValueError(
                    f"{self.prefix}: manifest entry {part} has unsupported "
                    f"bits_per_weight {K!r}"
                )
            Ks.add(K)
        if len(Ks) != 1:
            raise ValueError(
                f"{self.prefix}: fused parts {self.manifest_parts} have "
                f"different trellis depths {sorted(Ks)}; fusing layers with "
                "different K is not supported"
            )
        return Ks.pop()

    # -- weight creation ----------------------------------------------------

    def create_weights(
        self,
        layer: LinearBase,
        input_size: int | list[int],
        output_size: int | list[int],
        full_input_size: int | list[int],
        full_output_size: int | list[int],
        params_dtype: torch.dtype | None = None,
        weight_loader: Any = None,
        **extra_weight_attrs: Any,
    ) -> None:
        geo = self._layer_geometry(layer)
        K = self._manifest_K()
        for dim, name in (
            (geo["full_in"], "input"),
            (geo["full_out"], "output"),
        ):
            if dim % 128 != 0:
                raise ValueError(
                    f"{self.prefix}: EXL3 requires {name} size divisible by "
                    f"128, got {dim}"
                )

        layer.exl3_K = K
        layer.exl3_meta = {
            "prefix": self.prefix,
            "manifest_parts": self.manifest_parts,
            **geo,
        }

        layer.trellis = Parameter(
            torch.zeros(
                geo["local_in"] // 16,
                geo["local_out"] // 16,
                16 * K,
                dtype=torch.int16,
            ),
            requires_grad=False,
        )
        # Input-side scales: one column per fused piece. Parts of a fused
        # layer carry independently optimized suh vectors in the checkpoint
        # (verified on Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw), so they cannot be
        # collapsed into one vector. Single-piece layers keep the 2D shape
        # too; apply() squeezes where possible.
        num_pieces = 1 if geo["piece_full"] is None else len(geo["piece_full"])
        layer.suh = Parameter(
            torch.zeros(geo["local_in"], num_pieces, dtype=torch.half),
            requires_grad=False,
        )
        layer.exl3_num_pieces = num_pieces
        layer.svh = Parameter(
            torch.zeros(geo["local_out"], dtype=torch.half),
            requires_grad=False,
        )
        layer.mul1 = Parameter(torch.zeros((), dtype=torch.int32), requires_grad=False)
        layer.mul1_seen = False

        # Full control over shard/offset math lives here; the generic
        # parallel-linear loaders assume 1D/2D feature-indexed params and
        # cannot express trellis' 16x16 block grid.
        layer.weight_loader = self._make_weight_loader(layer)

    # -- weight loading -----------------------------------------------------

    def _piece_placement(
        self, layer: LinearBase, shard_id: Any, src: torch.Tensor
    ) -> dict[str, int]:
        """Local output slot of an incoming column-parallel piece.

        Returns ``local_start``/``local_len`` (features within this rank's
        fused output) plus ``span`` (number of fused sub-parts covered).
        """
        geo = layer.exl3_meta
        piece_full = geo["piece_full"]
        assert piece_full is not None

        idxs = _as_int_shards(shard_id)
        if idxs is None and len(piece_full) == 1:
            idxs = [0]
        if idxs is None or not all(0 <= i < len(piece_full) for i in idxs):
            # Fall back to matching the loaded extent against piece sizes.
            extent = src.shape[1] if src.dim() >= 2 else src.shape[0]
            idxs = None
            for i, size in enumerate(piece_full):
                if size == extent:
                    idxs = [i]
                    break
            if idxs is None:
                raise ValueError(
                    f"{self.prefix}: cannot place incoming piece "
                    f"(shard_id={shard_id!r}, shape={tuple(src.shape)})"
                )

        span = idxs[-1] - idxs[0] + 1
        piece_local = geo["piece_local"]
        assert piece_local is not None
        local_start = sum(piece_local[: idxs[0]]) + layer.tp_rank * piece_local[idxs[0]]
        local_len = sum(piece_local[idxs[0] : idxs[-1] + 1])
        return {"local_start": local_start, "local_len": local_len, "span": span}

    def _make_weight_loader(self, layer: LinearBase) -> Any:
        method = self

        def weight_loader(
            param_name: str,
            loaded_weight: torch.Tensor,
            shard_id: Any = None,
        ) -> None:
            geo = layer.exl3_meta
            dev = layer.trellis.device
            src = loaded_weight.to(dev)

            if shard_id is not None and geo["is_row"]:
                raise ValueError(
                    f"{method.prefix}: unexpected shard_id {shard_id!r} on a "
                    "row-parallel EXL3 layer"
                )

            # Placement of the incoming piece. Column-ish layers slice the
            # output dim (fused pieces + TP); row-parallel layers slice the
            # input dim (TP only). Values are in features; trellis block
            # offsets divide by 16 at the copy site.
            if geo["is_row"]:
                out_start, out_len = 0, geo["local_out"]
                in_start, in_len = (
                    layer.tp_rank * geo["local_in"],
                    geo["local_in"],
                )
                in_src = slice(in_start, in_start + in_len)
                out_src = slice(0, None)
                out_slot = slice(0, out_len)
            else:
                piece = method._piece_placement(layer, shard_id, src)
                if piece["span"] > 1 and layer.tp_size > 1:
                    raise NotImplementedError(
                        f"{method.prefix}: a checkpoint piece spanning "
                        f"{piece['span']} fused sub-parts cannot be "
                        "TP-sharded by one contiguous copy; per-part "
                        "placement is not implemented yet (TP=1 works)"
                    )
                out_start, out_len = piece["local_start"], piece["local_len"]
                in_start, in_len = 0, geo["local_in"]
                in_src = slice(0, geo["full_in"])
                # Source slice within the incoming (pre-TP) piece.
                r, n = layer.tp_rank, out_len
                out_src = (
                    slice(r * n, (r + 1) * n) if layer.tp_size > 1 else slice(0, None)
                )
                out_slot = slice(out_start, out_start + out_len)

            if param_name == "trellis":
                src_blk = src[in_src_slice16(in_src), out_src_slice16(out_src), :]
                expected = (in_len // 16, out_len // 16, layer.exl3_K * 16)
                if tuple(src_blk.shape) != expected:
                    raise ValueError(
                        f"{method.prefix}: trellis piece "
                        f"{tuple(src.shape)} maps to slot {expected}, got "
                        f"{tuple(src_blk.shape)}"
                    )
                layer.trellis.data[
                    in_start // 16 : (in_start + in_len) // 16,
                    out_slot.start // 16 : out_slot.stop // 16,
                    :,
                ].copy_(src_blk)
            elif param_name == "suh":
                piece_in = src[in_src]
                if piece_in.numel() != in_len:
                    raise ValueError(
                        f"{method.prefix}: suh piece has "
                        f"{piece_in.numel()} elements, expected {in_len}"
                    )
                piece_in = piece_in.half()
                if geo["is_row"]:
                    cols = [0]
                else:
                    piece_full = geo["piece_full"]
                    assert piece_full is not None
                    idxs = _as_int_shards(shard_id)
                    if idxs is None and len(piece_full) == 1:
                        idxs = [0]
                    if idxs is None:
                        raise ValueError(
                            f"{method.prefix}: cannot place suh piece "
                            f"(shard_id={shard_id!r})"
                        )
                    cols = list(range(idxs[0], idxs[-1] + 1))
                for c in cols:
                    layer.suh.data[:, c].copy_(piece_in)
            elif param_name == "svh":
                piece_out = src[out_src] if geo["is_row"] else src[out_src]
                if piece_out.numel() != out_len:
                    raise ValueError(
                        f"{method.prefix}: svh piece has "
                        f"{piece_out.numel()} elements, expected {out_len}"
                    )
                layer.svh.data[out_slot].copy_(piece_out.half())
            elif param_name == "mul1":
                value = int(loaded_weight.flatten()[0].item())
                if value & 0xFFFFFFFF != CODEBOOK_MUL1_MULT:
                    raise ValueError(
                        f"{method.prefix}: checkpoint records mul1 "
                        f"multiplier {value & 0xFFFFFFFF:#x}, but the "
                        f"upstream kernels hardcode {CODEBOOK_MUL1_MULT:#x}; "
                        "decoding would be wrong"
                    )
                if layer.mul1_seen:
                    if int(layer.mul1.item()) != value:
                        raise ValueError(
                            f"{method.prefix}: fused parts disagree on the "
                            "mul1 multiplier"
                        )
                else:
                    layer.mul1.data.fill_(value)
                    layer.mul1_seen = True
            else:
                raise ValueError(
                    f"{method.prefix}: unexpected EXL3 tensor {param_name!r}"
                )

        return weight_loader

    def process_weights_after_loading(self, layer: LinearBase) -> None:
        geo = layer.exl3_meta
        if not layer.mul1_seen:
            raise ValueError(
                f"{self.prefix}: no mul1 tensor arrived; incomplete EXL3 "
                "checkpoint shard?"
            )
        for param in (layer.trellis, layer.suh, layer.svh):
            if not param.is_contiguous():
                param.data = param.data.contiguous()
        piece_local = geo["piece_local"]
        if piece_local is None:
            layer.exl3_piece_ranges = [(0, geo["local_out"])]
        else:
            ranges = []
            off = 0
            for sz in piece_local:
                ranges.append((off, off + sz))
                off += sz
            layer.exl3_piece_ranges = ranges

        ext = _get_exllamav3_ext()
        device = layer.trellis.device
        layer.exl3_bc = None
        if layer.exl3_num_pieces == 1:
            # The BC kernels take a single suh vector, so only single-piece
            # layers (o_proj/down_proj/out_proj, lm_head later) get the
            # fused decode path. Fused layers run slab-dequant for all row
            # counts until per-piece BC routing lands (see REPORT).
            layer.exl3_xh = torch.empty(
                (1, geo["local_in"]), dtype=torch.half, device=device
            )
            layer.exl3_bc = ext.BC_LinearEXL3(
                layer.trellis.data,
                layer.suh.data[:, 0].contiguous(),
                layer.svh.data,
                layer.exl3_K,
                None,  # bias: EXL3 linears in these checkpoints are bias-free
                False,  # mcg
                True,  # mul1
                layer.exl3_xh,
            )
        else:
            logger.info_once(
                "%s: EXL3 fused layer (%d pieces) uses slab-dequant for "
                "all batch sizes; BC decode path is single-piece only",
                self.prefix,
                layer.exl3_num_pieces,
            )

    # -- forward ------------------------------------------------------------

    def apply(
        self,
        layer: LinearBase,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bias is not None:
            raise NotImplementedError(
                f"{self.prefix}: EXL3 layers with bias are not supported"
            )
        geo = layer.exl3_meta
        out_features = geo["local_out"]
        orig_dtype = x.dtype
        orig_shape = x.shape
        x2 = x.reshape(-1, x.shape[-1])
        rows = x2.shape[0]

        use_bc = rows <= self.decode_rows and layer.exl3_bc is not None
        if use_bc:
            xh = x2.to(torch.half).contiguous()
            y = layer.exl3_bc.run_alloc(xh, out_features, False)
            y = y.to(orig_dtype)
        else:
            slab = self.prefill_slab
            y = torch.empty(rows, out_features, dtype=orig_dtype, device=x.device)
            # Walk fused pieces: each has its own suh column and occupies a
            # contiguous column range of the trellis grid.
            pieces = layer.exl3_piece_ranges
            for piece_id, (p_start, p_end) in enumerate(pieces):
                suh_col = layer.suh.data[:, piece_id]
                for n0 in range(p_start, p_end, slab):
                    n1 = min(n0 + slab, p_end)
                    ws = torch.empty(
                        x2.shape[1], n1 - n0, dtype=torch.half, device=x.device
                    )
                    dequant_weight_slice(
                        layer.trellis.data,
                        suh_col,
                        layer.svh.data,
                        layer.exl3_K,
                        True,
                        n0,
                        n1,
                        ws,
                    )
                    y[:, n0:n1] = x2 @ ws.to(orig_dtype)
        return y.view(*orig_shape[:-1], out_features)


def in_src_slice16(in_src: slice) -> slice:
    """Block (//16) view of a feature-range slice, for trellis sources."""
    start = in_src.start if in_src.start is not None else 0
    stop = in_src.stop
    assert stop is not None
    return slice(start // 16, stop // 16)


def out_src_slice16(out_src: slice) -> slice:
    """Block (//16) view of an output-range slice, for trellis sources."""
    start = out_src.start if out_src.start is not None else 0
    stop = out_src.stop
    end = -1 if stop is None else stop
    return slice(start // 16, end // 16 if stop is not None else None)
