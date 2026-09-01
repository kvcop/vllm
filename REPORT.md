# EXL3 weight-quantization backend — day-1 report (track 1)

Branch `exl3-weights-backend` off `qwen38-v0271-fork`. Scope of this day:
format study, backend skeleton, first passing tests. Verified against the
real checkpoint `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` (tensors range-fetched
tensor-by-tensor, no full download) and against upstream
`turboderp-org/exllamav3` v1.4.5 as the kernel/format source of truth.

## §format — EXL3 tensor layout (verified against the checkpoint)

Every quantized linear `{key}` ships four tensors in the safetensors shards:

| tensor | dtype | shape | meaning |
|---|---|---|---|
| `{key}.trellis` | int16 | `(in//16, out//16, 16K)` | packed trellis codes; cell `(i,j)` covers the 16x16 weight block; last dim is a dense K-bit-per-weight bitstream |
| `{key}.suh` | fp16 | `(in,)` | input-side Hadamard sign **and scale** vector |
| `{key}.svh` | fp16 | `(out,)` | output-side sign and scale vector |
| `{key}.mul1` | int32 | scalar | mul1 codebook multiplier used by the encoder |

`K = trellis.shape[-1] // 16 = bits_per_weight` per layer. In this checkpoint
all body linears are K=4, `lm_head` is K=6; the manifest
(`quantization_config.json`, key `tensor_storage`) records per-module shapes,
`bits_per_weight` and `mul1_multiplier` — vLLM pre-allocation needs it to
know K before tensors arrive.

Facts that differ from the day's recon brief:

- **lm_head IS quantized** (K=6) in this checkpoint; only `embed_tokens` is
  plain bf16. Vision tower weights are present as plain bf16 too (the config
  is multimodal; recon called it text-only).
- `suh`/`svh` are **not** ±1 sign vectors: they carry arbitrary fp16 values
  (measured: suh ~±0.03, svh ~±1.6). The ±1 form is only what the *packed*
  `su`/`sv` bitfield variant produces (absent here).
- The stored `mul1` equals `0x83DCD12D`, the constant hardcoded in the
  upstream kernels (upstream ignores the stored value; a future checkpoint
  with a different multiplier would decode wrongly — our loader refuses it).
- `su`/`sv` packed forms and `mcg` exist in the format but are absent in
  this checkpoint; `mcg` is unimplemented here.

Codebook decode is procedural (`ext/quant/codebook.cuh`, `3INST`): main
table `x = idx*89226354 + 64248484` then a LOP3 that materializes two fp16s
whose sum is the codeword. Trellis unpacking (`ext/quant/exl3_dq.cuh`): each
weight reads a 16-bit window ending at its K-bit boundary out of the
circular per-tile bitstream — 256 weights per 16x16 tile, `256*K` bits.

Reconstruction (matches `LinearEXL3.get_weight_tensor` to 5e-4 max rel):

```
core = ext.reconstruct(trellis, K, mcg=False, mul1=True)   # (in, out) fp16
W    = blkhad_128(core, dim=in)  * suh[:, None]            # H_128 Sylvester / sqrt(128)
W    = blkhad_128(W,     dim=out) * svh[None, :]
```

End-to-end check against the **original bf16 weights**
(`Qwen/Qwen3.8-27B`, layer 3 `q_proj`): rel L2 7.05 %, cos 0.9975 — the
decode recovers the true weights, not just self-consistent output.

Checkpoint→vLLM name mapping (qwen3_5): the fork already fuses
`in_proj_qkv|in_proj_z → in_proj_qkvz` (shards (0,1,2)/(3)), `q|k|v →
qkv_proj`, `gate|up → gate_up_proj` via `WeightsMapper` + packed modules —
all suffix-agnostic, so `.trellis/.suh/.svh/.mul1` route through unchanged.

## §design

Chosen: a native vLLM `QuantizationConfig` ("exl3",
`vllm/model_executor/layers/quantization/exl3.py`) that

- resolves each vLLM layer prefix to its manifest parts (handles fused
  leaves `qkv_proj`/`gate_up_proj`/`in_proj_qkvz` and the
  `model.` vs `model.language_model.` prefix difference),
- creates params `trellis` (int16 3D), `suh` (fp16 `(in, pieces)` — see the
  fused-suh finding below), `svh` (fp16 `(out,)`), `mul1` (int32 scalar),
- installs its own `layer.weight_loader` (humming-style interception)
  computing stacked-shard + TP offsets itself; trellis offsets divide by 16,
- dequantizes via the upstream kernels: `BC_LinearEXL3` fused kernels for
  decode-sized batches (rows ≤ 144, env `VLLM_EXL3_DECODE_ROWS`), and
  slab-streamed `reconstruct_slice` + torch GEMM for prefill
  (`VLLM_EXL3_PREFILL_SLAB`, default 32768 columns), so no bf16 copy of a
  layer is ever materialized (27B at 3.5 bpw stays ~14 GB).

Alternatives considered: materialize bf16 at load (rejected: doubles memory,
defeats the 24 GB goal); fork their kernels into vLLM C++ (deferred: today
the python package import is the cheapest correct seam); per-part vLLM
linears instead of fusing (their architecture does this; deferred — would
touch `qwen3_5.py`/GDN modules).

**Fused-layer finding (main design discovery of the day):** parts of a fused
vLLM linear carry **independently optimized, different `suh` vectors**
(measured qkv vs z: ~5 % apart). One shared `suh` — the assumption in
exllamav3's own fused forward — would skew outputs by that amount. The
backend therefore stores `suh` as `(in, pieces)` and dequantizes each fused
piece with its own column. The BC kernels accept a single `suh`, so the
fused decode path is disabled: fused layers run slab-dequant at every batch
size. Single-piece layers (o_proj, down_proj, out_proj) get the fast BC
path.

## §works

`tests/quantization/test_exl3.py` — 11 passed (7 CPU, 4 GPU, on the RTX 4090
Laptop, no perf claims):

- registry membership, manifest part resolution (plain/fused/passthrough),
  loud `NotImplementedError` for a quantized lm_head, K-mismatch rejection;
- loader placement on real checkpoint tensors: plain q_proj byte-exact,
  fused in_proj_qkvz (4 pieces, spans + single) byte-exact concat,
  row-parallel down_proj, bad mul1 refused;
- numerics: backend dequant vs `LinearEXL3.get_weight_tensor` (5e-3);
  BC decode path and slab prefill path vs materialized-W GEMM (1e-2, fp16
  order-of-operations); fused gate_up with differing suh vs per-part
  references (1.5e-2, bf16 GEMM vs fp16 reference).

Environment notes: exllamav3 v1.4.5 ext builds on this box only after a
one-line host shim in `exllamav3_ext/util.cuh` (CUDA 12.0's `cuda_fp16.h`
declares `__halves2half2` device-only; the libtorch/cpu `.cpp` files include
it with plain g++). Patch lives in the local clone (`.tmp/exl3/exllamav3`,
untracked); it is an upstream PR candidate. Also pre-existing on this CPU
venv: `get_quantization_config()` dies in the quark import chain
(`vllm.vllm_flash_attn` not compiled) for any method — unrelated to this
branch.

## §blocked / debts

1. **lm_head (K=6) refused** → full-engine load of this checkpoint fails at
   model build with an explicit error. Needs either an EXL3 ParallelLMHead
   path (trellis decode for a 248k×5120 projection) or a checkpoint variant
   with bf16 lm_head.
2. **Fused layers have no BC decode path** (per-part suh) — correctness
   kept, decode-speed debt. Options: per-part BC objects with contiguous
   trellis copies, or unfused per-part linears at the model level.
3. **`tensor_storage` manifest only read via `--quantization exl3`** (sidecar
   `quantization_config.json`); config.json's embedded quantization_config
   lacks the manifest, so auto-detection builds a config that then fails per
   layer. Fix: merge the sidecar json in `_parse_quant_hf_config` (small
   upstreamable patch).
4. **TP>1 and PP untested**; multi-part span pieces (e.g. `in_proj_qkv`
   covering shards 0-2) raise `NotImplementedError` under TP>1 — per-part
   placement is written for TP1 only. TP slicing math for single pieces is
   implemented but unverified.
5. **No kernel-free path**: both forward paths need the compiled
   `exllamav3.ext`. A pure-torch trellis decoder (sliding 16-bit window +
   procedural codebook, both now documented) would enable CPU tests and a
   real dequant-to-bf16 mode.
6. Engine-level integration (model load through `Qwen3_5` AutoWeightsLoader,
   GDN `conv1d`/`in_proj_ba` passthrough, mamba state) is not exercised yet.

## §next — concrete steps

1. lm_head: extend `Exl3LinearMethod` with an `embedding()`-style gather over
   dequantized vocab slabs, or materialize bf16 lm_head at load
   (`vllm/model_executor/layers/quantization/exl3.py`, `get_quant_method`).
2. Manifest sidecar merge: `vllm/config/model.py`
   (`_parse_quant_hf_config`) + `vllm/transformers_utils/config.py`.
3. First end-to-end layer-level load through the real
   `Qwen3_5DecoderLayer` (needs 1+2), then a 2-layer mini-model smoke on the
   laptop GPU.
4. Fused decode path: per-part BC (measure VRAM cost of contiguous trellis
   copies) or model-level unfusing in `qwen3_5.py`.
5. TP2 test on the laptop (both GPUs free? else stand) using the existing
   slicing math; then PP.
6. Pure-torch trellis decoder in `exl3.py` (format facts in §format are the
   spec) → CPU tests, bf16-materialize mode for debugging.
7. Upstream PRs: `util.cuh` host shim; per-layer `mul1_multiplier` handling
   (kernels should read the stored value, not assume the constant).

## Repro pointers

- Fixture fetcher: `.tmp/exl3/refetch.py` (HTTP-range tensor extraction;
  safetensors `data_offsets` are relative to the data section, i.e. absolute
  offset = `8 + header_len + data_offset` — this cost an hour today).
- Oracle probe: `.tmp/exl3/` scripts; ext build log `/tmp/exl3_ext_build2.log`.
- Tests: `EXL3_FIXTURE_DIR=<...> pytest tests/quantization/test_exl3.py`.
