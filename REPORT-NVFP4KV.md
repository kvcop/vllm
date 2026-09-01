# NVFP4 KV Cache Port — Track 2 of the EXL3 Program

Date: 2026-09-01 · Branch: `nvfp4-kv-cache` (base `qwen38-v0271-fork` @ `963f60ced2`) · Scope: provenance, portability memo, port STARTED with first numeric tests. Laptop RTX 4090 16GB (SM89) only; no stand access; no performance measurements or claims anywhere in this file. Placed as `REPORT-NVFP4KV.md` because root `REPORT.md` is the inherited stage-safe CPU KV offload report.

## 1. Provenance verdict

**NVFP4-KV (E2M1 values + per-16 E4M3 block scales, online dequant in Triton paged attention) is a MiaAI-Lab fork addition. Upstream turboderp-org/exllamav3 does not have it — neither at the 2026-08-24 snapshot nor at v1.4.5.**

Refs examined (full clones under `vllm_benchmark/.tmp/exl3-provenance/`):

| Tree | Ref | Date |
|---|---|---|
| turboderp 2026-08-24 | `080bb8c7e0c610881f4cb85b2f1bdf95666a1bea` | 2026-08-24T17:11:20Z |
| turboderp v1.4.5 | `e648f1a131365aae15920073e761a3fa5a527654` | 2026-08-31 |
| MiaAI-Lab HEAD | `63b32f001d7b2cfed3b3e3aaf25f534ba53cc7ed` | 2 commits: squash `e308b7d` + `serve_openai.py` follow-up |

Evidence: NVFP4 lives only in the fork — `exllamav3/cache/nvfp4.py` (`nvfp4_quantize`, `nvfp4_dequantize`, `CacheLayer_nvfp4`), NVFP4 kernels in `exllamav3/modules/attention_fn/triton_paged.py` (`_nvfp4_encode`, `_paged_kv_update_nvfp4_kernel`, `_nvfp4_decode`, `_nvfp4_kt_tile`, `_nvfp4_v_tile`), routing in `attention_fn/dispatch.py`, selection `cache_quant == "nvfp4"` in `model_init.py`. Upstream has only the generic H32 2–8-bit quantized KV (`cache/quant.py` + `exllamav3_ext/cache/q_cache*`); those files are byte-identical between the two upstream refs (`cache/quant.py` SHA-256 `7c1cce97…` identical in all three trees), so upstream's KV-quant path did not change across the window and never gained NVFP4.

Corrections to the recon notes (kit docs vs code):
- There is **no NVFP4→Hadamard-4 fallback**. `had_4_inreg` in `q_cache_kernels.cuh` is an unconditional part of the generic H32 path, not an Ampere fallback for NVFP4. On Ampere the kit's alternative is simply the generic 2–8-bit path.
- `EXL3_QC_STAGING` (0 = in-kernel dequant, 1 = fp16-staging prefill default, 2 = debug full staging) applies **only to the generic `CacheLayer_quant`**. The NVFP4 path always runs fused append + online dequant and ignores it.

Code-trust audit (static, kernels/runtime/server): no telemetry, no hidden endpoints, no obfuscated code, no unexpected network/file I/O from kernels. Beyond the expected DFlash2/DSpark loaders, the fork adds: an unauthenticated `serve_openai.py` aiohttp server defaulting to `0.0.0.0` (LAN exposure if deployed as-is), `webconsole.py` with SSRF surfaces if exposed, `start.sh` that may rewrite the model's `config.json` at ctx > 262144, NanoChat loader and CPU value-embeddings modules. None of this affects porting the quant kernels.

## 2. Portability memo

### 2.1 Where it lives in exl3 (MiaAI fork)
- Store-side quant (torch reference): `cache/nvfp4.py::nvfp4_quantize` — per-16-group along head_dim: `scale = clamp(amax/6, max=448)` as E4M3; magnitude code by nearest-level thresholds `_E2M1_BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)` over `clamp(|x/scale_decoded|, 6)`, sign in bit 3, pack even|odd<<4.
- Store-side quant (Triton): `triton_paged.py::_paged_kv_update_nvfp4_kernel` — same arithmetic chain: `s16 = min((amax_f32/6).to(f16), 448)`, `s8 = s16.to(e4m3)`, encode with `s8.to(f16)` (the decoded scale). The exact rounding order matters for bit-identity; our port preserves it.
- Online dequant: Triton paged attention `_paged_attn_splitdv_kernel` with `NVFP4: tl.constexpr` + `k_scales`/`v_scales` pointers; tile loaders `_nvfp4_kt_tile`/`_nvfp4_v_tile` unpack nibbles and broadcast one E4M3 scale per 16 head_dim elements.
- Layout: per-layer paged, `PAGE_SIZE=256`, separate planes: K/V `uint8 [pages, 256, H, D/2]`, scales `e4m3 [pages, 256, H, D/16]`. Effective 4.5 bits/element.

### 2.2 The fork already has the NVFP4-KV plumbing — for SM100 only
This changes the port from "add a dtype" to "teach the Triton backend an existing layout":
- `CacheDType` already accepts `nvfp4` (`vllm/config/cache.py:19`).
- The allocator already budgets the packed layout via `nvfp4_kv_cache_full_dim = head_size//2 + head_size//16` (`vllm/utils/torch_utils.py:414`); at head_dim 256 that is 144 B/token/head (128 value + 16 scale) = 4.5 bits/element — same density as exl3.
- FlashInfer consumes it (`flashinfer.py:1837`, `nvfp4_split_data_scale` zero-copy views of one packed `[data|scale]` buffer, shape `(num_blocks, 2*num_kv_heads, block_size, full_dim)`), but admits `nvfp4` **only on SM100 trtllm-gen**; on SM89 it rejects the config outright.

### 2.3 Integration points (files that change for a full port)
1. `vllm/v1/attention/backends/triton_attn.py` — add `nvfp4` to `supported_kv_cache_dtypes` (:277), NVFP4 branch in `get_kv_cache_shape` (:322), plane split in `TritonAttentionImpl.forward` (:580) and cache update (:791).
2. `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py` — quantize-on-store branch (the leaf kernel on this branch already targets the exact fork layout).
3. `vllm/v1/attention/ops/triton_unified_attention.py` — online-dequant NVFP4 branch in `kernel_unified_attention` across its tiling variants (the hard part; see effort map).
4. No `CacheDType`, config, or allocator changes required.

### 2.4 Interactions and constraints
- **Spec-decode / DFlash**: DFlash requires a non-causal attention backend (`vllm/v1/spec_decode/dflash.py:301` assert). The FlashInfer NVFP4 path is causal-only, so a DFlash drafter cannot use it; the drafter already runs a separately-configurable backend/dtype (`SpeculativeConfig`, `llm_base_proposer.py:1292`), so a Triton non-causal NVFP4 variant (or a mixed fp8-drafter / nvfp4-target setup) is the path. DFlash also excludes PIECEWISE cudagraphs (full graph or eager — plain Triton kernels are graph-safe).
- **CPU KV offload (`OffloadingConnector`)**: the offload tier moves KV blocks as bytes and its paths assume per-dtype shapes; the packed 144 B/token/head layout changes block byte sizes. Must be explicitly gated (fail loud or verified byte-generic) before any offload profile sees `nvfp4`.
- **Hybrid GDN allocator**: Qwen3.8's 16 full-attention layers already form one uniform group sharing one pool (`kv_cache_utils.get_kv_cache_groups`); group-level dtype divergence must be re-verified through `unify_kv_cache_spec_page_size`.
- **Prefix cache**: store is deterministic per token, so shared prefix blocks stay bitwise consistent; cross-dtype block sharing (fp8↔nvfp4) must be forbidden.
- **SM89**: no native FP4 ALU — dequant happens in Triton registers. Decode attention is bandwidth-bound and the layout cuts KV bytes ~2.2x vs FP8, which is the (unproven, unmeasured-here) win mechanism. No perf claims until measured on the stand.
- **fp16-staging prefill**: exl3's staging trick belongs to its generic path; its NVFP4 path pays online dequant in prefill too (exl3 documents 6–26% kernel cost on the generic path for in-kernel expansion). A vLLM analog can be added later if prefill cost matters; not needed for correctness.

## 3. What runs today

Commits on `nvfp4-kv-cache`:
- `1f41b39a79` `[KV Cache][NVFP4] Add Triton NVFP4 KV store and online-dequant scaffold` — `vllm/v1/attention/ops/triton_nvfp4_kv.py` (574 LOC, standalone leaf: torch/triton only, no vllm imports; reference quantize/dequant, fork-layout allocator + zero-copy K/V data/scale views, Triton quantize-on-store, Triton gather-dequant, simple GQA paged-attention kernel with online dequant).
- `017081555a` `[Tests][KV Cache][NVFP4] Add standalone numeric round-trip and attention tests` — `tests/v1/attention/test_nvfp4_kv_numeric.py` (359 LOC, standalone runner + pytest-compatible) + measured evidence in `test_nvfp4_kv_numeric.results.md`.

Environment: python 3.13.11, torch 2.11.0+cu129, triton 3.6.0, RTX 4090 Laptop GPU. Real Qwen3.8-27B full-attention shapes (4 KV heads × head_dim 256, GQA 24q:4kv, block_size 16). **6/6 pass.**

| Measurement | Value |
|---|---|
| Cache density | 144 B/token/head = 4.5 bits/element (fp16 512 B → 3.6x denser; FP8 256 B → 1.78x denser) |
| Round-trip rel-RMSE vs fp16 (4096 & 8192 tokens, K & V, N(0,1) and 5%-group-outlier) | 0.0951 – 0.0954 |
| Triton store vs torch reference | bit-identical packed AND scale bytes; gather diff = 0 |
| Attention (single layer, 1024 ctx) NVFP4-KV vs fp16-KV | cos = 0.9914, rel-L2 = 0.1309 |
| Attention kernel-correctness isolation (Triton cache vs reference-dequant cache) | rel-L2 = 1.07e-06 |

Density note: ~1.8x context capacity per GB, not 2.0x; the "~2x" headline holds only vs the byte-counting that includes fp16 staging buffers.

Threshold rationale: the measured rel-RMSE ≈ 0.095 and cos 0.9914 are the **noise floor of the scheme itself** (E2M1, per-16 E4M3 scales, no rotation, no FP32 secondary scale) on synthetic Gaussian data — the port is bit-faithful to it. Test thresholds (rel-RMSE < 0.12, cos > 0.985, rel-L2 < 0.18, kernel isolation < 5e-3) are implementation regression guards, not quality claims.

## 4. Effort map to completion

Mechanical (1–2 focused sessions):
- Wire the leaf store kernel into `triton_reshape_and_cache_flash` behind a cache-dtype branch (addressing on the fork layout is already proven bit-exact).
- `supported_kv_cache_dtypes` + `get_kv_cache_shape` + plane-split in `TritonAttentionImpl.forward`.
- Move standalone tests into the repo's CI-shaped test layout.

Risky / multi-session:
- NVFP4 dequant branch across `triton_unified_attention` tiling variants (split-DV, BLOCK_N/M, GQA layouts, sliding window, sinks, spec-decode non-causal) — the largest single chunk.
- Hybrid GDN group dtype plumbing verification (`unify_kv_cache_spec_page_size`).
- CPU-offload interop: gate or byte-generic proof; fail-loud if unsupported.
- Numeric characterization on **real** Qwen3.8-27B activations (capture K/V of the 16 full-attention layers) + downstream eval (logprob-scoring corpus, McNemar vs the current FP8-KV baseline, per the established eval quality axis).
- Only after all of the above: stand measurements under the usual load-conditions contract.

## 5. Go/no-go

**Lean: conditional GO for continuing the Triton port; NO-GO for any serving or capacity claim yet.**

Decisive criterion (two gates):
1. Real-activation gate: single-layer attention-output error on real Qwen3.8-27B K/V must land materially better than the synthetic floor (target cos ≥ 0.995, rel-L2 ≤ 0.05) AND within ~1.5–2x of the FP8-KV-vs-fp16 error production already tolerates. If real activations reproduce the synthetic ~13% rel-L2 across 16 stacked full-attention layers, that is disqualifying without mitigation.
2. Downstream gate: logprob-scoring + McNemar divergence vs the current FP8-KV profile shows no significant quality regression before any profile switch (manual quality read outranks throughput, per standing decision).

Fallback lever if gate 1 fails: add a per-16 Hadamard-style rotation before quantization (upstream's generic H32 path exists precisely for this; rotation preserves density at the cost of a transform on store and in-kernel dequant). Secondary fallback: NVFP4 for V only, FP8 for K.

**Biggest risk: numeric quality on real activations** — the scheme's synthetic single-layer error is high, and 16 layers stack it; the whole prize (~1.8x context per GB) is worthless if Vladimir's quality read rejects it. Secondary risk: the `triton_unified_attention` variant surface and offload interop dragging the integration long.
