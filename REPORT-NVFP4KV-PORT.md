# NVFP4 KV Cache Port — Triton Backend Integration Report

Date: 2026-09-02 · Branch: `nvfp4-kv-cache` · Base of this pass: `bede82914d`
· Result: the port runs end-to-end on SM89 (RTX 4090 Laptop) through
`TRITON_ATTN`, including a real vLLM generation smoke. All numbers below are
laptop-local; nothing was measured on the stand, and no stand command was run.

## 1. What changed, per file

| File | Change |
|---|---|
| `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py` | `_reshape_cache_nvfp4_kernel` + `triton_reshape_and_cache_flash_nvfp4` wrapper: quantize-on-store for the fork's logical `(B, 2H, N, full_dim)` uint8 cache. Stride-aware (works on both HND-contiguous and NHD-permuted physical layouts); encoding chain is the leaf module's, shared via `_nvfp4_encode` so stored bytes stay bit-identical to `nvfp4_quantize_reference`. |
| `vllm/v1/attention/ops/triton_unified_attention.py` | `_load_kv_nvfp4_tiles` jit helper: loads the packed `[data | per-16 E4M3 scales]` planes and online-decodes to fp16 (shares `_nvfp4_decode` with the leaf module). Wired into the shared KV-tile load section of `kernel_unified_attention` behind `USE_NVFP4_KV` (`KV_QUANT_MODE == 5`), so every tiling variant (prefill 2D, mixed/chunked prefill, decode 2D, decode 3D segments, sliding window, sinks, alibi paths) inherits one dequant path. `USE_TD` is rejected (static assert + wrapper `ValueError`). Wrapper validates uint8 planes, `head_size % 32 == 0`, data-plane width `head_size//2`, contiguous packed dim. |
| `vllm/v1/attention/backends/triton_attn.py` | `nvfp4` admitted in `supported_kv_cache_dtypes`; `get_kv_cache_shape` returns `(num_blocks, 2*num_kv_heads, block_size, full_dim)` (same physical layout FlashInfer consumes on SM100); `TritonAttentionImpl.__init__` fails loudly below SM89 (fp8e4nv conversions); `forward` splits the K/V head halves and passes the data planes; `do_kv_cache_update` routes to the NVFP4 store wrapper; fused rope+store (`do_rope_and_kv_cache_update`) raises `NotImplementedError` for nvfp4 and `fused_rope_kvcache_supported()` returns False. |
| `vllm/config/vllm.py` | `_post_init_kv_transfer_config` raises if CPU KV offloading is combined with `cache_dtype == "nvfp4"` (fail-closed per REPORT §2.4 until the offload tier is proven byte-generic for the packed layout). This is the one config-layer change; no `CacheDType`, allocator, or spec changes were needed (`KVQuantMode.NVFP4`, uint8 spec dtype, and `nvfp4_kv_cache_full_dim` budgeting already existed). |
| `tests/v1/attention/test_nvfp4_kv_triton_backend.py` | Integration matrix (see §2). Standalone-runnable + pytest-discoverable. |
| `tests/v1/attention/e2e_nvfp4_kv_smoke.py` | E2E generation smoke (see §3). |
| `tests/v1/attention/bench_nvfp4_kv_decode.py` | Decode-step benchmark vs fp8/fp16 (see §4). |

Commits: `36c416e9a4` (ops), `a840a77f24` (backend admission), `3c9ad398f8`
(test matrix), `243b53a392` (e2e smoke), plus this report.

## 2. Numeric test matrix (laptop, RTX 4090 Laptop, SM89)

Environment: python 3.13.11, torch 2.11.0+cu129, triton 3.6.0, CUDA 12.9,
fork code from this branch. The vLLM C extensions (`_C_stable_libtorch`,
`_vllm_fa2_C/_vllm_fa3_C`, `flash_mla_interface.py`) are untracked build
artifacts borrowed from the `/home/user/e234-vllm-sp-rs-cu129-src` tree
(built against the same torch 2.11.0+cu129) because the worktree's own
cu130-built `.so` had disappeared (see memory `laptop-vllm-import-stale-so`);
the python code under test is 100 % from this branch.

### Store / round-trip (real Qwen3.8 full-attention shape: 4 kv heads × head_dim 256, block 16)

| Check | Result |
|---|---|
| Store bytes vs torch reference, HND layout | data and scale bytes identical (K and V) |
| Store bytes vs torch reference, NHD layout | data and scale bytes identical (K and V) |
| Round-trip rel-RMSE vs fp16 (8192 tokens) | < 0.12 guard, matches 0.095 scheme floor |
| Leaf standalone tests (commit `017081555a`, 6 tests) | 6/6 PASS (unchanged) |
| New integration tests (9: store, round-trip, 6 attention variants, admission) | 9/9 PASS |

### Single-layer attention through the real `kernel_unified_attention`

Same synthetic K/V (N(0,1)) quantized three ways; 24q : 4kv, head 256,
block 16; greedy-shaped metadata per variant; fp8 is per-tensor e4m3 with
amax/448 scales (the production FP8-KV shape on this fork).

| Variant | nvfp4 vs fp16 cos | nvfp4 vs fp16 rel-L2 | fp8 vs fp16 cos | fp8 vs fp16 rel-L2 | nvfp4 vs fp8 rel-L2 | kernel iso (nvfp4 vs dequant-cache) |
|---|---|---|---|---|---|---|
| prefill 2D (512 q) | 0.9923 | 0.1242 | 0.9994 | 0.0351 | 0.1301 | 1.8e-05 |
| mixed 2D (768 ctx + 256 q) | 0.9910 | 0.1342 | 0.9993 | 0.0380 | 0.1388 | 3.5e-05 |
| decode 2D (8×1024 ctx) | 0.9907 | 0.1365 | 0.9993 | 0.0376 | 0.1407 | 3.8e-05 |
| decode 3D (16 segments) | 0.9908 | 0.1353 | 0.9993 | 0.0381 | 0.1402 | 3.3e-05 |
| decode SWA (window 128) | 0.9912 | 0.1331 | 0.9993 | 0.0374 | 0.1394 | 3.2e-05 |
| decode + sinks | 0.9910 | 0.1338 | 0.9993 | 0.0381 | 0.1388 | 3.5e-05 |

Reading: the Triton port is numerically faithful to the scheme (kernel
isolation ~1e-5, far under the 5e-3 guard); the residual vs fp16 is the
scheme's synthetic noise floor (~0.13 rel-L2), unchanged from the leaf
measurements. On synthetic data the nvfp4 error is ~3.5× the fp8 error.

## 3. End-to-end generation smoke

Model `Qwen/Qwen3-0.6B` (downloaded; no cached small checkpoint had weights),
`--attention-backend TRITON_ATTN`, fp16 weights, greedy, 48 new tokens,
top-1 logprobs, 20 fixed raw prompts, `enforce_eager=True`,
`VLLM_USE_FLASHINFER_SAMPLER=0` (the venv's flashinfer 0.6.13 JIT fails to
build its sampling op on this machine — unrelated to the port).

| Comparison | Exact token match | mean abs Δtop-1-logprob | max abs Δ |
|---|---|---|---|
| auto vs auto (rerun, determinism control) | 20/20 | 0.000000 | 0.000000 |
| fp8 vs auto | 3/20 | 0.299 | 2.913 |
| nvfp4 vs auto | 0/20 | 0.476 | 3.008 |

All sequences ran to the 48-token cap; generation stays coherent under
nvfp4. Determinism control is exactly zero, so the deltas above come from
the KV dtype alone. Note that greedy exact-match is a chaotic metric here
(fp8 itself only matches 3/20); at the logprob level nvfp4's mean delta is
~1.6× fp8's on this model — better than the synthetic single-layer 3.5×
ratio, but this is a 0.6 B model with 48-token continuations, not the
real-activation gate.

## 4. Decode-step kernel time vs fp8 KV (laptop-synthetic)

`triton.testing.do_bench` around one `unified_attention` decode call (8
seqs, causal, 24q : 4kv, head 256, block 16, decode tile; includes the
python wrapper for all dtypes equally). RTX 4090 Laptop GPU, torch
2.11.0+cu129, triton 3.6.0. **Laptop-synthetic; not a stand claim.**

| Shape | fp16 | fp8 | nvfp4 | nvfp4/fp8 | nvfp4/fp16 |
|---|---|---|---|---|---|
| 8 seqs × ctx 1024 | 145 µs | 120 µs | 217 µs | 1.82× | 1.49× |
| 8 seqs × ctx 4096 | 385 µs | 342 µs | 696 µs | 2.04× | 1.81× |
| 8 seqs × ctx 16384 | 1497 µs | 1294 µs | 2554 µs | 1.97× | 1.71× |
| 64 seqs × ctx 4096 (occupancy probe) | 2417 µs | 1760 µs | 2508 µs | 1.43× | 1.04× |
| 32 seqs × ctx 8192 (occupancy probe) | 2283 µs | 1824 µs | 2538 µs | 1.39× | 1.11× |

**The first Triton implementation is slower than fp8 per decode step at
every tested shape (1.4–2.0×), and roughly at parity with fp16 at high
occupancy.** Diagnosis: at 8 seqs the kernel launches only 32 CTAs and is
latency-bound, so the 2.2× byte cut cannot show up (fp8 is itself only
0.73× of fp16, not the 0.5× a bandwidth-bound kernel would give); the
nibble unpack + scale broadcast + interleave shuffles add ALU work per
tile that dominates once bytes stop mattering. Levers not yet pulled:
NVFP4-specific `TILE_SIZE` (the shared 32 is tuned for unpacked tiles),
vectorized 32-bit unpacking, decoding K directly in transposed layout, and
a prefill staging pass. The capacity win (1.78× tokens per GB vs fp8)
stands regardless.

## 5. Go/no-go gates (from REPORT-NVFP4KV.md §5), restated

1. **Real-activation gate** — target: single-layer attention error on real
   Qwen3.8-27B K/V materially better than the synthetic floor (cos ≥ 0.995,
   rel-L2 ≤ 0.05) and within ~1.5–2× of FP8-KV error. **Still open; my
   numbers do not close it.** What they do say: the port adds no error
   beyond the scheme (isolation ~1e-5); on synthetic single-layer data the
   scheme sits at ~3.5× fp8's error (cos 0.991 vs 0.999), and at the e2e
   logprob level on Qwen3-0.6B it is ~1.6× fp8's. Real K/V distributions
   (outlier structure, larger head-count interactions) remain unmeasured —
   capturing the 16 full-attention layers' K/V on the stand and rerunning
   this matrix is the next concrete step. A new, second-order concern from
   §4: even if quality passes, the current Triton decode path gives back
   ~1.4–2.0× latency vs fp8, so the prize is capacity, not speed, until
   the kernel is tuned.
2. **Downstream gate** — logprob-scoring + McNemar vs the FP8-KV profile,
   no significant regression, before any profile switch; manual quality
   read outranks throughput. Unchanged, unmeasured here.

## 6. Open risks

- **Decode latency regression** (§4): first-implementation kernel is
  slower than fp8; tuning levers listed above are unpulled.
- **Real-activation quality** unmeasured; 16 stacked layers × ~0.13
  synthetic rel-L2 is the standing disqualification risk (REPORT §5).
- **CPU KV offload** is now fail-closed rather than verified: any offload
  profile + nvfp4 combination refuses to start until the tier is proven
  byte-generic for the packed layout.
- **DFlash/spec-decode interop** untested in e2e: DFlash needs non-causal
  attention (Triton supports it, but the non-causal NVFP4 branch has no
  test here), and drafter/target dtype mixing is config-level only.
- **Prefix caching**: cross-dtype block sharing is structurally impossible
  (different layouts), but I did not add an explicit guard against a
  hash-collision-style reuse across dtype changes within one process
  lifetime; vLLM keys blocks by content hash per engine run, so this
  requires an engine restart with a reused prefix-cache persistent store
  to matter (not a current stand feature).
- **Borrowed build artifacts** on the laptop (see §2 environment note):
  e2e used prebuilt C extensions from a dev2240 tree; ops exercised by the
  smoke ran fine, but a clean rebuild of the worktree against torch
  2.11.0+cu129 is advisable before any future laptop work beyond this
  scope.

## 7. Stand quality-gate boot command (NOT RUN)

On `chat.ct-sg.ru`, in the patched 0.27.1 runtime venv, with the queue
drained and every managed unit stopped (`sudo qwen-stand stop --yes`), the
E349-G-shaped TP4 profile with NVFP4 KV (quality-gate form — TRITON_ATTN,
since FlashInfer rejects nvfp4 below SM100):

```bash
/home/user/venv_vllm_v0271_pr52596/bin/python -m vllm.entrypoints.openai.api_server \
  --model RadixArk/Qwen3.8-27B-NVFP4@319f741c \
  --served-model-name Qwen/Qwen3.8-27B \
  --tensor-parallel-size 4 \
  --port 8003 \
  --max-model-len 262144 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.92 \
  --kv-cache-dtype nvfp4 \
  --attention-backend TRITON_ATTN
```

Proper form per repo policy is a tracked
`qwen38-*-nvfp4kv.candidate.service` with hash-compared install and the
usual post-start gates (`/v1/models` returns `Qwen/Qwen3.8-27B`, text/tool
smokes), before any quality-axis corpus run; the ad-hoc CLI above is the
semantic content of that unit. `pilot-offload` remains the restore target.
