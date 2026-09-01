# NVFP4 KV numeric results

Command: `/home/user/e233-venv-pinned-main/bin/python tests/v1/attention/test_nvfp4_kv_numeric.py`

| Test | Measured accuracy metrics | Status | Python | Torch | Triton | GPU | Date |
|---|---|---|---|---|---|---|---|
| `test_bytes_per_token` | 144 bytes/token/head; 4.5 bits/element | PASS | 3.13.11 | 2.11.0+cu129 | 3.6.0 | NVIDIA GeForce RTX 4090 Laptop GPU | 2026-09-01 |
| `test_layout_roundtrip_views` | data bytes 0:128; scale bytes 128:144; zero-copy views | PASS | 3.13.11 | 2.11.0+cu129 | 3.6.0 | NVIDIA GeForce RTX 4090 Laptop GPU | 2026-09-01 |
| `test_reference_roundtrip_rmse` | Normal rel-RMSE: 0.09509 K/4096, 0.09509 V/4096, 0.09512 K/8192, 0.09508 V/8192. Group-outlier rel-RMSE: 0.09540 K/4096, 0.09510 V/4096, 0.09520 K/8192, 0.09526 V/8192 | PASS, regression guard rel-RMSE < 0.12 | 3.13.11 | 2.11.0+cu129 | 3.6.0 | NVIDIA GeForce RTX 4090 Laptop GPU | 2026-09-01 |
| `test_triton_store_matches_reference` | Packed bytes equal; scale bytes equal; gather K/V max-abs error 0 | PASS | 3.13.11 | 2.11.0+cu129 | 3.6.0 | NVIDIA GeForce RTX 4090 Laptop GPU | 2026-09-01 |
| `test_attention_output_vs_fp16` | fp16 comparison: cosine 0.9914, rel-L2 0.1309. Kernel vs reference-dequantized KV: rel-L2 1.071e-06 | PASS, regression guards cosine > 0.985, fp16 rel-L2 < 0.18, and kernel rel-L2 < 0.005 | 3.13.11 | 2.11.0+cu129 | 3.6.0 | NVIDIA GeForce RTX 4090 Laptop GPU | 2026-09-01 |
| `test_scale_zero_guard` | Reference and Triton outputs finite; zero groups exact | PASS | 3.13.11 | 2.11.0+cu129 | 3.6.0 | NVIDIA GeForce RTX 4090 Laptop GPU | 2026-09-01 |

## Threshold rationale

The measured synthetic-Gaussian noise floor comes from the NVFP4 scheme itself:
no Hadamard rotation, one per-16 E4M3 scale, and no FP32 secondary scale. The
thresholds are implementation regression guards, not quality claims. They leave
headroom above the observed relative RMSE of 0.09508 to 0.09540 and the observed
attention cosine of 0.9914 with relative L2 error 0.1309. The kernel-isolation
guard remains 0.005; the measured error is 1.071e-06.

Provenance: semantics lifted from MiaAI-Lab/exllamav3 `cache/nvfp4.py`; Triton
store verified bit-identical to the reference.
