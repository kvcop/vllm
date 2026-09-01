# Fork differences

This branch is the source anchor for the internal Qwen3.8 vLLM runtime. The
fork exists so reviewed runtime changes are built from Git commits instead of
being applied directly inside virtual environments.

## Exact upstream base

- Upstream repository: `https://github.com/vllm-project/vllm.git`.
- Base version: vLLM `0.27.1`.
- Base commit: `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`.
- Local integration branch: `qwen38-v0271-fork`.

At the time this document was created, the branch contains the exact upstream
base plus this fork contract. The runtime patches listed below still live in
the sibling `vllm_benchmark` evidence repository and must be imported as
reviewable commits before this branch replaces the existing patched virtual
environments.

## Why the fork is required

The RND Qwen3.8 deployment combines NVFP4 weights, W4A8 activation kernels on
SM89, FP8 KV cache, multimodal Anthropic-shaped agent traffic, speculative
decoding and guarded native CPU KV offload. No released upstream revision used
by the stand contains that complete, validated combination.

The fork provides:

- one reproducible source revision for runtime images;
- explicit separation between production candidates and disposable
  experiments;
- focused tests and review history for each retained delta;
- a clear path for deleting local changes after equivalent upstream fixes land.

## Planned maintained runtime delta

These changes are candidates for the primary Qwen3.8 runtime line. Import each
as its own commit with its existing focused tests and provenance.

| Change | Source evidence | Intended status |
| --- | --- | --- |
| Native CPU KV-offload mmap cleanup adapted from upstream PR `#52596` | `../vllm_benchmark/specs/007-qwen38-model-evaluation/ops/e339-vllm-pr52596-unlink-after-rendezvous.patch` | Primary offload runtime delta until an admitted upstream version contains the lifecycle fix. |
| Recurrent-group native-offload load-boundary fix | Upstream PR `#52807`, source commit `9df8d8dc543f597160250550db6b1688570aca96`; locally adapted regression test covers the vLLM `0.27.1` scheduler shape. | Required while the upstream PR remains open. It prevents a legitimate unhashed GDN/Mamba state below the computed prefix from truncating the CPU-to-GPU load destination range and raising the fatal `num_locally_computed_tokens` assertion. |
| SM89 NVFP4 W4A8 kernel and mixed-layer selection | `../vllm_benchmark/specs/007-qwen38-model-evaluation/ops/e348-w4a8-sm89-candidate/` | Supervised Qwen3.8 candidate. Only eligible NVFP4 MLP layers use W4A8; FP8 GDN projections and the NVFP4 `lm_head` retain their Marlin routes. |
| ModelOpt mixed FP8-block handling required by the measured checkpoint | `../vllm_benchmark/specs/007-qwen38-model-evaluation/ops/e348-nvfp4-mtp-fp8-candidate/` | Import only with the adjacent W4A8/FP8 routing tests. |
| Per-request complete prefix-cache exclusion | `../vllm_benchmark/specs/007-qwen38-model-evaluation/ops/e356-echo-no-cache/` | Planned service-isolation feature for short-lived Echo requests; not enabled globally. |
| Opt-in per-stage pipeline-parallel timing (`VLLM_PP_STAGE_TIMING`) | `../vllm_benchmark/specs/007-qwen38-model-evaluation/DAY_TASKS_2026-08-31.md` item 2; measured PP=2 x TP=2 stage skew of 90-92% against 60-62% GPU busy. | Diagnostic delta, default off and a no-op when unset. It separates per-stage compute from time blocked in the PP recv, which raw GPU-busy percentages cannot distinguish. Retain until the stage skew is explained and closed. |
| Privacy-safe request-access KV events | `kv_connector_extra_config["request_access_events"]`, together with global KV event publishing. | Diagnostic delta for counterfactual LRU/ARC replay. It emits one ordered opaque-hash vector per request and KV group, with engine/DP/monotonic-sequence identity but no request IDs, token IDs, prompts or media. Disabled by default. |
| Qwen3.5 native MTP with a pipeline-parallel target | Upstream PRs `#46994`, `#52179` and `#52117`, ported to the `0.27.1` V2 runner with focused model, PP-relay and scheduler tests. | The draft parallel config is a truthful single-stage `PP=1` config while its TP width remains unchanged. The V2 runner owns the complete draft on the last target PP rank. Fixed-width sampled-token broadcast, proposed-draft relay and base-scheduler cadence keep every target stage on the same tokens under synchronous PP execution. Multimodal drafting skips embedding gather on later PP ranks, where no encoder runner exists. |
| Auxiliary hidden-state relay across pipeline stages | Local fork commit and focused `test_qwen3_next_aux_pp_relay.py` coverage. | EAGLE-3/DFlash/DSpark drafters read hidden states tapped from layers spread over the whole target, but the drafter is built only on the last PP rank. Each stage now taps its own layers and hands them to the next one inside the `IntermediateTensors` relay, keyed by global layer id, so the last stage reassembles the full ascending feature vector. Replaces the V2 runner's blanket refusal of these methods under PP with a per-architecture capability gate. Not yet exercised on hardware: no PP stand run has executed it. |

The request-access trace is effective only when both global KV event publishing
and `request_access_events` are enabled. It emits one `RequestAccess` event for
every KV group on the request's first offload lookup pass, including groups with
an empty hash vector. `request_seq` is gap-free among emitted accesses for one
`engine_id`; a cache reset does not renumber or duplicate an access, while an
engine restart changes the identity and starts a new sequence.

The Qwen3.8 chat template is intentionally tracked by the deployment repository
as an explicit asset. It should be moved into this fork only if the runtime
image, rather than the service bundle, becomes its owner.

## Maintained DFlash2/lookup stack

The accepted DFlash2/lookup stack is committed in `qwen38-v0271-fork`. It is
still opt-in: the V2 runner selects it only for a draft checkpoint whose
architecture is `DFlash2DraftModel`. Keeping the implementation in this branch
means a source-file replacement shows up as a Git diff instead of erasing an
ad-hoc-only patch with no durable reference.

The maintained source set is:

- the frozen E336/E343 DFlash2 model, selector, lookup, sampling and runner
  integration;
- the E345 rejection-sampler allocation fix;
- the native-MTP PP draft relay, auxiliary-hidden-state relay and PP memory
  warmup required by the combined PP target.

Source bundles live under
`../vllm_benchmark/specs/007-qwen38-model-evaluation/ops/` as
`e341-*`, `e343-*` and `e345-dflash-sampler-memory-v0271.patch`.

The accepted source commits were `d9cc8381ad` and `242f1e2c8b` on
`qwen38-v0271-dflash-pp2-e358`. They are preserved on this branch as
`d0b986c9f0` and `d4036db3e1`. The integration deliberately excludes the
optional E343 split-KV attention backend and speed-knob registry. The selected
`k7`/FP8-KV profile cannot use the bf16-only attention path and does not require
those compile-cache controls.

## Experiments that are not default fork behavior

The following work is retained as evidence or an optional research branch. It
must not be presented as part of the production runtime without a new selection
decision:

- E352 native-MTP suffix tail-fill;
- E353 lookup threshold and history tuning, whose tested design was rejected
  for the Monday pilot;
- GDN metadata synchronization experiments;
- speculative-decode attention and graph diagnostics not required by a selected
  profile;
- model-loader route and GPU-memory audit instrumentation;
- DSpark confidence scheduling, fixed-K graph experiments and unrelated W8A8
  tuner branches from the older local fork.

## Verification contract

A fork branch is deployable only when its build records:

1. the exact upstream base and fork commit;
2. focused unit tests for every retained delta;
3. kernel-route evidence for W4A8 MLP, FP8 GDN and NVFP4 `lm_head` when W4A8 is
   enabled;
4. text, tool, substantive-image and long-context smokes;
5. CPU-offload lifecycle and stale-mmap checks when offload is enabled;
6. model revision, vLLM launch configuration, hardware topology and load
   conditions for every benchmark claim.

## Upstream and GitLab policy

The future internal RND GitLab repository is the writable `origin`. Public
vLLM remains the read-only `upstream` remote. Do not push local changes to the
public upstream project or build production images from an uncommitted virtual
environment. When an admitted upstream release supersedes a delta, remove that
delta in a dedicated commit after adjacent runtime validation.
