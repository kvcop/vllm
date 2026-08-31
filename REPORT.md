# Stage-safe native CPU KV offload under pipeline parallelism

Branch `pp-stage-safe-offload` on `origin` (github.com/kvcop/vllm), based on
`qwen38-v0271-fork` @ `53b873b9a0`.

- `338719c797` — `[Bugfix][KV Offload] Port unlink-after-rendezvous mmap lifecycle (#52596)`
- `dead119371` — `[Bugfix][KV Offload] Size the shared CPU tier from per-rank KV layouts under PP`

Worktree: `/home/user/code/work/rnd/llm/vllm-qwen38-v0271-ppstage`.
CPU code + tests only: no stand access, no systemd, no GPU, no performance
claims. Unblocks (source-wise) the arm `night-pp2-tp2-mtp-k4-kvfp8-offload56`
closed fail-closed by benchmark gate commit `66549fd`.

## 1. Problem

Under PP2 x TP2 the Qwen3.8 hybrid projection gives per-stage KV group
signatures `(8,8,8,8)` (stage 0) and `(8,8,8,9)` (stage 1, one extra native-MTP
full-attention layer). Each worker derives its mmap row width from its LOCAL
projected tensors, but all four workers rendezvous on one pathname
`/dev/shm/vllm_offload_{engine_id}.mmap`. With unequal stage views:

- whichever worker wins `O_EXCL` truncates the file to ITS stage's total; a
  peer with a larger layout times out in `_wait_for_file_size`, a peer with a
  smaller layout maps a prefix and interprets rows with the wrong stride —
  cross-stage aliasing of block storage;
- `_wait_for_file_size` accepts any size `>=` expected, so a stale larger file
  is attached silently;
- the scheduler-side `CPUOffloadingManager` is sized from stage 0's view
  (`world_size * S0` row) and can allocate block IDs beyond another stage's
  mapped rows.

## 2. Design (implemented)

**One shared mmap + registered per-rank slot-width table ("per-stage size
accounting").**

1. **Engine core registers the per-worker footprints.**
   `compute_per_worker_kv_bytes_per_block()` (`vllm/v1/core/kv_cache_utils.py:1855`)
   runs at the tail of `get_kv_cache_configs()` — after the num_blocks
   equalization, which preserves the per-block ratio — and stores
   `per_worker_kv_bytes_per_block` (rank order) on every worker's
   `KVCacheConfig` (`vllm/v1/kv_cache_interface.py:997`). `None` when single
   worker or homogeneous (TP-only keeps its previous config shape).
   `generate_scheduler_kv_cache_config()` deep-copies worker 0's config, so
   the scheduler carries the same table with zero new IPC. This is the only
   place in the engine where all workers' post-projection geometry exists.

2. **Connector validates and forwards it.**
   `_resolve_registered_worker_kv_bytes()` (`offloading/config.py:31`) fails
   closed when PP>1 has no registered layout, when the entry count differs
   from `world_size`, when `TP*PP != world_size`, or when TP peers inside one
   stage disagree. The tuple lands on
   `OffloadingParallelConfig.worker_kv_bytes_per_block_by_rank`
   (`vllm/v1/kv_offload/config.py:59`).

3. **CPU spec sizes one layout from the table, not from the local view.**
   `CPUOffloadingSpec._resolve_slot_chunk_widths()` (`cpu/spec.py:150`):
   row = `round_up(sum(per-rank chunk widths), PAGESIZE)`;
   `num_blocks = cpu_bytes // row` — identical on the scheduler and every
   worker (verified by test), so the manager can never allocate an ID that
   overflows another stage. Each process also verifies its LOCAL
   `worker_kv_bytes_per_block` equals its own table slot — refusal, not
   best-effort attach. Legacy arithmetic is bit-identical when the table is
   uniform or absent (PP=1): `sum([S*bpc]*W) == S*W*bpc`.

4. **Region places per-rank slots at prefix-sum offsets.**
   `SharedOffloadRegion(..., slot_page_sizes=...)` (`cpu/shared_offload_region.py:75-104`):
   slot for rank r spans `[sum(w[0..r)), +w[r])` inside each row; stride is the
   padded sum. Stage 0 ranks and stage 1 ranks get disjoint byte ranges — a
   block key resolves to its own stage's slot only. `create_next_view` cursor
   bounds now use the table. Uniform inputs keep the legacy two-value form
   exactly (`kv_bytes_per_block`, `cpu_page_size`).

5. **Fail closed on attach.**
   - Opener requires the file size to be EXACTLY its computed total
     (`shared_offload_region.py:120-131`): a stale or foreign-sized file is
     refused with both sizes in the message. (Peer with a LARGER registered
     table still hits the pre-existing `_wait_for_file_size` timeout — the
     second documented failure mode, now covered by a test.)
   - `create_worker` re-checks the device-derived slot against the table
     (`cpu/spec.py:225-243`): config-rank and device-rank must agree with the
     local footprint.
   - Region refuses a rank outside the table (`shared_offload_region.py:97`).

6. **#52596 lifecycle preserved and generalized.**
   The unlink-after-rendezvous port (commit `338719c797`) keeps the exact
   marker strings the E358 preflight greps for: `barrier:
   Callable[[], None] | None = None` and `Unlinked mmap file %s` in the
   region; `def _all_workers_barrier() -> None:` and
   `barrier=_all_workers_barrier` in the spec. The barrier runs after every
   mapping is established, creator unlinks afterwards, mappings stay usable;
   barrier failure closes fd/mapping and only the creator unlinks. The port
   was previously venv-only (e339 patch did not apply cleanly to this tree —
   the venv copy had drifted); it is now a reviewable commit.

### Alternative rejected: per-stage mmap regions

Per-stage files (`vllm_offload_{engine}_s{pp}.mmap`) would also give disjoint
storage and per-stage lifecycles, and the uniform assumption holds inside one
stage. Rejected because (a) the scheduler would still need the cross-stage
geometry for the shared block-ID capacity (`min_s N_s`), so the
engine-core registration is unavoidable either way; (b) the E358 post-start
gate's evidence shape — exactly ONE created mmap ~60.1 GB, three opens of the
same path, one unlink — survives unchanged, so the benchmark preflight needs
hash updates plus a pre-check swap, not a structural rewrite; (c) one region
keeps `pin_mmap_region` and the tiering memoryview contract untouched.

## 3. Assumption sites (uniform group view), before -> after

All "before" line numbers are `qwen38-v0271-fork` @ `53b873b9a0`.

| # | Site (before) | Assumption | Fix (after) |
| --- | --- | --- | --- |
| A1 | `vllm/v1/kv_offload/cpu/spec.py:91-110` | `kv_bytes_per_block = worker_kv_bytes_per_block * world_size` — local (stage-0) bytes treated as world-uniform; `num_blocks` from that single-view row | `cpu/spec.py:100-146,150-190` — row/`num_blocks` from the registered table; local view only used for verification |
| A2 | `vllm/v1/kv_offload/cpu/spec.py:149-167` | one slot per rank, uniform `cpu_page_size`, `rank = device_index % world_size` | `cpu/spec.py:213-256` — same slot identity, table-checked; table only passed when genuinely unequal |
| A3 | `vllm/v1/kv_offload/cpu/shared_offload_region.py:53-63` | `total_size` from local view; offsets `rank * cpu_page_size` | `shared_offload_region.py:75-104` — prefix-sum slot table, padded-sum stride |
| A4 | `vllm/v1/kv_offload/cpu/shared_offload_region.py:15-25` | `_wait_for_file_size` accepts `>=` (silent attach to wrong-size file) | `shared_offload_region.py:117-131` — exact-size requirement, refusal names both sizes |
| A5 | `vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py:98-111` | `worker_kv_bytes_per_block = total_gpu_kv_bytes // num_blocks` from the local projection (gate-pinned marker) | unchanged derivation; consumed via `_resolve_registered_worker_kv_bytes` (`config.py:31-84`) |
| A6 | `vllm/v1/core/kv_cache_utils.py:1855-1874` + `vllm/v1/engine/core.py:315` | scheduler config = deepcopy of worker 0 ("arbitrary"), so scheduler-side capacity uses stage 0's view | registration now rides the same deepcopy: attach at `kv_cache_utils.py:2271-2275`, field at `kv_cache_interface.py:993-997` |

Out of scope (unchanged, documented upstream of this work): the tiering FS
tier consumes `layer_names` through `FileMapper` and hashes a per-stage base
path under PP (LIVE_PLAN `E358-NIGHT-...` tiering note); native offload does
not use that path.

## 4. Gate `66549fd` evidence requirements

| Gate requirement (from the commit + handoff §2a) | Status on this branch |
| --- | --- |
| Scheduler `#52807` hash `5ce2518d…` | unchanged (`5ce2518dc021e89a61b478d650d5f5b2d6dae5f2aa031bfe8b5b33a5b255eecd`) — the port was already in the fork tree |
| Region + spec carry the `#52596` lifecycle markers | verified verbatim (4/4 markers, see §2.6); the lifecycle is now a fork commit, not a venv overlay |
| Equal stage group signatures before sharing one mmap | superseded constructively: unequal signatures are now SAFE because every worker derives stride/offsets from one registered table; the equal-signature check should be REPLACED in the preflight by "registered layout present + validated" (see §7 checklist item P3) |
| Exact source hashes for the offload contract | new values in §6 — preflight must be updated before any arm boots this runtime |
| Stale-mmap and `64,000,000,000`-byte SHM prechecks | unchanged semantics; the runtime additionally self-defends (exact-size refusal) against a stale file that slips past the precheck |
| Post-start: one ~60.1 GB mmap, three opens of the same path, one unlink after rendezvous | layout preserves the single-region shape: total = `num_blocks * round_up(sum(widths))` still consumes the same `cpu_bytes_to_use` (56 GiB -> ~60.1 GB decimal row total unchanged: row = padded true full-model row instead of the naive stage-0 row, `num_blocks` adjusted accordingly); the existing `gate_native_offload_region` regexes still match `Created/Opened/Unlinked mmap file` |
| "Do not clear the offload arguments" (no silent no-offload fallback) | respected: PP>1 without a registered layout raises — no fallback path was added anywhere |

## 5. Test evidence

CPU env: night-executor venv (py3.12.3, torch 2.13.0+cu130 as CPU) +
`PYTHONPATH=<worktree>`; local shim `cpu_noop_cleanup_plugin.py` (untracked,
copied from `/tmp/day31-verify`) no-ops the CUDA cleanup fixture.

```
PYTHONPATH=$PWD pytest tests/v1/kv_offload/ -q -p no:cacheprovider -p cpu_noop_cleanup_plugin
  base 53b873b9a0: 503 passed, 1 skipped
  this branch:      523 passed, 1 skipped   (+3 factory, +17 new file)

PYTHONPATH=$PWD pytest tests/v1/core/ -q … --ignore={test_async_scheduler,test_kv_sharing,test_deferred_block_free}.py --continue-on-collection-errors
  this branch:     49 failed, 899 passed, 3 errors
  clean day31-verify tree, same env: 49 failed, 376 passed(shard), 3 errors
  -> diff of FAILED lists: IDENTICAL_FAILURE_SETS (env: LLaVA model
     inspection, vllm_flash_attn CUDA import, kernel_warmup import; not the diff)

ruff format --check + ruff check on all touched files: clean
mypy on the 5 touched source files: Success, no issues
```

New coverage in `tests/v1/kv_offload/cpu/test_pp_stage_safe_offload.py`
(+ `tests/v1/kv_offload/test_factory.py` additions):

- heterogeneous round-trip per stage: 4 ranks, widths (2P,2P,2.5P+64,…), each
  rank writes/reads its slot; raw-mmap assertion that no byte lands outside a
  rank's own slot and row-tail padding stays zero;
- sizing: all ranks + scheduler resolve identical `num_blocks`/stride from the
  table; uniform table reproduces legacy layout byte-for-byte (legacy form vs
  table form, same offsets/stride/total);
- refusal: PP without registered table; local-view drift vs own slot (config
  rank AND device rank); rank outside table; stale/foreign-sized file (both
  directions: smaller table -> exact-size RuntimeError; larger table ->
  `_wait_for_file_size` TimeoutError);
- lifecycle: barrier rendezvous unlinks the path while mappings stay shared;
  barrier failure closes mapping and only the creator unlinks;
- engine-core attach: homogeneous -> None (TP shape preserved), PP-shaped ->
  table on every config, scheduler deepcopy carries it.

## 6. New SHA-256 for the benchmark preflight (E358)

Update `PATCHED_OFFLOAD_SHA256_BY_PATH` / `AUDITED_PP_OFFLOAD_SHA256_BY_PATH`
in `scripts/stand/qwen38_e358_tp4_pilot_preflight.py` from this branch:

| path | old (pinned) | new (this branch) |
| --- | --- | --- |
| `distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py` | `5ce2518d…` | `5ce2518dc021e89a61b478d650d5f5b2d6dae5f2aa031bfe8b5b33a5b255eecd` (unchanged) |
| `v1/kv_offload/cpu/shared_offload_region.py` | `855635a0…` | `e8f9276bbb9a0b09682b37a5b273acca2e3381938827ce8cfc10f8a35cf925c7` |
| `v1/kv_offload/cpu/spec.py` | `f590d2f1…` | `9a5e2117f2fcaaa053922a325377756177b0455ff4acfb4872f5875b91667add` |
| `distributed/kv_transfer/kv_connector/v1/offloading/config.py` | `49a6aa9c…` | `9ed92e889559f76a60e6c8d1e3b6e8a1443ece842bf860fe5af482c62b8f56f7` |
| `v1/core/kv_cache_utils.py` | `0ab53859…` | `bd4f6f4e9d4f4db12ee99954a816568f287c151fa7c14f9b554ab6a0db9c9352` |
| `model_executor/models/qwen3_5.py` | `29a9e4c1…` | `d63f913d60196c422e711d70d81d067e4a6ac015a46e6ea316607c4d44835f64` (changed by fork `13b123c7c7`, NOT by this branch) |

New preflight marker suggestions (replace the equal-signature check): the
four `#52596` markers (already grepped today, still verbatim), plus
`per_worker_kv_bytes_per_block` present in `kv_cache_interface.py`,
`compute_per_worker_kv_bytes_per_block` in `kv_cache_utils.py`, and
`slot_page_sizes` in `shared_offload_region.py`.

## 7. GPU verification boot checklist (feeds the fail-closed preflight)

Environment gates before start (unchanged): queue drained, confirmed
`sudo qwen-stand stop` boundary, no stale `/dev/shm/vllm_offload_*.mmap`,
`>= 64,000,000,000` bytes free in `/dev/shm`, preflight hashes from §6.

- **P1 boot**: `night-pp2-tp2-mtp-k4-kvfp8-offload56` reaches
  `Application startup complete` on PP2 x TP2; all four workers attach without
  the exact-size refusal or a `_wait_for_file_size` timeout.
- **P2 journal**: exactly one `Created mmap file … (≈60.x GB)` (row total
  stays `cpu_bytes_to_use`-driven), three `Opened existing mmap file`, one
  `Unlinked mmap file` AFTER the last open (rendezvous order), zero
  `Removed mmap file` before rendezvous.
- **P3 preflight swap**: equal-signature requirement replaced by
  registered-layout validation (hashes §6 + markers); `_pp_stage_group_signatures`
  kept as informational output `(8,8,8,8)/(8,8,8,9)`, no longer a blocker.
- **P4 negative controls**: (a) restart with a hand-sized stale larger mmap →
  workers must refuse with the exact-size error, engine fails loudly;
  (b) mismatched runtime (one file from the old layout) → same refusal.
- **P5 smokes**: text + tool + substantive image on the exact profile, plus
  the local-argmax MTP marker — the E358 standard set.
- **P6 correctness under load**: verify stores/loads actually reconstruct
  per-stage layers — run a repeated-prefix workload and confirm CPU-tier hits
  return identical tokens vs the no-offload sibling (byte-identical responses
  on a frozen copy-shaped probe is the established bar).
- **P7 metrics**: `/metrics` offload gauges sane (no saturation growth), KV
  event trace shows CPU store/load per stage, no unexplained preemptions.
- **P8 recovery**: kill one worker mid-transfer → peers fail loudly (barrier
  timeout / executor teardown), no partial-garbage serving; restart clean.
- **P9 rollback**: `pilot-offload` (TP4 profile) still boots byte-compatibly
  on this runtime — uniform layout must be untouched.

## 8. Open risks

1. **Not GPU-verified.** All evidence is CPU-unit-level; P1-P9 above are the
   admission bar. No performance claim is made or implied.
2. **Barrier deadlock on asymmetric failure**: a worker that fails the new
   checks before mapping leaves peers blocked in the world-group barrier until
   the executor tears the engine down (pre-existing property of the #52596
   design, now more likely to fire on genuine mismatches — by intent: loud
   failure instead of corruption).
3. **`torch.accelerator.current_device_index() % world_size` as slot id** is
   the pre-existing TP4 production assumption; under PP2xTP2 each rank still
   owns a distinct GPU on the stand. If a future topology breaks the
   device==global-rank bijection, the create_worker check refuses rather than
   aliasing — but the refusal would look like a boot failure.
4. **`num_blocks` shrinks slightly** vs the old (unsafe) formula: the row is
   now the true padded full-model row; the stage-0 naive row understated it by
   stage 1's extra MTP layer (roughly 2 of ~130 per-layer shares, ~1.5%, for
   this geometry — exact figure depends on GDN/full per-layer byte mix).
   `cpu_bytes_to_use` consumption stays ~56 GiB.
5. **Group-count/tokens-per-block equality across stages** remains an
   assumption of the transfer machinery (already enforced at runtime by the
   worker handler's `group_sizes` assertion); the registration does not
   re-check it. Fine for the registered arm; a model whose per-stage
   projection changed GROUP COUNT would fail in the existing assert.
6. **`qwen3_5.py` hash drift** vs the audited pin predates this branch
   (fork `13b123c7c7`); preflight owners must decide whether to re-pin from
   this branch wholesale.
7. The exact-size opener check compares against `os.fstat` after the wait
   loop; a creator that legitimately truncates twice (never happens today:
   single `ftruncate` per creator) would trip it.

## 9. Reproduction

```
git worktree add <path> -b pp-stage-safe-offload qwen38-v0271-fork  # base
git fetch origin pp-stage-safe-offload && git checkout pp-stage-safe-offload
V=<cpu venv with torch+pytest>
PYTHONPATH=$PWD $V/bin/python -m pytest tests/v1/kv_offload/ -q \
  -p no:cacheprovider -p cpu_noop_cleanup_plugin
ruff format --check vllm/v1/kv_offload vllm/v1/core/kv_cache_utils.py
ruff check vllm/v1/kv_offload vllm/v1/core/kv_cache_utils.py
mypy vllm/v1/kv_offload/cpu/spec.py vllm/v1/kv_offload/cpu/shared_offload_region.py \
  vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py
```
