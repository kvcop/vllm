# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
from typing_extensions import override

from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingCounterMetadata,
    OffloadingGaugeMetadata,
    OffloadingHistogramMetadata,
    OffloadingManager,
    OffloadingMetricMetadata,
    OffloadingSpec,
    OffloadingWorker,
)
from vllm.v1.kv_offload.config import OffloadingConfig
from vllm.v1.kv_offload.cpu.common import CPUOffloadingMetrics
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion


def _all_workers_barrier() -> None:
    """Wait until every worker rank has mapped the shared offload region."""
    from vllm.distributed.parallel_state import (
        get_inner_dp_world_group,
        get_world_group,
    )

    try:
        group = get_inner_dp_world_group()
    except AssertionError:
        group = get_world_group()
    group.barrier()


class CPUOffloadingSpec(OffloadingSpec):
    BLOCK_SIZE_ALIGNMENT = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT

    @classmethod
    def build_metric_definitions(
        cls, extra_config: dict[str, Any]
    ) -> dict[str, OffloadingMetricMetadata]:
        definitions: dict[str, OffloadingMetricMetadata] = {
            CPUOffloadingMetrics.CPU_CACHE_USAGE_PERC: OffloadingGaugeMetadata(
                documentation=(
                    "Fraction of CPU KV-cache space currently pinned by active "
                    "transfers (0.0 = idle, 1.0 = saturated). Sustained high "
                    "values indicate transfers (stores or promotions) may be "
                    "dropped due to insufficient capacity."
                ),
            ),
            CPUOffloadingMetrics.CPU_CACHE_WRITE_USAGE_PERC: OffloadingGaugeMetadata(
                documentation=(
                    "Fraction of CPU KV-cache space currently pinned by "
                    "in-flight stores that have not yet "
                    "completed (0.0 = idle, 1.0 = saturated)."
                ),
            ),
            CPUOffloadingMetrics.CPU_CACHE_READ_USAGE_PERC: OffloadingGaugeMetadata(
                documentation=(
                    "Fraction of CPU KV-cache space currently pinned by "
                    "in-flight loads that have not yet "
                    "completed (0.0 = idle, 1.0 = saturated)."
                ),
            ),
            CPUOffloadingMetrics.CPU_ALLOCATION_SIZE: OffloadingHistogramMetadata(
                documentation=(
                    "Histogram of the number of CPU blocks requested by each "
                    "KV offload prepare_store call."
                ),
                buckets=(1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144),
            ),
        }
        store_threshold = int(extra_config.get("store_threshold", 0))
        if store_threshold >= 2:
            definitions[CPUOffloadingMetrics.STORES_SKIPPED] = (
                OffloadingCounterMetadata(
                    documentation=(
                        "Number of KV offload stores skipped because the reuse "
                        "threshold was not reached."
                    ),
                )
            )
        return definitions

    def __init__(self, config: OffloadingConfig):
        super().__init__(config)

        cpu_bytes_to_use = self.extra_config.get("cpu_bytes_to_use")
        if not cpu_bytes_to_use:
            raise Exception(
                "cpu_bytes_to_use must be specified in kv_connector_extra_config"
            )

        world_size = config.parallel.world_size
        self.num_blocks = 0
        self.kv_bytes_per_chunk = 0
        self.cpu_page_size_per_worker = 0
        self.replicated_layout = config.replicated_layout and self._uses_shared_region()
        # Chunk width of every worker's slot in the shared mmap region, in
        # rank order. Under pipeline parallelism each PP stage owns a
        # different layer set, so per-block KV footprints are stage-dependent
        # and the local view alone cannot size the shared row: every process
        # resolves the identical registered table instead, and the local view
        # is verified against it (fail closed) rather than trusted.
        self._slot_chunk_widths = self._resolve_slot_chunk_widths(world_size)
        if config.worker_kv_bytes_per_block > 0 and world_size > 0:
            if self.replicated_layout:
                num_copies = 1
                slot_widths = [config.worker_kv_bytes_per_block * self.blocks_per_chunk]
            else:
                num_copies = world_size
                slot_widths = self._slot_chunk_widths
            kv_bytes_per_chunk = sum(slot_widths)

            # calculate cpu_page_size_per_worker
            self.cpu_page_size_per_worker = slot_widths[
                min(config.parallel.rank, num_copies - 1)
            ]

            # calculate num_blocks
            aligned_kv_bytes_per_chunk = round_up(
                kv_bytes_per_chunk, self.BLOCK_SIZE_ALIGNMENT
            )
            self.num_blocks = int(cpu_bytes_to_use) // aligned_kv_bytes_per_chunk

            # Expose aligned_kv_bytes_per_chunk as
            # kv_bytes_per_chunk. Note that this might contain
            # some padding. i.e. each offloaded block is of the form,
            # |--- W0-B0---|---- W1-B0---| ... |---- Wn-B0---| *** maybe-pad *** |
            # or |--- B0 (single copy) ---| *** maybe-pad *** |
            self.kv_bytes_per_chunk = aligned_kv_bytes_per_chunk

        # scheduler-side
        self._manager: OffloadingManager | None = None

        # worker-side
        self._worker: CPUOffloadingWorker | None = None

        self.eviction_policy: str = self.extra_config.get("eviction_policy", "lru")
        self.cache_policy_module_path: str | None = self.extra_config.get(
            "cache_policy_module_path"
        )

    def _resolve_slot_chunk_widths(self, world_size: int) -> list[int]:
        """Return the per-rank chunk-width table of the shared offload region.

        Without a registered per-worker layout this is the local width
        replicated world-wide, which is only safe when every rank holds the
        same layers (TP-only). With pipeline parallelism a registered layout
        is mandatory: stage views differ, so the table sizes the shared row
        with per-stage accounting and each process verifies its local view
        against its own slot before attaching.
        """
        config = self.config
        by_rank = config.parallel.worker_kv_bytes_per_block_by_rank
        local_chunk_width = config.worker_kv_bytes_per_block * self.blocks_per_chunk
        if by_rank is None:
            if config.parallel.pp_size > 1:
                raise Exception(
                    "CPU offloading with pipeline_parallel_size > 1 requires "
                    "the registered per-worker KV layout "
                    "(KVCacheConfig.per_worker_kv_bytes_per_block); this "
                    "configuration path did not provide one. Refusing to size "
                    "one shared mmap from unequal stage views."
                )
            return [local_chunk_width] * world_size
        if len(by_rank) != world_size:
            raise Exception(
                f"registered per-worker KV layout has {len(by_rank)} entries "
                f"but world_size is {world_size}"
            )
        if by_rank[config.parallel.rank] * self.blocks_per_chunk != local_chunk_width:
            raise Exception(
                f"local KV view is {config.worker_kv_bytes_per_block} bytes per "
                f"block but the registered per-worker layout slot "
                f"{config.parallel.rank} is {by_rank[config.parallel.rank]}; "
                "refusing to attach to a mismatched offload layout"
            )
        return [width * self.blocks_per_chunk for width in by_rank]

    @override
    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            # store_threshold: how many times a block must appear in lookup()
            # before it is eligible for CPU offloading.  Values < 2 disable
            # filtering (a threshold of 1 equals no filter; 0 is the default).
            store_threshold = int(self.extra_config.get("store_threshold", 0))

            # Maximum entries in the internal tracker's LRU table.
            max_tracker_size = int(self.extra_config.get("max_tracker_size", 64_000))

            self._manager = CPUOffloadingManager(
                num_blocks=self.num_blocks,
                cache_policy=self.eviction_policy,
                cache_policy_module_path=self.cache_policy_module_path,
                enable_events=self.kv_events_config.enable_kv_cache_events,
                store_threshold=store_threshold,
                max_tracker_size=max_tracker_size,
            )
        return self._manager

    def _uses_shared_region(self) -> bool:
        """Whether the worker CPU buffer is the shared mmap region (vs a private
        per-rank tensor); replicated-layout dedup is gated on this being True."""
        return current_platform.is_cuda_alike()

    def _region_rank(self) -> int:
        """Slot this worker occupies in the shared region's row layout."""
        if self.replicated_layout:
            # Replicated layout puts all ranks on slot 0 (single MLA copy).
            return 0
        world_size = self.config.parallel.world_size
        return torch.accelerator.current_device_index() % world_size

    def create_worker(self, kv_caches: CanonicalKVCaches) -> CPUOffloadingWorker:
        mmap_region: SharedOffloadRegion | None = None
        # num_blocks == 0 would size the region to zero bytes, which cannot be
        # mmap'd; fall back to the tensor path (empty tensors) as before.
        if self._uses_shared_region() and self.num_blocks > 0:
            rank = self._region_rank()
            if not self.replicated_layout:
                # Fail closed: the slot this worker would occupy must match
                # the device-derived rank and its own local KV footprint.
                if rank >= len(self._slot_chunk_widths):
                    raise Exception(
                        f"worker slot {rank} is outside the registered offload "
                        f"layout ({len(self._slot_chunk_widths)} slots)"
                    )
                local_chunk_width = (
                    self.config.worker_kv_bytes_per_block * self.blocks_per_chunk
                )
                if self._slot_chunk_widths[rank] != local_chunk_width:
                    raise Exception(
                        f"worker slot {rank} expects "
                        f"{self._slot_chunk_widths[rank]} chunk bytes but the "
                        f"local KV view is {local_chunk_width}; refusing to "
                        "attach to a mismatched offload layout"
                    )
            # Uniform tables keep the legacy two-value region form, so the
            # TP-only layout stays byte-compatible; only genuinely unequal
            # stage views switch to the explicit per-rank slot table.
            uniform_slots = len(set(self._slot_chunk_widths)) <= 1
            mmap_region = SharedOffloadRegion(
                engine_id=self.config.engine_id,
                num_blocks=self.num_blocks,
                rank=rank,
                kv_bytes_per_block=self.kv_bytes_per_chunk,
                cpu_page_size=self.cpu_page_size_per_worker,
                slot_page_sizes=None if uniform_slots else self._slot_chunk_widths,
                barrier=_all_workers_barrier,
            )
        return CPUOffloadingWorker(
            kv_caches=kv_caches,
            blocks_per_chunk=self.blocks_per_chunk,
            num_cpu_blocks=self.num_blocks,
            mmap_region=mmap_region,
        )

    @override
    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        if not self._worker:
            if not (current_platform.is_cuda_alike() or current_platform.is_xpu()):
                raise Exception(
                    "CPU Offloading is currently only supported on CUDA-alike "
                    "and XPU GPUs"
                )
            self._worker = self.create_worker(kv_caches)

        assert self._worker is not None
        return self._worker
