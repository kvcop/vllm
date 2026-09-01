# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
FileSystemTierManager: Pure-Python file system secondary tier for KV cache offloading.

Store path:
    Data is written to a temp file (<dest_path.tmp>) via os.write,
    then linked to the final path without replacing an existing block.

Load path:
    Data is read from the block file directly via os.readv into the
    provided memoryview slice.

File naming:  <base_path>_r<rank>/<hhh>/<hh>_g<group_idx>/<hash_hex>.bin
              (hash-based subdirectories to limit directory fan-out)
"""

import functools
import json
import os
import stat
import threading
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, ClassVar

try:
    from vllm.fs_io_C import batch_lookup as batch_lookup_C

    _HAS_BATCH_LOOKUP_C = True
except ImportError:
    _HAS_BATCH_LOOKUP_C = False

from typing_extensions import override

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    Locality,
    LookupResult,
    Medium,
    OffloadingCounterMetadata,
    OffloadingEvent,
    OffloadingMetricMetadata,
    OffloadKey,
    ReqContext,
)
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.async_lookup import AsyncLookupManager
from vllm.v1.kv_offload.tiering.base import (
    JobId,
    JobMetadata,
    JobResult,
    RequestOffloadingContext,
    ScheduleEndContext,
    SecondaryTierManager,
)
from vllm.v1.kv_offload.tiering.fs.io import (
    batch_load_block,
    batch_store_block_results,
    probe_o_direct,
)
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec

logger = init_logger(__name__)


class FileSystemTierMetrics:
    """Metric names for filesystem-tier block I/O."""

    LOAD_BYTES = "vllm:kv_offload_fs_load_bytes"
    LOAD_OPS = "vllm:kv_offload_fs_load_ops"
    STORE_BYTES = "vllm:kv_offload_fs_store_bytes"
    STORE_OPS = "vllm:kv_offload_fs_store_ops"


class FsAsyncLookupManager(AsyncLookupManager):
    """Async lookup manager for FileSystemTierManager."""

    def __init__(
        self,
        tier: "FileSystemTierManager",
        tier_type: str,
    ) -> None:
        super().__init__(tier_type=tier_type)
        self._tier = tier

    def batch_lookup(
        self, keys: list[OffloadKey], req_context: ReqContext
    ) -> Iterable[bool]:
        paths = [self._tier.file_mapper.get_file_name(k) for k in keys]
        if _HAS_BATCH_LOOKUP_C:
            # C extension: GIL released for the entire faccessat() batch.
            return batch_lookup_C(paths)
        return (os.path.exists(p) for p in paths)


class FileSystemTierManager(SecondaryTierManager):
    """
    Pure-Python disk-backed secondary tier.

    Read-priority threads service load jobs preferentially; write-priority
    threads service store jobs preferentially.  Both groups can drain either
    queue, so neither starves.

    submit_store / submit_load are non-blocking: they enqueue tasks and return.
    get_finished_jobs() polls job completion and returns completed JobResults.

    Cross-process sharing:
        In order to enable KV cache sharing between multiple vLLM instances
        using the same ``root_dir`` (e.g., via a shared PVC) the environment
        variable ``PYTHONHASHSEED`` must be set to the same fixed value
        (e.g., "0") on all instances. Without this, each process initializes
        ``NONE_HASH`` (the chain-hash seed for block content hashes) with
        random bytes, producing different block filenames for identical token
        content.
    """

    medium: ClassVar[Medium] = Medium.STORAGE

    @classmethod
    def build_metric_definitions(
        cls, extra_config: dict[str, Any]
    ) -> dict[str, OffloadingMetricMetadata]:
        return {
            FileSystemTierMetrics.LOAD_BYTES: OffloadingCounterMetadata(
                documentation=(
                    "Total bytes successfully read from the filesystem KV tier."
                ),
            ),
            FileSystemTierMetrics.LOAD_OPS: OffloadingCounterMetadata(
                documentation=(
                    "Number of KV blocks successfully read from the filesystem tier."
                ),
            ),
            FileSystemTierMetrics.STORE_BYTES: OffloadingCounterMetadata(
                documentation=(
                    "Total bytes successfully written as new files in the "
                    "filesystem KV tier."
                ),
            ),
            FileSystemTierMetrics.STORE_OPS: OffloadingCounterMetadata(
                documentation=(
                    "Number of KV blocks successfully written as new files in "
                    "the filesystem tier."
                ),
            ),
        }

    def __init__(
        self,
        offloading_spec: "OffloadingSpec",
        primary_kv_view: memoryview,
        tier_type: str,
        root_dir: str,
        n_read_threads: int = 16,
        n_write_threads: int = 16,
        enable_kv_events: bool = False,
        locality: str | None = None,
        require_o_direct: bool = False,
        expected_python_hash_seed: str | None = None,
        required_root_mode: int | None = None,
    ):
        """
        Args:
            offloading_spec: Contains normalized offloading configuration and
                blocks_per_chunk.
            primary_kv_view: Memoryview of the primary tier's CPU KV cache.
            tier_type: Tier type identifier, set by SecondaryTierFactory.
            root_dir: Root directory for block files.
            n_read_threads: Number of read-priority I/O threads.
            n_write_threads: Number of write-priority I/O threads.
            enable_kv_events: Emit BlockStored KV events for blocks
                successfully stored to this tier. Effective only when KV
                cache events are enabled globally (kv_events_config).
            locality: Whether this tier's storage is LOCAL or REMOTE relative
                to the publishing vLLM instance.
            require_o_direct: Reject construction instead of using buffered I/O
                when the filesystem does not support ``O_DIRECT``.
            expected_python_hash_seed: Require ``PYTHONHASHSEED`` to equal this
                value and bind it into the configuration directory identity.
            required_root_mode: Require the existing root directory to have
                this exact permission mode, for example ``0o700``.
        """
        super().__init__(offloading_spec, primary_kv_view, tier_type)
        self.locality = Locality(locality) if locality is not None else None

        if offloading_spec.config.parallel.pp_size != 1:
            raise ValueError(
                "filesystem KV tier does not support pipeline parallelism; "
                "pipeline_parallel_size must be 1"
            )

        python_hash_seed = None
        if expected_python_hash_seed is not None:
            python_hash_seed = os.getenv("PYTHONHASHSEED")
            if python_hash_seed != expected_python_hash_seed:
                raise ValueError(
                    "filesystem KV tier requires PYTHONHASHSEED="
                    f"{expected_python_hash_seed}, got {python_hash_seed!r}"
                )

        if required_root_mode is not None:
            if not os.path.isabs(root_dir):
                raise ValueError("filesystem KV tier root_dir must be absolute")
            root_stat = os.stat(root_dir, follow_symlinks=False)
            if not stat.S_ISDIR(root_stat.st_mode):
                raise ValueError(
                    "filesystem KV tier root_dir must be an existing directory"
                )
            actual_mode = stat.S_IMODE(root_stat.st_mode)
            if actual_mode != required_root_mode:
                raise PermissionError(
                    f"filesystem KV tier root_dir mode must be "
                    f"{required_root_mode:#o}, got {actual_mode:#o}"
                )

        self.events: list[OffloadingEvent] | None = None
        if enable_kv_events:
            if offloading_spec.kv_events_config.enable_kv_cache_events:
                self.events = []
            else:
                logger.warning(
                    "enable_kv_events is set on secondary tier '%s' but KV "
                    "cache events are disabled globally; the tier will not "
                    "emit events.",
                    tier_type,
                )
        # Keys and exact per-block publications for in-flight store jobs.
        self._store_job_keys: dict[JobId, list[OffloadKey]] = {}
        self._store_job_published_keys: dict[JobId, list[OffloadKey]] = {}
        self._store_job_lock = threading.Lock()

        # Extract block size from primary view
        assert primary_kv_view.strides is not None, (
            "primary_kv_view.strides cannot be None"
        )
        self._block_size: int = primary_kv_view.strides[0]

        # Opt in; FileMapper enables it only for a parallelism-invariant block.
        self.file_mapper = FileMapper.from_offloading_spec(
            root_dir=root_dir,
            offloading_spec=offloading_spec,
            blocks_per_file=offloading_spec.blocks_per_chunk,
            parallel_agnostic=True,
            python_hash_seed=python_hash_seed,
        )

        # Write or validate the deterministic restart identity.
        config_path = self.file_mapper.get_config_file_path()
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        run_config = self.file_mapper.get_run_config()
        if os.path.exists(config_path):
            with open(config_path, encoding="utf-8") as config_file:
                existing_config = json.load(config_file)
            if existing_config != run_config:
                raise ValueError(
                    "filesystem KV tier config.json does not match the current "
                    "runtime configuration"
                )
        else:
            with open(config_path, "w", encoding="utf-8") as config_file:
                json.dump(run_config, config_file, indent=2, sort_keys=True)

        # Prefer O_DIRECT to bypass the page cache, but fall back to buffered
        # I/O on filesystems that reject it (e.g. overlayfs, some NFS mounts)
        # rather than failing every block.
        self._use_o_direct = probe_o_direct(os.path.dirname(config_path))
        if not self._use_o_direct:
            if require_o_direct:
                raise RuntimeError(
                    f"O_DIRECT is required for filesystem KV tier at {root_dir!r}"
                )
            logger.warning(
                "O_DIRECT is not supported at '%s'; falling back to buffered "
                "I/O for the '%s' KV offload tier.",
                root_dir,
                tier_type,
            )

        logger.info(
            "FS_TIER_EVIDENCE tier=%s root_dir=%s base_path=%s config_path=%s "
            "o_direct=%s python_hash_seed=%s",
            tier_type,
            root_dir,
            self.file_mapper.base_path,
            config_path,
            self._use_o_direct,
            python_hash_seed,
        )

        self._stats = OffloadingConnectorStats()
        self._stats_lock = threading.Lock()

        self._pool = DualQueueThreadPool(
            n_read_threads,
            n_write_threads,
            thread_name_prefix="vllm_kv_py_fs",
        )

        self._lookup_manager = FsAsyncLookupManager(tier=self, tier_type=self.tier_type)

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        result = self._lookup_manager.lookup(key, req_context)
        if result is None:
            return LookupResult.RETRY
        return LookupResult.HIT if result else LookupResult.MISS

    @override
    def submit_store(self, job_metadata: JobMetadata) -> None:
        with self._store_job_lock:
            self._store_job_keys[job_metadata.job_id] = list(job_metadata.keys)
            self._store_job_published_keys[job_metadata.job_id] = []
        task = functools.partial(
            self._store_blocks,
            job_metadata.job_id,
            list(job_metadata.keys),
            [self.file_mapper.get_file_name(key) for key in job_metadata.keys],
            self._primary_kv_view,
            [int(bid) * self._block_size for bid in job_metadata.block_ids],
            self._block_size,
            self._use_o_direct,
        )
        self._pool.enqueue_store(job_metadata.job_id, 1, [task])

    @override
    def submit_load(self, job_metadata: JobMetadata) -> None:
        task = functools.partial(
            self._load_blocks,
            [self.file_mapper.get_file_name(key) for key in job_metadata.keys],
            self._primary_kv_view,
            [int(bid) * self._block_size for bid in job_metadata.block_ids],
            self._block_size,
            self._use_o_direct,
        )

        self._pool.enqueue_load(job_metadata.job_id, 1, [task])

    def _store_blocks(
        self,
        job_id: JobId,
        keys: list[OffloadKey],
        paths: list[str],
        view: memoryview,
        offsets: list[int],
        block_size: int,
        use_o_direct: bool,
    ) -> None:
        stored = batch_store_block_results(
            paths, view, offsets, block_size, use_o_direct
        )
        published_keys = [key for key, published in zip(keys, stored) if published]
        with self._store_job_lock:
            self._store_job_published_keys[job_id] = published_keys
        self._record_io(
            FileSystemTierMetrics.STORE_BYTES,
            FileSystemTierMetrics.STORE_OPS,
            len(published_keys),
            block_size,
        )

    def _load_blocks(
        self,
        paths: list[str],
        view: memoryview,
        offsets: list[int],
        block_size: int,
        use_o_direct: bool,
    ) -> None:
        blocks_loaded = batch_load_block(paths, view, offsets, block_size, use_o_direct)
        self._record_io(
            FileSystemTierMetrics.LOAD_BYTES,
            FileSystemTierMetrics.LOAD_OPS,
            blocks_loaded,
            block_size,
        )

    def _record_io(
        self,
        bytes_metric: str,
        ops_metric: str,
        block_count: int,
        block_size: int,
    ) -> None:
        if block_count == 0:
            return
        with self._stats_lock:
            self._stats.increase_counter(bytes_metric, block_count * block_size)
            self._stats.increase_counter(ops_metric, block_count)

    @override
    def get_finished_jobs(self) -> Iterable[JobResult]:
        """
        Collect completed jobs from the finished-jobs queue.
        """
        results = []
        for job_id, success in self._pool.get_finished():
            with self._store_job_lock:
                keys = self._store_job_keys.pop(job_id, None)
                published_keys = self._store_job_published_keys.pop(job_id, [])
            stored = None if keys is None else success and bool(published_keys)
            if self.events is not None and success and published_keys:
                self.events.append(
                    OffloadingEvent(
                        keys=published_keys,
                        medium=self.medium,
                        removed=False,
                        locality=self.locality,
                    )
                )
            results.append(JobResult(job_id=job_id, success=success, stored=stored))
        return results

    @override
    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()

    @override
    def get_stats(self) -> OffloadingConnectorStats | None:
        with self._stats_lock:
            if self._stats.is_empty():
                return None
            stats = self._stats
            self._stats = OffloadingConnectorStats()
            return stats

    @override
    def drain_jobs(self) -> None:
        """Block until all in-flight transfers in the threadpool finish."""
        self._pool.wait_idle()

    def on_request_finished(self, req_context: ReqContext) -> None:
        self._lookup_manager.cleanup(req_context.req_id)

    @override
    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        self._lookup_manager.flush()

    @override
    def shutdown(self) -> None:
        """
        Release resources held by this tier.

        Shuts down the lookup manager and the thread pool,
        clearing pending tasks and waiting for active threads to complete.
        """
        self._lookup_manager.shutdown()
        self._pool.shutdown(wait=True)
        with self._store_job_lock:
            self._store_job_keys.clear()
            self._store_job_published_keys.clear()
