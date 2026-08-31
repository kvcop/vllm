# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import mmap
import os
import time
from collections.abc import Callable, Sequence

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up

logger = init_logger(__name__)


def _wait_for_file_size(fd: int, expected_size: int, timeout: float = 30.0) -> None:
    """Spin-wait until the file reaches expected_size (creator truncated it)."""
    deadline = time.monotonic() + timeout
    while True:
        if os.fstat(fd).st_size >= expected_size:
            return
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Timed out waiting for mmap file to reach {expected_size} bytes"
            )
        time.sleep(0.005)


class SharedOffloadRegion:
    """
    Single mmap-backed memory region shared across all workers for a
    vLLM instance.  Workers coordinate via the filesystem: the first worker
    to open the file with O_EXCL becomes the creator and calls ftruncate;
    the rest open the existing file and wait until it reaches the expected
    size.  Each worker then mmap()s the full file.

    File path: /dev/shm/vllm_offload_{engine_id}.mmap. When a barrier is
    provided, the path is unlinked after every worker has mapped the file. The
    kernel can then reclaim the inode after the last mapping closes even when
    workers exit without running cleanup.

    Each block row holds one slot per worker:

        worker0_block0 | worker1_block0 | ... | worker{M-1}_block0
        worker0_block1 | worker1_block1 | ... | worker{M-1}_block1
        ...

    Under pipeline parallelism the per-worker slot width is stage-dependent:
    each PP stage owns a different set of layers, so its per-block KV
    footprint differs (e.g. the Qwen3.8 hybrid projection gives group widths
    (8,8,8,8) on stage 0 and (8,8,8,9) on stage 1). Workers with unequal
    views must therefore pass ``slot_page_sizes``: the per-rank chunk widths
    in rank order, registered identically on every worker. All workers then
    derive the same row stride and slot offsets from that table, so a block
    row never aliases storage across stages. The legacy two-value form
    (``kv_bytes_per_block`` + ``cpu_page_size``) keeps the uniform TP-only
    layout byte-compatible.
    """

    BLOCK_SIZE_ALIGNMENT: int = mmap.PAGESIZE

    def __init__(
        self,
        engine_id: str,
        num_blocks: int,
        rank: int | None,
        kv_bytes_per_block: int,
        cpu_page_size: int,
        slot_page_sizes: Sequence[int] | None = None,
        barrier: Callable[[], None] | None = None,
    ) -> None:
        self.page_size = mmap.PAGESIZE

        if slot_page_sizes is None:
            # Uniform layout: every worker's slot is cpu_page_size wide and
            # kv_bytes_per_block is the (already page-aligned) row stride.
            assert kv_bytes_per_block % self.page_size == 0
            assert kv_bytes_per_block % cpu_page_size == 0
            self._slot_page_sizes = [
                cpu_page_size for _ in range(kv_bytes_per_block // cpu_page_size)
            ]
            self._row_stride = kv_bytes_per_block
        else:
            # Registered per-rank layout: stage-heterogeneous slot widths.
            self._slot_page_sizes = list(slot_page_sizes)
            assert self._slot_page_sizes, "slot_page_sizes must not be empty"
            self._row_stride = round_up(sum(self._slot_page_sizes), self.page_size)

        self.num_blocks = num_blocks
        self.total_size_bytes = self.num_blocks * self._row_stride

        self.mmap_path = f"/dev/shm/vllm_offload_{engine_id}.mmap"
        self._creator = False  # set True only if this worker creates the file
        self.rank = rank
        if rank is not None:
            assert rank < len(self._slot_page_sizes), (
                f"rank {rank} has no slot in the registered offload layout "
                f"({len(self._slot_page_sizes)} slots)"
            )
            # byte offset to this worker's first slot within each block row
            self._worker_offset = sum(self._slot_page_sizes[:rank])
            # exclusive upper bound for this worker's area within each row
            self._worker_area_end = self._worker_offset + self._slot_page_sizes[rank]
        try:
            # Exclusive create — only one worker succeeds
            self.fd: int | None = os.open(
                self.mmap_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600
            )
            os.ftruncate(self.fd, self.total_size_bytes)
            self._creator = True
            logger.info(
                "Created mmap file %s (%.2f GB)",
                self.mmap_path,
                self.total_size_bytes / 1e9,
            )
        except FileExistsError:
            self.fd = os.open(self.mmap_path, os.O_RDWR)
            _wait_for_file_size(self.fd, self.total_size_bytes)
            observed_size = os.fstat(self.fd).st_size
            if observed_size != self.total_size_bytes:
                # A peer sized the file for a different layout, or a stale
                # file from a previous engine survived. Refuse to attach:
                # mapping a foreign stride would alias block storage across
                # stages and serve garbage.
                raise RuntimeError(
                    f"Offload mmap {self.mmap_path} is {observed_size} bytes but "
                    f"the registered per-worker layout requires exactly "
                    f"{self.total_size_bytes}; refusing to attach"
                ) from None
            logger.info("Opened existing mmap file %s", self.mmap_path)

        self.mmap_obj: mmap.mmap | None = mmap.mmap(
            self.fd,
            self.total_size_bytes,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )

        if barrier is not None:
            try:
                barrier()
            except Exception:
                self.mmap_obj.close()
                os.close(self.fd)
                if self._creator:
                    os.unlink(self.mmap_path)
                raise
            if self._creator:
                os.unlink(self.mmap_path)
                logger.info("Unlinked mmap file %s", self.mmap_path)

        # MADV_POPULATE_WRITE was added in Linux 5.14 (value 23).
        _MADV_POPULATE_WRITE = getattr(mmap, "MADV_POPULATE_WRITE", 23)
        if rank is not None:
            # Populate only this worker's pages (one slot per block row).
            worker_offset = self._worker_offset
            worker_page_size = self._slot_page_sizes[rank]
            _t0 = time.perf_counter()
            page_size = self.page_size
            for block in range(num_blocks):
                raw_offset = block * self._row_stride + worker_offset
                aligned_offset = (raw_offset // page_size) * page_size
                end = raw_offset + worker_page_size
                aligned_length = end - aligned_offset
                self.mmap_obj.madvise(
                    _MADV_POPULATE_WRITE, aligned_offset, aligned_length
                )
            logger.debug(
                "MADV_POPULATE_WRITE loop: %d blocks in %.3f s",
                num_blocks,
                time.perf_counter() - _t0,
            )
        else:
            # No rank — populate the entire shared region in one call.
            _t0 = time.perf_counter()
            self.mmap_obj.madvise(_MADV_POPULATE_WRITE, 0, self.total_size_bytes)
            logger.debug(
                "MADV_POPULATE_WRITE entire region: %.3f s", time.perf_counter() - _t0
            )

        self._base = torch.frombuffer(memoryview(self.mmap_obj), dtype=torch.int8)
        self._views: list[torch.Tensor] = []
        self.is_pinned: bool = False

    @property
    def row_stride(self) -> int:
        """Distance in bytes between consecutive block rows."""
        return self._row_stride

    def create_next_view(self, tensor_page_size: int) -> torch.Tensor:
        """Allocate a strided int8 view for this worker, one canonical tensor.

        Must be called once per canonical tensor. The full mmap layout is:

            worker0_block0 | worker1_block0 | ... | worker{M-1}_block0
            worker0_block1 | worker1_block1 | ... | worker{M-1}_block1
            ...

        Each worker_block cell is that worker's slot width (cpu_page_size in
        the uniform layout) and holds all canonical tensors for that worker
        and block concatenated:
            [ tensor0_data | tensor1_data | ... | tensor{L-1}_data ]

        Consecutive rows are separated by row_stride, the padded sum of all
        workers' slot widths.

        Returns an int8 tensor of shape (num_blocks, tensor_page_size) with stride
        (row_stride, 1).  Using int8 keeps stride == bytes, so swap_blocks
        address arithmetic works without any dtype conversion.

        Args:
            tensor_page_size: Bytes per block for this  tensor.
        """
        assert self.rank is not None
        new_offset = self._worker_offset + tensor_page_size
        assert new_offset <= self._worker_area_end, (
            f"Worker offset {new_offset} exceeds worker area end "
            f"{self._worker_area_end} (overflowed by "
            f"{new_offset - self._worker_area_end} bytes)"
        )
        worker_layer_view = torch.as_strided(
            self._base,
            size=(self.num_blocks, tensor_page_size),
            stride=(self._row_stride, 1),
            storage_offset=self._worker_offset,
        )
        self._worker_offset = new_offset
        self._views.append(worker_layer_view)
        return worker_layer_view

    def create_kv_memoryview(self) -> memoryview:
        """Return a zero-copy memoryview over the entire KV buffer.

        Shape: (num_blocks, row_stride_bytes). Secondary tiers address
        block *b* as ``view[b]``.
        """
        kv_tensor = self._base.view(self.num_blocks, self._row_stride)
        np_arr = kv_tensor.numpy()
        assert np_arr.ctypes.data == self._base.data_ptr(), (
            "view()/numpy() created a copy instead of sharing the mmap buffer; "
            "secondary tiers require zero-copy access to primary KV data"
        )
        return memoryview(np_arr)

    def cleanup(self) -> None:
        if self.is_pinned and self._base is not None:
            if current_platform.is_cuda_alike():
                base_ptr = self._base.data_ptr()
                result = torch.cuda.cudart().cudaHostUnregister(base_ptr)
                if result.value != 0:
                    logger.warning(
                        "cudaHostUnregister failed for rank=%d (code=%d)",
                        self.rank,
                        result,
                    )
            self.is_pinned = False
        # Release views before _base: each view holds a _base reference and a
        # direct StorageImpl reference.  Freeing views first lets both refcounts
        # drop so the storage (which holds the mmap_obj buffer export) is freed
        # before mmap_obj.close() is called below.
        if self._views is not None:
            self._views.clear()
        self._base = None
        if self.mmap_obj:
            try:
                self.mmap_obj.close()
            except Exception:
                logger.warning("Failed to close mmap_obj", exc_info=True)
            self.mmap_obj = None
        if self.fd is not None:
            try:
                os.close(self.fd)
            except Exception:
                logger.warning("Failed to close fd %s", self.fd, exc_info=True)
            self.fd = None
        if self._creator and getattr(self, "mmap_path", None):
            try:
                os.unlink(self.mmap_path)
                logger.info("Removed mmap file %s", self.mmap_path)
            except FileNotFoundError:
                pass
            except Exception:
                logger.warning(
                    "Failed to unlink path %s", self.mmap_path, exc_info=True
                )
            self._creator = False
