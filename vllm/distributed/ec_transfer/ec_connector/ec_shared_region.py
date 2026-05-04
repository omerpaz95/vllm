# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Lightweight mmap-backed shared memory region for encoder cache (EC) data.

Modeled after SharedOffloadRegion (vllm/v1/kv_offload/cpu/) but simplified
for EC: flat shared layout, no multi-tensor cursor, no block_size_factor.
"""

import mmap
import os
import time

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def _wait_for_file_size(fd: int, expected_size: int, timeout: float = 30.0):
    """Spin-wait until the file reaches expected_size (creator truncated it)."""
    deadline = time.monotonic() + timeout
    while True:
        if os.fstat(fd).st_size >= expected_size:
            return
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Timed out waiting for EC mmap file to reach {expected_size} bytes"
            )
        time.sleep(0.005)


class ECSharedRegion:
    """
    Flat mmap-backed memory region shared across TP workers for
    encoder cache blocks.

    Layout: (num_blocks, block_size_bytes) — contiguous, no per-worker
    interleaving.  All workers map the same file and see identical data.

    File path: /dev/shm/vllm_ec_{instance_id}.mmap
    """

    def __init__(
        self,
        instance_id: str,
        num_blocks: int,
        block_size_bytes: int,
    ) -> None:
        self.num_blocks = num_blocks
        self.block_size_bytes = block_size_bytes

        self.total_size_bytes = num_blocks * block_size_bytes
        self.mmap_path = f"/dev/shm/vllm_ec_{instance_id}.mmap"
        self._creator = False

        # Creator/joiner race: first worker creates, others wait
        try:
            self.fd: int | None = os.open(
                self.mmap_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600
            )
            os.ftruncate(self.fd, self.total_size_bytes)
            self._creator = True
            logger.info(
                "Created EC mmap file %s (%.2f MB)",
                self.mmap_path,
                self.total_size_bytes / 1e6,
            )
        except FileExistsError:
            self.fd = os.open(self.mmap_path, os.O_RDWR)
            _wait_for_file_size(self.fd, self.total_size_bytes)
            logger.info("Opened existing EC mmap file %s", self.mmap_path)

        self.mmap_obj: mmap.mmap | None = mmap.mmap(
            self.fd,
            self.total_size_bytes,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )

        if self._creator:
            _MADV_POPULATE_WRITE = getattr(mmap, "MADV_POPULATE_WRITE", 23)
            self.mmap_obj.madvise(_MADV_POPULATE_WRITE, 0, self.total_size_bytes)

        self._base: torch.Tensor | None = torch.frombuffer(
            memoryview(self.mmap_obj), dtype=torch.int8
        )
        self.is_pinned: bool = False

        self.blocks: torch.Tensor = self._base.view(num_blocks, block_size_bytes)

    def pin_memory(self) -> None:
        """Register the entire mmap as CUDA pinned memory for fast DMA."""
        if self._base is None:
            return
        base_ptr = self._base.data_ptr()
        result = torch.cuda.cudart().cudaHostRegister(
            base_ptr, self.total_size_bytes, 0
        )
        if result.value != 0:
            logger.warning(
                "cudaHostRegister failed (code=%d) — "
                "transfers will still work but may be slower (unpinned DMA)",
                result,
            )
        else:
            logger.debug(
                "cudaHostRegister %.2f MB",
                self.total_size_bytes / 1e6,
            )
            self.is_pinned = True

    def cleanup(self) -> None:
        if self.is_pinned and self._base is not None:
            base_ptr = self._base.data_ptr()
            result = torch.cuda.cudart().cudaHostUnregister(base_ptr)
            if result.value != 0:
                logger.warning(
                    "cudaHostUnregister failed (code=%d)",
                    result,
                )
            self.is_pinned = False
        self.blocks = None  # type: ignore[assignment]
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
                logger.info("Removed EC mmap file %s", self.mmap_path)
            except Exception:
                logger.warning(
                    "Failed to unlink path %s", self.mmap_path, exc_info=True
                )
            self._creator = False
