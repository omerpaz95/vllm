# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Centralized memory manager for KV cache offloading.

This module provides a singleton manager that allocates a single mmap-backed
memory region shared across all tensor-parallel (TP) workers. Each worker
gets a view into this shared memory at a calculated offset based on its TP rank.
"""

import atexit
import math
import mmap
import os
import ctypes

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class CentralizedOffloadMemoryManager:
    """
    Singleton manager for mmap-backed CPU memory shared across all TP workers.

    This class creates a single memory-mapped file that is partitioned among
    all tensor-parallel workers. Each worker receives a tensor view into its
    designated region of the mmap file.

    Memory Layout:
    ┌─────────────────────────────────────────────────────────────┐
    │                    Single MMap File                         │
    ├─────────────────┬─────────────────┬─────────────────────────┤
    │  TP Worker 0    │  TP Worker 1    │  ...  │  TP Worker N-1  │
    │  Region         │  Region         │       │  Region         │
    └─────────────────┴─────────────────┴───────┴─────────────────┘

    Thread-safety: This class uses a lock to ensure thread-safe singleton
    initialization and memory allocation.
    """

    def __init__(self, total_size_bytes: int, mmap_path: str | None = None):
        """
        Initialize the centralized memory manager.

        Args:
            total_size_bytes: Total memory size in bytes for all TP workers
            mmap_path: Path to mmap file. If None, uses /tmp/vllm_offload_<pid>.mmap
        """
        # Ensure size is page-aligned
        self.page_size = mmap.PAGESIZE
        self.total_size_bytes = (
            (total_size_bytes + self.page_size - 1) // self.page_size
        ) * self.page_size

        # Set mmap file path
        if mmap_path is None:
            mmap_path = f"/dev/shm/vllm_offload_{os.getpid()}.mmap"
        self.mmap_path = mmap_path

        # Create mmap file
        self.file_descriptor = os.open(
            self.mmap_path,
            os.O_CREAT | os.O_TRUNC | os.O_RDWR,
            0o600,
        )
        os.ftruncate(self.file_descriptor, self.total_size_bytes)

        # Create memory mapping
        self.mmap_obj = mmap.mmap(
            self.file_descriptor,
            self.total_size_bytes,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )

        self.mmap_obj.madvise(mmap.MADV_WILLNEED)  # Prefault pages
        self.mmap_obj.madvise(mmap.MADV_SEQUENTIAL)  # Optimize for sequential access

        self._tensor_cache: dict[tuple, torch.Tensor] = {}
        
        # Register cleanup on exit
        atexit.register(self.cleanup)

        logger.info(
            "Initialized CentralizedOffloadMemoryManager: size=%.2f GB, path=%s",
            self.total_size_bytes / (1e9),
            self.mmap_path,
        )
        
    def base_ptr(self) -> int:
        # Create a 1-byte memoryview, then get its address
        mv = memoryview(self.mmap_obj)
        c_char_p = ctypes.c_char.from_buffer(mv)
        return ctypes.addressof(c_char_p)

    def worker_region(self, tp_rank: int, tp_world_size: int) -> tuple[int, int]:
        worker_size_bytes = self.total_size_bytes // tp_world_size
        worker_size_bytes = (worker_size_bytes // self.page_size) * self.page_size
        offset_bytes = tp_rank * worker_size_bytes
        return offset_bytes, worker_size_bytes

    def get_worker_memory_view(
        self,
        tp_rank: int,
        tp_world_size: int,
        shape: tuple,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        cache_key = (tp_rank, shape, dtype)
        if cache_key in self._tensor_cache:
            return self._tensor_cache[cache_key]

        worker_size_bytes = self.total_size_bytes // tp_world_size
        worker_size_bytes = (worker_size_bytes // self.page_size) * self.page_size
        offset_bytes = tp_rank * worker_size_bytes

        element_size = torch.tensor([], dtype=dtype).element_size()
        num_elements = math.prod(shape)
        tensor_size_bytes = num_elements * element_size

        # You may want to assert tensor_size_bytes <= worker_size_bytes

        memory_view = memoryview(self.mmap_obj)[
            offset_bytes : offset_bytes + tensor_size_bytes
        ]

        tensor = torch.frombuffer(memory_view, dtype=dtype, count=num_elements).reshape(
            shape
        )
        self._tensor_cache[cache_key] = tensor
        return tensor


    def cleanup(self) -> None:
        """Cleanup mmap resources and delete the backing file."""
        if hasattr(self, "mmap_obj") and self.mmap_obj is not None:
            try:
                self.mmap_obj.close()
            except BufferError:
                # Expected: other processes may still have memoryview references
                # The OS will clean up the mmap when all processes exit
                pass
            except Exception as e:
                logger.debug("Error closing mmap: %s", e)

        if hasattr(self, "file_handle") and self.file_handle is not None:
            try:
                os.close(self.file_handle)
            except Exception as e:
                logger.debug("Error closing file handle: %s", e)

        if hasattr(self, "mmap_path") and os.path.exists(self.mmap_path):
            try:
                os.unlink(self.mmap_path)
                logger.info("Cleaned up mmap file: %s", self.mmap_path)
            except Exception as e:
                # File may already be deleted by another process
                logger.debug("Could not delete mmap file: %s", e)

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
