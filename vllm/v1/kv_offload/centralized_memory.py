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
            mmap_path = f"/tmp/vllm_offload_{os.getpid()}.mmap"
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

        # Register cleanup on exit
        atexit.register(self.cleanup)

        logger.info(
            "Initialized CentralizedOffloadMemoryManager: size=%.2f GB, path=%s",
            self.total_size_bytes / (1e9),
            self.mmap_path,
        )

    def get_worker_memory_view(
        self,
        tp_rank: int,
        tp_world_size: int,
        shape: tuple,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Get a tensor view into the mmap region for a specific TP worker.

        Args:
            tp_rank: Tensor-parallel rank of the worker (0-indexed)
            tp_world_size: Total number of tensor-parallel workers
            shape: Desired tensor shape
            dtype: PyTorch data type for the tensor

        Returns:
            A torch.Tensor backed by the mmap file at the calculated offset
        """
        # Calculate worker's region size (page-aligned)
        worker_size_bytes = self.total_size_bytes // tp_world_size
        worker_size_bytes = (worker_size_bytes // self.page_size) * self.page_size

        # Calculate offset for this worker
        offset_bytes = tp_rank * worker_size_bytes

        # Convert dtype and calculate tensor size
        element_size = torch.tensor([], dtype=dtype).element_size()
        num_elements = math.prod(shape)
        tensor_size_bytes = num_elements * element_size

        logger.debug(
            "Creating tensor view for TP rank %d/%d:",
            tp_rank,
            tp_world_size,
        )

        logger.debug(
            "shape=%s, dtype=%s, offset=%d bytes",
            shape,
            dtype,
            offset_bytes,
        )

        # Create a storage view into the mmap at the correct offset
        memory_view = memoryview(self.mmap_obj)[
            offset_bytes : offset_bytes + tensor_size_bytes
        ]

        # Create tensor from memory view
        tensor = torch.frombuffer(memory_view, dtype=dtype, count=num_elements).reshape(
            shape
        )

        return tensor

    def cleanup(self) -> None:
        """Cleanup mmap resources and delete the backing file."""
        if hasattr(self, "mmap_obj") and self.mmap_obj is not None:
            self.mmap_obj.close()

        if hasattr(self, "file_handle") and self.file_handle is not None:
            os.close(self.file_handle)

        if hasattr(self, "mmap_path") and os.path.exists(self.mmap_path):
            os.unlink(self.mmap_path)
            logger.info("Cleaned up mmap file: %s", self.mmap_path)

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
