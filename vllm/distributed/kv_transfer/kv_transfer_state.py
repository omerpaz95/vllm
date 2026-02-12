# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBaseType
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.kv_offload.centralized_memory import (
        CentralizedOffloadMemoryManager,
    )

logger = init_logger(__name__)

_KV_CONNECTOR_AGENT: KVConnectorBaseType | None = None
_OFFLOAD_MEMORY_MANAGER: "CentralizedOffloadMemoryManager | None" = None


def get_kv_transfer_group() -> KVConnectorBaseType:
    assert _KV_CONNECTOR_AGENT is not None, (
        "disaggregated KV cache transfer parallel group is not initialized"
    )
    return _KV_CONNECTOR_AGENT


def has_kv_transfer_group() -> bool:
    return _KV_CONNECTOR_AGENT is not None


def is_v1_kv_transfer_group(connector: KVConnectorBaseType | None = None) -> bool:
    """Check if the KV connector is the v1 connector.
    If the argument is None, it will check the global KV connector

    Args:
        connector: The KV connector to check. If None, it will check the
            global KV connector.

    Note:
        This function will no-longer be needed after the v1 KV connector
        becomes the default.
    """
    if connector is None:
        connector = _KV_CONNECTOR_AGENT

    if connector is None:
        return False

    return isinstance(connector, KVConnectorBase_V1)


def get_offload_memory_manager() -> "CentralizedOffloadMemoryManager | None":
    """Get the global offload memory manager instance."""
    return _OFFLOAD_MEMORY_MANAGER


def ensure_offload_memory_manager_initialized(
    vllm_config: "VllmConfig",
) -> "CentralizedOffloadMemoryManager | None":
    """
    Initialize the global CentralizedOffloadMemoryManager if needed.
    This must be called before any connectors are created.
    
    Returns the memory manager instance or None if not using OffloadingConnector.
    """
    global _OFFLOAD_MEMORY_MANAGER
    
    if vllm_config.kv_transfer_config is None:
        return None
    
    kv_transfer_config = vllm_config.kv_transfer_config
    connector_name = kv_transfer_config.kv_connector
    
    # Only create for OffloadingConnector
    if connector_name != "OffloadingConnector":
        return None
    
    # Create singleton if not already created
    if _OFFLOAD_MEMORY_MANAGER is None:
        from vllm.v1.kv_offload.centralized_memory import (
            CentralizedOffloadMemoryManager,
        )
        
        extra_config = kv_transfer_config.kv_connector_extra_config
        cpu_bytes_to_use = extra_config.get("cpu_bytes_to_use")
        
        if cpu_bytes_to_use:
            import os
            total_cpu_bytes = int(cpu_bytes_to_use)
            mmap_path = extra_config.get("mmap_path")
            _OFFLOAD_MEMORY_MANAGER = CentralizedOffloadMemoryManager(
                total_size_bytes=total_cpu_bytes,
                mmap_path=mmap_path,
            )
            logger.info(
                "Created global CentralizedOffloadMemoryManager: "
                "size=%.2f GB, path=%s",
                total_cpu_bytes / (1e9),
                mmap_path or f"/tmp/vllm_offload_{os.getpid()}.mmap",
            )
    
    return _OFFLOAD_MEMORY_MANAGER


def ensure_kv_transfer_initialized(
    vllm_config: "VllmConfig", kv_cache_config: "KVCacheConfig | None" = None
) -> None:
    """
    Initialize KV cache transfer parallel group.
    """

    global _KV_CONNECTOR_AGENT

    if vllm_config.kv_transfer_config is None:
        return

    if (
        vllm_config.kv_transfer_config.is_kv_transfer_instance
        and _KV_CONNECTOR_AGENT is None
    ):
        # Ensure memory manager is initialized first for OffloadingConnector
        memory_manager = ensure_offload_memory_manager_initialized(vllm_config)
        
        _KV_CONNECTOR_AGENT = KVConnectorFactory.create_connector(
            config=vllm_config,
            role=KVConnectorRole.WORKER,
            kv_cache_config=kv_cache_config,
            memory_manager=memory_manager,
        )


def ensure_kv_transfer_shutdown() -> None:
    global _KV_CONNECTOR_AGENT, _OFFLOAD_MEMORY_MANAGER
    
    if _KV_CONNECTOR_AGENT is not None:
        _KV_CONNECTOR_AGENT.shutdown()
        _KV_CONNECTOR_AGENT = None
    
    # Cleanup memory manager
    if _OFFLOAD_MEMORY_MANAGER is not None:
        logger.info("Cleaning up global CentralizedOffloadMemoryManager")
        _OFFLOAD_MEMORY_MANAGER.cleanup()
        _OFFLOAD_MEMORY_MANAGER = None
