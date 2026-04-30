# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class ECCPUConnectorMetadata(ECConnectorMetadata):
    # mm_hash -> (offset, size) in the flat CPU buffer
    mm_hash_to_cpu_loc: dict[str, tuple[int, int]] = field(default_factory=dict)


class ECCPUConnector(ECConnectorBase):
    """EC connector for E-PD disaggregation.

    Loads encodings from a remote Encoder node into a dedicated CPU cache,
    and loads them to the GPU on demand."""

    def __init__(self, vllm_config: "VllmConfig", role: ECConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        # mm_hash -> (offset, size) in the flat CPU buffer
        self._encoding_map: dict[str, tuple[int, int]] = {}

        self._cpu_buffer: torch.Tensor | None = None

    # ==============================
    # Worker-side methods
    # ==============================

    def register_caches(self, ec_caches: dict[str, torch.Tensor]):
        """
        This needs to be called by someone
        Or, alternatively, create the cpu buffer in init().
        """

        if "cpu_buffer" not in ec_caches:
            raise ValueError("ECCPUConnector requires 'cpu_buffer' in ec_caches")
        self._cpu_buffer = ec_caches["cpu_buffer"]

    def start_load_caches(
        self, encoder_cache: dict[str, torch.Tensor], **kwargs
    ) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, ECCPUConnectorMetadata)

        for mm_hash, (offset, size) in metadata.mm_hash_to_cpu_loc.items():
            if mm_hash in encoder_cache or not self._cpu_buffer:
                continue
            encoder_cache[mm_hash] = self._cpu_buffer[offset : offset + size].to(
                device=current_platform.device_type, non_blocking=True
            )

    def save_caches(
        self, encoder_cache: dict[str, torch.Tensor], mm_hash: str, **kwargs
    ) -> None:
        raise NotImplementedError

    # ==============================
    # Scheduler-side methods
    # ==============================

    def has_cache_item(
        self,
        identifier: str,
    ) -> bool | None:
        raise NotImplementedError

    def update_state_after_alloc(self, request: "Request", index: int):
        raise NotImplementedError

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> ECCPUConnectorMetadata:
        meta = ECCPUConnectorMetadata(
            mm_hash_to_cpu_loc=self._encoding_map,
        )
        self._encoding_map = {}
        return meta
