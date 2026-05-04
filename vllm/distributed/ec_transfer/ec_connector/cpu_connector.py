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
from vllm.distributed.ec_transfer.ec_connector.ec_shared_region import (
    ECSharedRegion,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class ECCPUConnectorMetadata(ECConnectorMetadata):
    # mm_hash -> list of block indices in the shared CPU region
    mm_hash_to_cpu_blocks: dict[str, list[int]] = field(default_factory=dict)


class ECCPUConnector(ECConnectorBase):
    """EC connector for E-PD disaggregation.

    Loads encodings from a remote Encoder node into a dedicated CPU cache,
    and loads them to the GPU on demand."""

    def __init__(self, vllm_config: "VllmConfig", role: ECConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)

        if self.role == ECConnectorRole.SCHEDULER:
            # mm_hash -> list of block indices in the shared CPU region
            self._encoding_map: dict[str, list[int]] = {}

        if self.role == ECConnectorRole.WORKER:
            ec_config = vllm_config.ec_transfer_config
            assert ec_config is not None

            hidden_dim = vllm_config.model_config.get_inputs_embeds_size()
            element_size = torch.tensor(
                [], dtype=vllm_config.model_config.dtype
            ).element_size()
            block_size_bytes = hidden_dim * element_size

            num_ec_blocks = int(ec_config.get_from_extra_config("num_ec_blocks", 256))

            self._region = ECSharedRegion(
                instance_id=ec_config.engine_id,
                num_blocks=num_ec_blocks,
                block_size_bytes=block_size_bytes,
            )
            if is_pin_memory_available() and vllm_config.parallel_config.rank == 0:
                self._region.pin_memory()
            self._cpu_blocks = self._region.blocks

            self._copy_stream: torch.cuda.Stream | None = None
            self._copy_event: torch.Event | None = None

    def register_caches(
        self,
        ec_caches: dict[str, torch.Tensor],
    ):
        raise NotImplementedError

    # ==============================
    # Worker-side methods
    # ==============================

    def start_load_caches(
        self, encoder_cache: dict[str, torch.Tensor], **kwargs
    ) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, ECCPUConnectorMetadata)

        assert self._cpu_blocks is not None

        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream()
            self._copy_event = torch.Event()
        assert self._copy_event is not None

        with torch.cuda.stream(self._copy_stream):
            for mm_hash, block_indices in metadata.mm_hash_to_cpu_blocks.items():
                if mm_hash in encoder_cache:
                    continue
                encoder_cache[mm_hash] = self._cpu_blocks[block_indices].to(
                    device=current_platform.device_type, non_blocking=True
                )
            self._copy_event.record(self._copy_stream)

        torch.cuda.current_stream().wait_event(self._copy_event)

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
            mm_hash_to_cpu_blocks=self._encoding_map,
        )
        self._encoding_map = {}
        return meta
