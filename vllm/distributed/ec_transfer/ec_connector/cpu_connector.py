# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import msgspec.msgpack
import torch
import zmq

from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.distributed.ec_transfer.ec_connector.ec_shared_region import (
    ECSharedRegion,
)
from vllm.distributed.ec_transfer.ec_connector.messages import (
    XferReq,
    compute_ec_compatibility_hash,
)
from vllm.distributed.nixl_utils import NixlWrapper, nixl_agent_config
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.v1.request import Request


logger = init_logger(__name__)


def _vllm_version() -> str:
    from vllm import __version__

    return __version__


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

        ec_config = vllm_config.ec_transfer_config
        assert ec_config is not None

        hidden_dim = vllm_config.model_config.get_inputs_embeds_size()
        element_size = torch.tensor(
            [], dtype=vllm_config.model_config.dtype
        ).element_size()
        self._block_size_bytes = hidden_dim * element_size

        num_ec_blocks = int(ec_config.get_from_extra_config("num_ec_blocks", 256))
        self._num_ec_blocks = num_ec_blocks

        if role == ECConnectorRole.WORKER:
            self._init_worker(vllm_config, ec_config, num_ec_blocks)
            return

        # Scheduler role below.
        self._zmq_ctx: zmq.Context | None = None
        self._region: ECSharedRegion | None = None
        self._nixl: NixlWrapper | None = None  # type: ignore[type-arg]
        self._reg_descs = None

        self._compat_hash = compute_ec_compatibility_hash(
            vllm_version=_vllm_version(),
            model=str(vllm_config.model_config.model),
            dtype=str(vllm_config.model_config.dtype),
            block_size_bytes=self._block_size_bytes,
        )

        if self.is_consumer:
            self._init_consumer(vllm_config, ec_config, num_ec_blocks)
        if self.is_producer:
            self._init_producer(vllm_config, ec_config, num_ec_blocks)

    def _init_worker(
        self,
        vllm_config: "VllmConfig",
        ec_config,
        num_ec_blocks: int,
    ) -> None:
        self._region = ECSharedRegion(
            instance_id=ec_config.engine_id,
            num_blocks=num_ec_blocks,
            block_size_bytes=self._block_size_bytes,
        )
        if is_pin_memory_available() and vllm_config.parallel_config.rank == 0:
            self._region.pin_memory()
        self._cpu_blocks = self._region.blocks

        self._copy_stream: torch.cuda.Stream | None = None
        self._copy_event: torch.Event | None = None

    def _init_consumer(
        self,
        vllm_config: "VllmConfig",
        ec_config,
        num_ec_blocks: int,
    ) -> None:
        # Same mmap workers use; creator/joiner race handled inside ECSharedRegion.
        self._region = ECSharedRegion(
            instance_id=ec_config.engine_id,
            num_blocks=num_ec_blocks,
            block_size_bytes=self._block_size_bytes,
        )

        # NIXL agent + region registration.  Used only to export our
        # metadata/descriptor to producer peers; no outgoing xfers.
        if NixlWrapper is not None:
            agent_cfg = (
                nixl_agent_config(num_threads=1, capture_telemetry=False)
                if nixl_agent_config is not None
                else None
            )
            self._nixl = NixlWrapper(str(uuid.uuid4()), agent_cfg)
            mem_type = current_platform.get_nixl_memory_type()
            # Work out the base pointer of the mmap.  ECSharedRegion exposes
            # it via ``self._region.blocks.data_ptr()`` (flat view over the
            # whole mmap).
            base_ptr = self._region.blocks.data_ptr()
            reg_tuples = [(base_ptr, self._region.total_size_bytes, 0, "")]
            self._reg_descs = self._nixl.get_reg_descs(reg_tuples, mem_type)
            self._nixl.register_memory(self._reg_descs)
            self._mem_type = mem_type

            # xfer_descs over every block slot, for producer peers to address.
            xfer_tuples = [
                (
                    base_ptr + i * self._block_size_bytes,
                    self._block_size_bytes,
                    0,
                )
                for i in range(num_ec_blocks)
            ]

            self._agent_metadata: bytes = self._nixl.get_agent_metadata()
            # Serialize xfer tuples for inclusion in XferReq.  Producer
            # rebuilds xfer_descs from these via get_xfer_descs.
            self._mem_descriptor_bytes: bytes = msgspec.msgpack.encode(xfer_tuples)
        else:
            logger.warning(
                "NIXL is not available; ECCPUConnector consumer scheduler "
                "will not be able to receive encodings via NIXL transfer."
            )
            self._mem_type = None
            self._agent_metadata = b""
            self._mem_descriptor_bytes = b""

        # Scheduler state.
        self._encoding_map: dict[str, list[int]] = {}
        self._ready: set[str] = set()
        self._dealers: dict[tuple[str, int], zmq.Socket] = {}
        self._zmq_ctx = zmq.Context.instance()
        self._nixl_agent_name = self._nixl.name if self._nixl is not None else ""

    def _init_producer(
        self,
        vllm_config: "VllmConfig",
        ec_config,
        num_ec_blocks: int,
    ) -> None:
        # Producer-side init — scaffolded here; listener thread added in Task 7.
        self._encodings: dict[str, torch.Tensor] = {}
        self._waiting: dict[str, tuple[bytes, XferReq]] = {}
        if NixlWrapper is not None:
            agent_cfg = (
                nixl_agent_config(num_threads=1, capture_telemetry=False)
                if nixl_agent_config is not None
                else None
            )
            self._nixl = NixlWrapper(str(uuid.uuid4()), agent_cfg)
        else:
            logger.warning(
                "NIXL is not available; ECCPUConnector producer scheduler "
                "will not be able to send encodings via NIXL transfer."
            )
        self._mem_type = current_platform.get_nixl_memory_type()
        self._zmq_ctx = zmq.Context.instance()
        self._router: zmq.Socket | None = None
        self._listener_thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None

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

    def shutdown(self) -> None:
        if self.role != ECConnectorRole.SCHEDULER:
            return
        stop_event: threading.Event | None = getattr(self, "_stop_event", None)
        if self.is_producer and stop_event is not None:
            stop_event.set()
            listener: threading.Thread | None = getattr(self, "_listener_thread", None)
            if listener is not None:
                listener.join(timeout=5)
            router: zmq.Socket | None = getattr(self, "_router", None)
            if router is not None:
                router.close(linger=0)
        if self.is_consumer:
            for sock in getattr(self, "_dealers", {}).values():
                sock.close(linger=0)
        if self._nixl is not None and self._reg_descs is not None:
            try:
                self._nixl.deregister_memory(self._reg_descs)
            except Exception:  # noqa: BLE001
                logger.debug("ec: deregister_memory raised", exc_info=True)
        if self._region is not None:
            self._region.cleanup()
