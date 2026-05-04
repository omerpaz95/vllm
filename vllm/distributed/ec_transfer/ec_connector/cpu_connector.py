# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import msgspec
import msgspec.msgpack
import torch
import zmq

import vllm.envs as envs
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
    XferAck,
    XferReq,
    compute_ec_compatibility_hash,
)
from vllm.distributed.nixl_utils import NixlWrapper, nixl_agent_config
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.network_utils import make_zmq_path, make_zmq_socket
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
        self._device_id = 0

        self._zmq_ctx = zmq.Context.instance()
        host = envs.VLLM_EC_SIDE_CHANNEL_HOST
        port = envs.VLLM_EC_SIDE_CHANNEL_PORT
        path = make_zmq_path("tcp", host, port)
        self._router = make_zmq_socket(self._zmq_ctx, path, zmq.ROUTER, bind=True)
        self._router.setsockopt(zmq.RCVTIMEO, 500)
        self._stop_event = threading.Event()
        self._listener_thread = threading.Thread(
            target=self._run_listener,
            name="ec-listener",
            daemon=True,
        )
        self._listener_thread.start()

    def _run_listener(self) -> None:
        assert self._router is not None
        assert self._stop_event is not None
        decoder = msgspec.msgpack.Decoder(XferReq)
        while not self._stop_event.is_set():
            try:
                frames = self._router.recv_multipart()
                # DEALER→ROUTER: [identity, payload] (2 frames)
                # REQ→ROUTER:   [identity, b"", payload] (3 frames)
                if len(frames) == 2:
                    identity, payload = frames
                elif len(frames) == 3:
                    identity, _, payload = frames
                else:
                    logger.warning("ec: unexpected frame count %d", len(frames))
                    continue
            except zmq.Again:
                continue
            except zmq.ContextTerminated:
                return
            try:
                req = decoder.decode(payload)
            except (msgspec.DecodeError, msgspec.ValidationError):
                logger.warning("ec: dropped malformed XferReq")
                continue

            if req.compatibility_hash != self._compat_hash:
                logger.warning(
                    "ec: compatibility hash mismatch from consumer %s",
                    req.consumer_agent_name,
                )
                self._send_ack(identity, req.mm_hash, ok=False)
                continue

            tensor = self._encodings.get(req.mm_hash)
            if tensor is None:
                self._waiting[req.mm_hash] = (identity, req)
                continue

            ok = self._do_nixl_xfer(req, tensor)
            self._send_ack(identity, req.mm_hash, ok)

    def _send_ack(self, identity: bytes, mm_hash: str, ok: bool) -> None:
        assert self._router is not None
        payload = msgspec.msgpack.encode(XferAck(mm_hash=mm_hash, ok=ok))
        # DEALER receives [payload]; ROUTER must send [identity, payload].
        with contextlib.suppress(zmq.ContextTerminated):
            self._router.send_multipart([identity, payload])

    def _do_nixl_xfer(self, req: "XferReq", tensor: "torch.Tensor") -> bool:
        assert self._nixl is not None
        n = len(req.dst_block_indices)
        expected = n * self._block_size_bytes
        if tensor.nbytes != expected:
            logger.warning(
                "ec: size mismatch for mm_hash=%s: tensor=%d expected=%d",
                req.mm_hash,
                tensor.nbytes,
                expected,
            )
            return False

        reg_tuples = [(tensor.data_ptr(), tensor.nbytes, self._device_id, "")]
        reg_descs = self._nixl.get_reg_descs(reg_tuples, self._mem_type)
        self._nixl.register_memory(reg_descs)

        remote_agent_name: str | None = None
        local_dlist: int | None = None
        remote_dlist: int | None = None
        xfer_handle: int | None = None
        try:
            remote_agent_name = self._nixl.add_remote_agent(req.consumer_nixl_metadata)

            local_tuples = [
                (
                    tensor.data_ptr() + i * self._block_size_bytes,
                    self._block_size_bytes,
                    self._device_id,
                )
                for i in range(n)
            ]
            local_descs = self._nixl.get_xfer_descs(local_tuples, self._mem_type)
            local_dlist = self._nixl.prep_xfer_dlist("NIXL_INIT_AGENT", local_descs)

            # Consumer sent a msgpack-encoded list[tuple[int,int,int]].
            # msgpack turns tuples into lists on the wire; convert back.
            raw_remote = msgspec.msgpack.decode(req.consumer_mem_descriptor)
            remote_tuples = [tuple(item) for item in raw_remote]
            remote_descs = self._nixl.get_xfer_descs(remote_tuples, self._mem_type)
            remote_dlist = self._nixl.prep_xfer_dlist(remote_agent_name, remote_descs)

            xfer_handle = self._nixl.make_prepped_xfer(
                "WRITE",
                local_dlist,
                list(range(n)),
                remote_dlist,
                req.dst_block_indices,
                notif_msg=b"",
            )
            self._nixl.transfer(xfer_handle)
            while True:
                state = self._nixl.check_xfer_state(xfer_handle)
                if state == "DONE":
                    return True
                if state != "PROC":
                    logger.warning(
                        "ec: NIXL xfer ended in state=%s for mm_hash=%s",
                        state,
                        req.mm_hash,
                    )
                    return False
                time.sleep(0.0005)
        except Exception as e:  # noqa: BLE001
            logger.warning("ec: NIXL xfer failed for mm_hash=%s: %s", req.mm_hash, e)
            return False
        finally:
            if xfer_handle is not None:
                try:
                    self._nixl.release_xfer_handle(xfer_handle)
                except Exception:  # noqa: BLE001
                    logger.debug("ec: release_xfer_handle raised", exc_info=True)
            if local_dlist is not None:
                try:
                    self._nixl.release_dlist_handle(local_dlist)
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "ec: release_dlist_handle(local) raised", exc_info=True
                    )
            if remote_dlist is not None:
                try:
                    self._nixl.release_dlist_handle(remote_dlist)
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "ec: release_dlist_handle(remote) raised", exc_info=True
                    )
            if remote_agent_name is not None:
                try:
                    self._nixl.remove_remote_agent(remote_agent_name)
                except Exception:  # noqa: BLE001
                    logger.debug("ec: remove_remote_agent raised", exc_info=True)
            try:
                self._nixl.deregister_memory(reg_descs)
            except Exception:  # noqa: BLE001
                logger.debug("ec: deregister_memory raised", exc_info=True)

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
        if not self.is_consumer:
            return False
        self._drain_acks()
        if identifier in self._ready:
            return True
        if identifier in self._encoding_map:
            return None
        return False

    def _drain_acks(self) -> None:
        decoder = msgspec.msgpack.Decoder(XferAck)
        for sock in self._dealers.values():
            while True:
                try:
                    payload = sock.recv(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
                try:
                    ack = decoder.decode(payload)
                except (msgspec.DecodeError, msgspec.ValidationError):
                    logger.warning("ec: dropped malformed XferAck")
                    continue
                self._complete(ack.mm_hash, ack.ok)

    def _complete(self, mm_hash: str, ok: bool) -> None:
        if mm_hash not in self._encoding_map:
            return  # stale / duplicate ack
        if ok:
            self._ready.add(mm_hash)
        else:
            indices = self._encoding_map.pop(mm_hash)
            assert self._region is not None
            self._region.free(indices)

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
