# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the consumer (is_consumer=True) side of ECCPUConnector."""

import uuid
from unittest.mock import patch

import pytest
import torch
import zmq

from tests.v1.ec_connector._fakes import FakeNixlWrapper, reset_agents
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
from vllm.distributed.ec_transfer.ec_connector.cpu_connector import (
    ECCPUConnector,
)


# Build a minimal VllmConfig stub sufficient for ECCPUConnector scheduler init.
class _MMConfig:
    def get_inputs_embeds_size(self) -> int:
        return 32

    model = "fake-model"
    dtype = torch.bfloat16


class _ParallelConfig:
    world_size = 1
    rank = 0


class _ECConfig:
    ec_connector = "ECCPUConnector"
    engine_id = ""  # filled per test
    ec_role = "ec_consumer"
    ec_connector_extra_config: dict = {}

    @property
    def is_ec_producer(self) -> bool:
        return self.ec_role in ("ec_producer", "ec_both")

    @property
    def is_ec_consumer(self) -> bool:
        return self.ec_role in ("ec_consumer", "ec_both")

    def get_from_extra_config(self, key, default):
        return self.ec_connector_extra_config.get(key, default)


class _VllmConfig:
    def __init__(
        self,
        engine_id: str,
        extra: dict | None = None,
        role: str = "ec_consumer",
    ):
        self.model_config = _MMConfig()
        self.parallel_config = _ParallelConfig()
        self.ec_transfer_config = _ECConfig()
        self.ec_transfer_config.engine_id = engine_id
        self.ec_transfer_config.ec_role = role
        if extra is not None:
            self.ec_transfer_config.ec_connector_extra_config = extra


@pytest.fixture(autouse=True)
def _reset_fakes():
    reset_agents()
    yield
    reset_agents()


@pytest.fixture
def vllm_config():
    return _VllmConfig(
        engine_id=f"test-{uuid.uuid4()}",
        extra={"default_encoder_node": "127.0.0.1:65000", "num_ec_blocks": 4},
    )


def test_consumer_scheduler_init_creates_region_and_nixl_agent(vllm_config):
    with patch(
        "vllm.distributed.ec_transfer.ec_connector.cpu_connector.NixlWrapper",
        FakeNixlWrapper,
    ):
        conn = ECCPUConnector(vllm_config, ECConnectorRole.SCHEDULER)
        try:
            assert conn._region is not None
            assert conn._region.num_blocks == 4
            # NIXL agent created and mmap registered.
            assert conn._nixl is not None
            assert len(conn._nixl._registrations) == 1
            reg = conn._nixl._registrations[0]
            assert reg.size == conn._region.total_size_bytes
            # State dicts initialized.
            assert conn._encoding_map == {}
            assert conn._ready == set()
            assert conn._dealers == {}
        finally:
            conn.shutdown()


def test_consumer_scheduler_exports_agent_metadata_and_descriptor(vllm_config):
    with patch(
        "vllm.distributed.ec_transfer.ec_connector.cpu_connector.NixlWrapper",
        FakeNixlWrapper,
    ):
        conn = ECCPUConnector(vllm_config, ECConnectorRole.SCHEDULER)
        try:
            assert isinstance(conn._agent_metadata, bytes)
            assert conn._agent_metadata == conn._nixl.name.encode("utf-8")
            assert isinstance(conn._mem_descriptor_bytes, bytes)
            assert len(conn._mem_descriptor_bytes) > 0
        finally:
            conn.shutdown()


def test_has_cache_item_returns_false_for_unknown(vllm_config):
    with patch(
        "vllm.distributed.ec_transfer.ec_connector.cpu_connector.NixlWrapper",
        FakeNixlWrapper,
    ):
        conn = ECCPUConnector(vllm_config, ECConnectorRole.SCHEDULER)
        try:
            assert conn.has_cache_item("nope") is False
        finally:
            conn.shutdown()


def test_has_cache_item_none_then_true_on_ack(vllm_config):
    with patch(
        "vllm.distributed.ec_transfer.ec_connector.cpu_connector.NixlWrapper",
        FakeNixlWrapper,
    ):
        conn = ECCPUConnector(vllm_config, ECConnectorRole.SCHEDULER)
        try:
            conn._encoding_map["h"] = [0, 1, 2]
            assert conn.has_cache_item("h") is None
            conn._complete("h", ok=True)
            assert conn.has_cache_item("h") is True
        finally:
            conn.shutdown()


def test_complete_on_fail_frees_blocks(vllm_config):
    with patch(
        "vllm.distributed.ec_transfer.ec_connector.cpu_connector.NixlWrapper",
        FakeNixlWrapper,
    ):
        conn = ECCPUConnector(vllm_config, ECConnectorRole.SCHEDULER)
        try:
            indices = conn._region.alloc(3)
            conn._encoding_map["h"] = indices
            conn._complete("h", ok=False)
            assert "h" not in conn._encoding_map
            # Blocks returned to the free list.
            assert set(indices) <= set(conn._region._free_blocks)
        finally:
            conn.shutdown()


class _FakeMMFeature:
    def __init__(self, mm_hash: str, length: int):
        self.mm_hash = mm_hash
        self.identifier = mm_hash

        class _Pos:
            length: int

        self.mm_position = _Pos()
        self.mm_position.length = length


class _FakeRequest:
    def __init__(self, features):
        self.mm_features = features
        self.kv_transfer_params = None


def test_update_state_after_alloc_sends_xfer_req(vllm_config):
    with patch(
        "vllm.distributed.ec_transfer.ec_connector.cpu_connector.NixlWrapper",
        FakeNixlWrapper,
    ):
        conn = ECCPUConnector(vllm_config, ECConnectorRole.SCHEDULER)
        try:
            # Start a ROUTER to stand in for the E node.
            import msgspec as _ms

            ctx = zmq.Context.instance()
            router = ctx.socket(zmq.ROUTER)
            router.bind("tcp://127.0.0.1:65000")
            router.setsockopt(zmq.RCVTIMEO, 2000)

            from vllm.distributed.ec_transfer.ec_connector.messages import XferReq

            req = _FakeRequest([_FakeMMFeature("hash-1", length=3)])
            conn.update_state_after_alloc(req, 0)

            frames = router.recv_multipart()
            # DEALER→ROUTER is 2 frames: [identity, payload]
            payload = frames[-1]
            decoded = _ms.msgpack.decode(payload, type=XferReq)
            assert decoded.mm_hash == "hash-1"
            assert decoded.dst_block_indices == [0, 1, 2]
            assert decoded.compatibility_hash == conn._compat_hash
            assert decoded.consumer_nixl_metadata == conn._agent_metadata
            assert decoded.consumer_mem_descriptor == conn._mem_descriptor_bytes
            router.close(linger=0)
        finally:
            conn.shutdown()


def test_update_state_after_alloc_is_idempotent(vllm_config):
    with patch(
        "vllm.distributed.ec_transfer.ec_connector.cpu_connector.NixlWrapper",
        FakeNixlWrapper,
    ):
        conn = ECCPUConnector(vllm_config, ECConnectorRole.SCHEDULER)
        try:
            ctx = zmq.Context.instance()
            router = ctx.socket(zmq.ROUTER)
            router.bind("tcp://127.0.0.1:65001")
            router.setsockopt(zmq.RCVTIMEO, 500)

            # Override resolve_encoder_address via extra config.
            conn._vllm_config.ec_transfer_config.ec_connector_extra_config[
                "default_encoder_node"
            ] = "127.0.0.1:65001"

            feat = _FakeMMFeature("dup", length=2)
            req = _FakeRequest([feat])
            conn.update_state_after_alloc(req, 0)
            conn.update_state_after_alloc(req, 0)  # second call: no-op

            router.recv_multipart()  # only one XferReq expected
            with pytest.raises(zmq.Again):
                router.recv_multipart()
            router.close(linger=0)
        finally:
            conn.shutdown()


def test_build_connector_meta_hands_off_ready_entries(vllm_config):
    with patch(
        "vllm.distributed.ec_transfer.ec_connector.cpu_connector.NixlWrapper",
        FakeNixlWrapper,
    ):
        conn = ECCPUConnector(vllm_config, ECConnectorRole.SCHEDULER)
        try:
            conn._encoding_map["a"] = [0, 1]
            conn._encoding_map["b"] = [2, 3]
            conn._ready = {"a"}  # only 'a' is ready

            class _SO:
                pass

            meta = conn.build_connector_meta(_SO())

            assert meta.mm_hash_to_cpu_blocks == {"a": [0, 1]}
            # 'a' handed off and cleared; 'b' still pending.
            assert "a" not in conn._encoding_map
            assert conn._encoding_map == {"b": [2, 3]}
            assert conn._ready == set()
            # Second call returns empty mapping.
            meta2 = conn.build_connector_meta(_SO())
            assert meta2.mm_hash_to_cpu_blocks == {}
        finally:
            conn.shutdown()
