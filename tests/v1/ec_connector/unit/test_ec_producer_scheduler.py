# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the producer (is_producer=True) side of ECCPUConnector."""

import time
import uuid
from unittest.mock import patch

import msgspec
import pytest
import torch
import zmq

from tests.v1.ec_connector._fakes import FakeNixlWrapper, reset_agents
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
from vllm.distributed.ec_transfer.ec_connector.cpu_connector import (
    ECCPUConnector,
)
from vllm.distributed.ec_transfer.ec_connector.messages import (
    XferReq,
)


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
    engine_id = ""
    ec_role = "ec_producer"
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
    def __init__(self, engine_id: str, role: str = "ec_producer", extra=None):
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


def _free_port() -> int:
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def producer_config(monkeypatch):
    port = _free_port()
    monkeypatch.setattr("vllm.envs.VLLM_EC_SIDE_CHANNEL_HOST", "127.0.0.1")
    monkeypatch.setattr("vllm.envs.VLLM_EC_SIDE_CHANNEL_PORT", port)
    cfg = _VllmConfig(
        engine_id=f"prod-{uuid.uuid4()}",
        role="ec_producer",
        extra={"num_ec_blocks": 4},
    )
    return cfg, port


def test_producer_starts_listener_thread(producer_config):
    cfg, port = producer_config
    with patch(
        "vllm.distributed.ec_transfer.ec_connector.cpu_connector.NixlWrapper",
        FakeNixlWrapper,
    ):
        conn = ECCPUConnector(cfg, ECConnectorRole.SCHEDULER)
        try:
            assert conn._listener_thread is not None
            assert conn._listener_thread.is_alive()
            assert conn._router is not None
        finally:
            conn.shutdown()
            assert not conn._listener_thread.is_alive()


def test_producer_parks_xfer_req_when_encoding_absent(producer_config):
    cfg, port = producer_config
    with patch(
        "vllm.distributed.ec_transfer.ec_connector.cpu_connector.NixlWrapper",
        FakeNixlWrapper,
    ):
        conn = ECCPUConnector(cfg, ECConnectorRole.SCHEDULER)
        try:
            ctx = zmq.Context.instance()
            dealer = ctx.socket(zmq.DEALER)
            dealer.connect(f"tcp://127.0.0.1:{port}")

            req = XferReq(
                mm_hash="notyet",
                dst_block_indices=[0, 1, 2],
                consumer_agent_name="unused",
                consumer_nixl_metadata=b"",
                consumer_mem_descriptor=b"",
                compatibility_hash=conn._compat_hash,
            )
            dealer.send(msgspec.msgpack.encode(req))

            # Listener should park it in _waiting without acking.
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and "notyet" not in conn._waiting:
                time.sleep(0.01)
            assert "notyet" in conn._waiting

            # And no ack came back.
            dealer.setsockopt(zmq.RCVTIMEO, 100)
            with pytest.raises(zmq.Again):
                dealer.recv()
            dealer.close(linger=0)
        finally:
            conn.shutdown()
