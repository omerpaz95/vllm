# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end test with real ZMQ sockets and FakeNixlWrapper.

Wires one producer ECCPUConnector and one consumer ECCPUConnector
inside a single process, drives an XferReq through the full pipeline,
and asserts the consumer's mmap receives the producer's tensor bytes.
"""

import time
import uuid
from unittest.mock import patch

import pytest
import torch

from tests.v1.ec_connector._fakes import FakeNixlWrapper, reset_agents
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
from vllm.distributed.ec_transfer.ec_connector.cpu_connector import (
    ECCPUConnector,
)


class _MM:
    def get_inputs_embeds_size(self):
        return 16

    model = "m"
    dtype = torch.bfloat16


class _P:
    world_size = 1
    rank = 0


class _EC:
    ec_connector = "ECCPUConnector"
    engine_id = ""
    ec_role = ""
    ec_connector_extra_config: dict = {}

    @property
    def is_ec_producer(self):
        return self.ec_role in ("ec_producer", "ec_both")

    @property
    def is_ec_consumer(self):
        return self.ec_role in ("ec_consumer", "ec_both")

    def get_from_extra_config(self, k, d):
        return self.ec_connector_extra_config.get(k, d)


class _V:
    def __init__(self, engine_id, role, extra):
        self.model_config = _MM()
        self.parallel_config = _P()
        self.ec_transfer_config = _EC()
        self.ec_transfer_config.engine_id = engine_id
        self.ec_transfer_config.ec_role = role
        self.ec_transfer_config.ec_connector_extra_config = extra or {}


def _free_port():
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class _FakeFeat:
    def __init__(self, h, L):
        self.mm_hash = h
        self.identifier = h

        class _Pos:
            length: int = 0

        self.mm_position = _Pos()
        self.mm_position.length = L


class _FakeReq:
    def __init__(self, feats):
        self.mm_features = feats
        self.kv_transfer_params = None


@pytest.fixture(autouse=True)
def _reset():
    reset_agents()
    yield
    reset_agents()


def _poll_until(pred, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_end_to_end_single_transfer(monkeypatch):
    port = _free_port()
    monkeypatch.setattr("vllm.envs.VLLM_EC_SIDE_CHANNEL_HOST", "127.0.0.1")
    monkeypatch.setattr("vllm.envs.VLLM_EC_SIDE_CHANNEL_PORT", port)

    prod_cfg = _V(f"p-{uuid.uuid4()}", "ec_producer", extra={"num_ec_blocks": 8})
    cons_cfg = _V(
        f"c-{uuid.uuid4()}",
        "ec_consumer",
        extra={"num_ec_blocks": 8, "default_encoder_node": f"127.0.0.1:{port}"},
    )

    with patch(
        "vllm.distributed.ec_transfer.ec_connector.cpu_connector.NixlWrapper",
        FakeNixlWrapper,
    ):
        prod = ECCPUConnector(prod_cfg, ECConnectorRole.SCHEDULER)
        cons = ECCPUConnector(cons_cfg, ECConnectorRole.SCHEDULER)
        try:
            # Producer owns an encoding for mm_hash="h".
            n = 3
            bs = prod._block_size_bytes
            tensor = torch.arange(n * bs, dtype=torch.uint8).contiguous()
            prod._encodings["h"] = tensor

            # Consumer kicks off the fetch.
            req = _FakeReq([_FakeFeat("h", L=n)])
            cons.update_state_after_alloc(req, 0)

            assert _poll_until(lambda: cons.has_cache_item("h") is True), (
                "consumer never saw transfer complete"
            )

            class _SO:
                pass

            meta = cons.build_connector_meta(_SO())
            assert "h" in meta.mm_hash_to_cpu_blocks
            block_indices = meta.mm_hash_to_cpu_blocks["h"]

            arrived = cons._region.blocks[block_indices].flatten()
            assert torch.equal(arrived, tensor.view(n, bs).flatten())
        finally:
            cons.shutdown()
            prod.shutdown()


def test_end_to_end_wait_then_drain(monkeypatch):
    port = _free_port()
    monkeypatch.setattr("vllm.envs.VLLM_EC_SIDE_CHANNEL_HOST", "127.0.0.1")
    monkeypatch.setattr("vllm.envs.VLLM_EC_SIDE_CHANNEL_PORT", port)

    prod_cfg = _V(f"p-{uuid.uuid4()}", "ec_producer", extra={"num_ec_blocks": 8})
    cons_cfg = _V(
        f"c-{uuid.uuid4()}",
        "ec_consumer",
        extra={"num_ec_blocks": 8, "default_encoder_node": f"127.0.0.1:{port}"},
    )

    with patch(
        "vllm.distributed.ec_transfer.ec_connector.cpu_connector.NixlWrapper",
        FakeNixlWrapper,
    ):
        prod = ECCPUConnector(prod_cfg, ECConnectorRole.SCHEDULER)
        cons = ECCPUConnector(cons_cfg, ECConnectorRole.SCHEDULER)
        try:
            # Consumer requests BEFORE producer has the data.
            req = _FakeReq([_FakeFeat("late", L=2)])
            cons.update_state_after_alloc(req, 0)

            assert _poll_until(lambda: "late" in prod._waiting), (
                "producer never parked request"
            )

            # Now populate and drain.
            n, bs = 2, prod._block_size_bytes
            tensor = torch.full((n * bs,), 5, dtype=torch.uint8).contiguous()
            prod._encodings["late"] = tensor
            prod._drain_waiting("late")

            assert _poll_until(lambda: cons.has_cache_item("late") is True)

            class _SO:
                pass

            meta = cons.build_connector_meta(_SO())
            idx = meta.mm_hash_to_cpu_blocks["late"]
            arrived = cons._region.blocks[idx].flatten()
            assert torch.all(arrived == 5)
        finally:
            cons.shutdown()
            prod.shutdown()
