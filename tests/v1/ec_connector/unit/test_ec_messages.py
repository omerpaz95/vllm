# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import msgspec

from vllm.distributed.ec_transfer.ec_connector.messages import (
    EC_CONNECTOR_VERSION,
    XferAck,
    XferReq,
    compute_ec_compatibility_hash,
)


def test_xfer_req_roundtrip():
    req = XferReq(
        mm_hash="abc",
        dst_block_indices=[3, 7, 9],
        consumer_agent_name="agent-X",
        consumer_nixl_metadata=b"\x01\x02",
        consumer_mem_descriptor=b"\x03\x04",
        compatibility_hash="0" * 64,
    )
    wire = msgspec.msgpack.encode(req)
    back = msgspec.msgpack.decode(wire, type=XferReq)
    assert back == req
    assert back.connector_version == EC_CONNECTOR_VERSION


def test_xfer_ack_roundtrip():
    ack = XferAck(mm_hash="abc", ok=True)
    wire = msgspec.msgpack.encode(ack)
    back = msgspec.msgpack.decode(wire, type=XferAck)
    assert back == ack


def test_compatibility_hash_stable():
    h1 = compute_ec_compatibility_hash(
        vllm_version="0.1.0", model="m", dtype="bfloat16", block_size_bytes=8192
    )
    h2 = compute_ec_compatibility_hash(
        vllm_version="0.1.0", model="m", dtype="bfloat16", block_size_bytes=8192
    )
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex


def test_compatibility_hash_changes_on_factor_change():
    h1 = compute_ec_compatibility_hash(
        vllm_version="0.1.0", model="m", dtype="bfloat16", block_size_bytes=8192
    )
    h2 = compute_ec_compatibility_hash(
        vllm_version="0.1.0", model="m", dtype="float16", block_size_bytes=8192
    )
    assert h1 != h2
