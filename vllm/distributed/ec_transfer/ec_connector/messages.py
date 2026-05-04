# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wire messages and compatibility hashing for the EC NIXL connector.

Increment EC_CONNECTOR_VERSION whenever the wire schema or NIXL protocol
changes in a way that breaks peer interoperability.
"""

import hashlib

import msgspec

EC_CONNECTOR_VERSION: int = 1


class XferReq(
    msgspec.Struct,
    tag="req",  # type: ignore[call-arg]
):
    """Consumer → Producer request to transfer the encoding for ``mm_hash``.

    The consumer allocates ``dst_block_indices`` locally, registers its mmap
    region with NIXL, and sends its agent metadata + xfer descriptor bytes so
    the producer can load the consumer as a NIXL peer and issue a WRITE that
    lands at exactly those destination block offsets.
    """

    mm_hash: str
    dst_block_indices: list[int]
    consumer_agent_name: str
    consumer_nixl_metadata: bytes
    consumer_mem_descriptor: bytes
    compatibility_hash: str
    connector_version: int = EC_CONNECTOR_VERSION


class XferAck(
    msgspec.Struct,
    tag="ack",  # type: ignore[call-arg]
):
    """Producer → Consumer completion signal.

    ``ok=False`` means the producer could not complete the transfer (NIXL
    failure, length mismatch, compatibility hash mismatch, etc.).  The
    consumer frees the destination blocks and lets the engine decide.
    """

    mm_hash: str
    ok: bool


def compute_ec_compatibility_hash(
    vllm_version: str,
    model: str,
    dtype: str,
    block_size_bytes: int,
) -> str:
    """SHA-256 of factors that must match for two EC peers to interoperate.

    Producer computes the same hash locally; ``XferReq.compatibility_hash``
    must match, otherwise the producer returns ``XferAck(ok=False)``.
    """

    h = hashlib.sha256()
    parts: list[str] = [
        vllm_version,
        str(EC_CONNECTOR_VERSION),
        model,
        dtype,
        str(block_size_bytes),
    ]
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")  # separator so ab|c != a|bc
    return h.hexdigest()


__all__ = [
    "EC_CONNECTOR_VERSION",
    "XferAck",
    "XferReq",
    "compute_ec_compatibility_hash",
]
