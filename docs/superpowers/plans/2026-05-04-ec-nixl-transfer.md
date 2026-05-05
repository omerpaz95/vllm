# EC Connector — NIXL Transfer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a NIXL-based remote transport between two `ECCPUConnector` instances so the consumer (PD) can fetch an encoder output by `mm_hash` from a remote producer (E) node over ZMQ + NIXL, landing it directly in the consumer's CPU encoder-cache mmap.

**Architecture:** Scheduler-only NIXL; control plane is ZMQ scheduler↔scheduler (DEALER on consumer, ROUTER on producer); data plane is NIXL CPU→CPU; hand-off-and-forget state on the consumer; transient per-transfer NIXL metadata on the producer; block allocator lives on `ECSharedRegion`.

**Tech Stack:** Python 3.12, pytest, zmq (pyzmq), msgspec, torch, NIXL via `vllm.distributed.nixl_utils.NixlWrapper`, mmap-backed `/dev/shm` via `ECSharedRegion`.

**Spec:** `docs/superpowers/specs/2026-05-04-ec-nixl-transfer-design.md`

**Environment note:** Use `uv` and the repo's venv. All `python`/`pytest` commands go through `.venv/bin/python` (see `AGENTS.md`). Never use system `python3` or bare `pip`.

---

## File structure

**New files (produced by this plan):**

- `vllm/distributed/ec_transfer/ec_connector/messages.py` — msgspec structs + compat hash
- `tests/v1/ec_connector/_fakes.py` — `FakeNixlWrapper`
- `tests/v1/ec_connector/unit/test_ec_shared_region_alloc.py`
- `tests/v1/ec_connector/unit/test_ec_messages.py`
- `tests/v1/ec_connector/unit/test_ec_consumer_scheduler.py`
- `tests/v1/ec_connector/unit/test_ec_producer_scheduler.py`
- `tests/v1/ec_connector/integration/__init__.py`
- `tests/v1/ec_connector/integration/test_ec_zmq_loopback.py`

**Modified files:**

- `vllm/distributed/ec_transfer/ec_connector/ec_shared_region.py`
- `vllm/distributed/ec_transfer/ec_connector/cpu_connector.py`
- `vllm/distributed/ec_transfer/ec_connector/base.py`
- `vllm/envs.py`

**Out of scope / unchanged:**

- Any worker-process code
- `factory.py`, `example_connector.py`
- `save_caches` on producer worker (future PR)

---

## Task 1: Add `alloc` / `free` to `ECSharedRegion`

**Files:**

- Modify: `vllm/distributed/ec_transfer/ec_connector/ec_shared_region.py`
- Create: `tests/v1/ec_connector/unit/test_ec_shared_region_alloc.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/v1/ec_connector/unit/test_ec_shared_region_alloc.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import uuid

import pytest

from vllm.distributed.ec_transfer.ec_connector.ec_shared_region import (
    ECSharedRegion,
)


@pytest.fixture
def region():
    instance_id = f"test-{uuid.uuid4()}"
    r = ECSharedRegion(instance_id=instance_id, num_blocks=8, block_size_bytes=64)
    yield r
    r.cleanup()


def test_alloc_returns_sequential_indices(region):
    idx = region.alloc(3)
    assert idx == [0, 1, 2]


def test_alloc_advances_over_multiple_calls(region):
    assert region.alloc(2) == [0, 1]
    assert region.alloc(3) == [2, 3, 4]


def test_free_returns_indices_to_pool(region):
    a = region.alloc(3)
    region.free(a)
    # Freed indices go to the tail; next alloc pulls from the head first.
    b = region.alloc(4)
    assert b == [3, 4, 5, 6]
    c = region.alloc(3)
    assert sorted(c) == [0, 1, 2]                    # the freed ones come back


def test_alloc_raises_on_exhaustion(region):
    region.alloc(8)
    with pytest.raises(RuntimeError, match="exhausted"):
        region.alloc(1)


def test_free_list_not_shared_across_instances(tmp_path, monkeypatch):
    # Two instances of the same region (simulating two workers) get
    # independent free-lists. Allocation in one must not affect the other.
    instance_id = f"test-{uuid.uuid4()}"
    a = ECSharedRegion(instance_id=instance_id, num_blocks=4, block_size_bytes=16)
    b = ECSharedRegion(instance_id=instance_id, num_blocks=4, block_size_bytes=16)
    try:
        assert a.alloc(2) == [0, 1]
        assert b.alloc(2) == [0, 1]                  # independent, not [2, 3]
    finally:
        a.cleanup()
        b.cleanup()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/v1/ec_connector/unit/test_ec_shared_region_alloc.py -v
```

Expected: FAIL with `AttributeError: 'ECSharedRegion' object has no attribute 'alloc'`.

- [ ] **Step 3: Implement `alloc` and `free` on `ECSharedRegion`**

In `vllm/distributed/ec_transfer/ec_connector/ec_shared_region.py`, add to the `__init__` (right after `self.is_pinned: bool = False`, before `self.blocks = ...`):

```python
        # Scheduler-only free list of block indices.  Workers that also
        # instantiate ECSharedRegion get their own (unused) copy.
        self._free_blocks: list[int] = list(range(num_blocks))
```

Then add two methods, immediately after `pin_memory`:

```python
    def alloc(self, n: int) -> list[int]:
        """Allocate ``n`` block indices from the free list.

        Returns a list of ints — these are the only way callers refer to
        a piece of the region, because indices serialize over ZMQ, flow
        to worker processes via connector metadata, and drive NIXL desc
        offset math.  For raw bytes, use ``region.blocks[indices]``.
        """
        if len(self._free_blocks) < n:
            raise RuntimeError(
                f"ECSharedRegion exhausted: need {n} blocks, "
                f"{len(self._free_blocks)} free"
            )
        out, self._free_blocks = self._free_blocks[:n], self._free_blocks[n:]
        return out

    def free(self, indices: list[int]) -> None:
        """Return ``indices`` to the free list.  Idempotent on the caller's side."""
        self._free_blocks.extend(indices)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/v1/ec_connector/unit/test_ec_shared_region_alloc.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Lint**

```bash
cd /home/omer/Project9_vllm/worktrees/ec_connector_nixl_transfer
pre-commit run --files \
  vllm/distributed/ec_transfer/ec_connector/ec_shared_region.py \
  tests/v1/ec_connector/unit/test_ec_shared_region_alloc.py
```

Expected: all hooks pass (ruff may autofix; re-run tests if it does).

- [ ] **Step 6: Commit**

```bash
git add vllm/distributed/ec_transfer/ec_connector/ec_shared_region.py \
  tests/v1/ec_connector/unit/test_ec_shared_region_alloc.py
git commit -m "$(cat <<'EOF'
Add alloc/free to ECSharedRegion

The block free-list belongs with the memory it tracks.  Scheduler-only
usage (workers never call alloc/free) so no locking is needed.
alloc() returns block indices (not tensor views) because that's what
downstream consumers — ZMQ payloads, connector metadata, NIXL descriptor
offset math — actually need.

Co-authored-by: Claude
EOF
)"
```

---

## Task 2: Add env vars `VLLM_EC_SIDE_CHANNEL_HOST` / `PORT`

**Files:**

- Modify: `vllm/envs.py`

- [ ] **Step 1: Read existing NIXL env vars for pattern**

```bash
grep -n "VLLM_NIXL_SIDE_CHANNEL" vllm/envs.py
```

Note the type-annotation block around line 189-190 and the `environment_variables` dict block around line 1363-1370 (exact lines may drift). The pattern is: declare type+default at module top, then provide a lambda-reader in the dict.

- [ ] **Step 2: Add type declarations**

In the module-level type block (alongside `VLLM_NIXL_SIDE_CHANNEL_HOST`, `VLLM_NIXL_SIDE_CHANNEL_PORT`), add:

```python
    VLLM_EC_SIDE_CHANNEL_HOST: str = "localhost"
    VLLM_EC_SIDE_CHANNEL_PORT: int = 5601
```

- [ ] **Step 3: Add dict entries**

In the `environment_variables` dict, alongside the existing `VLLM_NIXL_SIDE_CHANNEL_*` entries, add:

```python
    "VLLM_EC_SIDE_CHANNEL_HOST": lambda: os.getenv(
        "VLLM_EC_SIDE_CHANNEL_HOST", "localhost"
    ),
    "VLLM_EC_SIDE_CHANNEL_PORT": lambda: int(
        os.getenv("VLLM_EC_SIDE_CHANNEL_PORT", "5601")
    ),
```

(Port 5601 chosen one above the default NIXL port 5600 to avoid collision when both connectors run on the same host.)

- [ ] **Step 4: Verify via import**

```bash
.venv/bin/python -c "from vllm import envs; print(envs.VLLM_EC_SIDE_CHANNEL_HOST, envs.VLLM_EC_SIDE_CHANNEL_PORT)"
```

Expected output: `localhost 5601`.

- [ ] **Step 5: Lint**

```bash
pre-commit run --files vllm/envs.py
```

- [ ] **Step 6: Commit**

```bash
git add vllm/envs.py
git commit -m "$(cat <<'EOF'
Add VLLM_EC_SIDE_CHANNEL_{HOST,PORT} env vars

Producer ROUTER socket binds to these; consumer DEALERs connect to the
host:port advertised by each E node.  Default 5601 (one above NIXL)
so both connectors can coexist on one host in tests.

Co-authored-by: Claude
EOF
)"
```

---

## Task 3: Add no-op `shutdown()` to `ECConnectorBase`

**Files:**

- Modify: `vllm/distributed/ec_transfer/ec_connector/base.py`

- [ ] **Step 1: Add the method**

In `ECConnectorBase` (open `base.py`, find the class body, add the method after `request_finished`):

```python
    def shutdown(self) -> None:
        """Release connector resources.

        Subclasses override to close sockets, join threads, and release
        NIXL state.  The default is a no-op so callers can invoke it
        unconditionally.
        """
        return
```

- [ ] **Step 2: Verify base class still importable**

```bash
.venv/bin/python -c "from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorBase; c = ECConnectorBase.__abstractmethods__; print('abstract:', sorted(c))"
```

Expected: a non-empty set of abstract method names (the existing abstracts: `start_load_caches`, `save_caches`, `has_cache_item`, `update_state_after_alloc`, `build_connector_meta`). `shutdown` must NOT be in the set.

- [ ] **Step 3: Lint + commit**

```bash
pre-commit run --files vllm/distributed/ec_transfer/ec_connector/base.py
git add vllm/distributed/ec_transfer/ec_connector/base.py
git commit -m "$(cat <<'EOF'
Add ECConnectorBase.shutdown() no-op

Enables unconditional ``connector.shutdown()`` from teardown paths
without subclass checks.  Subclasses override to release resources.

Co-authored-by: Claude
EOF
)"
```

---

## Task 4: Create `messages.py` — wire schemas + compatibility hash

**Files:**

- Create: `vllm/distributed/ec_transfer/ec_connector/messages.py`
- Create: `tests/v1/ec_connector/unit/test_ec_messages.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/v1/ec_connector/unit/test_ec_messages.py`:

```python
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
    assert len(h1) == 64                           # sha256 hex


def test_compatibility_hash_changes_on_factor_change():
    h1 = compute_ec_compatibility_hash(
        vllm_version="0.1.0", model="m", dtype="bfloat16", block_size_bytes=8192
    )
    h2 = compute_ec_compatibility_hash(
        vllm_version="0.1.0", model="m", dtype="float16", block_size_bytes=8192
    )
    assert h1 != h2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/v1/ec_connector/unit/test_ec_messages.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'vllm.distributed.ec_transfer.ec_connector.messages'`.

- [ ] **Step 3: Implement `messages.py`**

Create `vllm/distributed/ec_transfer/ec_connector/messages.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wire messages and compatibility hashing for the EC NIXL connector.

Increment EC_CONNECTOR_VERSION whenever the wire schema or NIXL protocol
changes in a way that breaks peer interoperability.
"""

import hashlib
from typing import Annotated

import msgspec

EC_CONNECTOR_VERSION: int = 1


class XferReq(msgspec.Struct, tag="req"):
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


class XferAck(msgspec.Struct, tag="ack"):
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

    Producer computes the same hash locally; XferReq.compatibility_hash must
    match, otherwise the producer returns XferAck(ok=False).
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
        h.update(b"\x00")                          # separator so ab|c != a|bc
    return h.hexdigest()


__all__ = [
    "EC_CONNECTOR_VERSION",
    "XferReq",
    "XferAck",
    "compute_ec_compatibility_hash",
]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/v1/ec_connector/unit/test_ec_messages.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Lint + commit**

```bash
pre-commit run --files \
  vllm/distributed/ec_transfer/ec_connector/messages.py \
  tests/v1/ec_connector/unit/test_ec_messages.py
git add vllm/distributed/ec_transfer/ec_connector/messages.py \
  tests/v1/ec_connector/unit/test_ec_messages.py
git commit -m "$(cat <<'EOF'
Add EC connector wire messages and compatibility hash

XferReq/XferAck are the sole messages on the scheduler↔scheduler ZMQ
side channel.  Compatibility hash guards against silently talking to a
peer built from a drifted codebase.

Co-authored-by: Claude
EOF
)"
```

---

## Task 5: Add `FakeNixlWrapper` for tests

**Files:**

- Create: `tests/v1/ec_connector/_fakes.py`
- Create: `tests/v1/ec_connector/__init__.py` (empty, if not already present)
- Create: `tests/v1/ec_connector/unit/__init__.py` (empty, if not already present)

- [ ] **Step 1: Ensure test package `__init__.py` files exist**

```bash
ls tests/v1/ec_connector/__init__.py tests/v1/ec_connector/unit/__init__.py 2>/dev/null || true
touch tests/v1/ec_connector/__init__.py tests/v1/ec_connector/unit/__init__.py
```

(Safe no-op if they already exist.)

- [ ] **Step 2: Write the fake**

Create `tests/v1/ec_connector/_fakes.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-process fake of the subset of NixlWrapper the EC connector uses.

WRITE transfers synchronously memcpy bytes between registered regions
so end-to-end tests can assert real data arrival without a real NIXL
runtime.
"""

from __future__ import annotations

import itertools
import uuid
from dataclasses import dataclass, field


# -- Global registry so producer fake and consumer fake can "see" each other.
_AGENTS: dict[str, "FakeNixlWrapper"] = {}


@dataclass
class _RegEntry:
    base_addr: int
    size: int


@dataclass
class _XferDescList:
    entries: list[tuple[int, int, int]]   # (addr, size, device_id)


@dataclass
class _Dlist:
    agent_name: str                       # owner of the memory
    descs: _XferDescList
    handle: int


@dataclass
class _Xfer:
    op: str
    local_dlist: _Dlist
    local_ids: list[int]
    remote_dlist: _Dlist
    remote_ids: list[int]
    notif_msg: bytes
    state: str = "PROC"
    error: str | None = None


class FakeNixlWrapper:
    def __init__(self, name: str, config: object | None = None) -> None:
        self.name = name
        self._config = config
        self._registrations: list[_RegEntry] = []
        self._remote_agents: dict[str, FakeNixlWrapper] = {}
        self._dlists: dict[int, _Dlist] = {}
        self._xfers: dict[int, _Xfer] = {}
        self._dlist_counter = itertools.count(1)
        self._xfer_counter = itertools.count(1)
        _AGENTS[name] = self

    # --- agent lifecycle -------------------------------------------------

    def get_agent_metadata(self) -> bytes:
        # Encode the agent name so add_remote_agent can look it up.
        return self.name.encode("utf-8")

    def add_remote_agent(self, metadata: bytes) -> str:
        remote_name = metadata.decode("utf-8")
        self._remote_agents[remote_name] = _AGENTS[remote_name]
        return remote_name

    def remove_remote_agent(self, name: str) -> None:
        self._remote_agents.pop(name, None)

    # --- memory registration --------------------------------------------

    def get_reg_descs(self, data, mem_type):   # noqa: ARG002 — mem_type unused in fake
        return list(data)                        # opaque pass-through

    def register_memory(self, descs, backends=None):   # noqa: ARG002
        for addr, size, _dev, _lbl in descs:
            self._registrations.append(_RegEntry(base_addr=addr, size=size))

    def deregister_memory(self, descs) -> None:
        for addr, size, _dev, _lbl in descs:
            self._registrations = [
                r for r in self._registrations
                if not (r.base_addr == addr and r.size == size)
            ]

    # --- xfer descriptors + dlists --------------------------------------

    def get_xfer_descs(self, blocks_data, mem_type):   # noqa: ARG002
        return _XferDescList(entries=list(blocks_data))

    def prep_xfer_dlist(self, agent_name: str, descs) -> int:
        handle = next(self._dlist_counter)
        # "NIXL_INIT_AGENT" maps to the local agent in the real NIXL API.
        owner = self.name if agent_name == "NIXL_INIT_AGENT" else agent_name
        self._dlists[handle] = _Dlist(agent_name=owner, descs=descs, handle=handle)
        return handle

    def release_dlist_handle(self, handle: int) -> None:
        self._dlists.pop(handle, None)

    # --- xfer execution --------------------------------------------------

    def make_prepped_xfer(
        self,
        op: str,
        local_dlist: int,
        local_ids: list[int],
        remote_dlist: int,
        remote_ids: list[int],
        notif_msg: bytes = b"",
    ) -> int:
        xfer_handle = next(self._xfer_counter)
        self._xfers[xfer_handle] = _Xfer(
            op=op,
            local_dlist=self._dlists[local_dlist],
            local_ids=list(local_ids),
            remote_dlist=self._dlists[remote_dlist],
            remote_ids=list(remote_ids),
            notif_msg=notif_msg,
        )
        return xfer_handle

    def transfer(self, xfer_handle: int) -> None:
        xfer = self._xfers[xfer_handle]
        try:
            # WRITE: local (source) → remote (dest).  We do a raw memcpy
            # via ctypes so tests can observe bytes landing in the dest.
            import ctypes

            assert len(xfer.local_ids) == len(xfer.remote_ids), "id length mismatch"

            src_entries = xfer.local_dlist.descs.entries
            dst_entries = xfer.remote_dlist.descs.entries

            for src_i, dst_i in zip(xfer.local_ids, xfer.remote_ids, strict=True):
                src_addr, src_size, _ = src_entries[src_i]
                dst_addr, dst_size, _ = dst_entries[dst_i]
                assert src_size == dst_size, f"size mismatch {src_size} vs {dst_size}"
                ctypes.memmove(dst_addr, src_addr, src_size)

            xfer.state = "DONE"
        except Exception as e:       # noqa: BLE001 — mirror real NIXL error surface
            xfer.state = "ERR"
            xfer.error = str(e)

    def check_xfer_state(self, xfer_handle: int) -> str:
        return self._xfers[xfer_handle].state

    def release_xfer_handle(self, xfer_handle: int) -> None:
        self._xfers.pop(xfer_handle, None)


def reset_agents() -> None:
    """Clear the cross-agent registry between tests."""
    _AGENTS.clear()
```

- [ ] **Step 3: Smoke-test the fake**

```bash
.venv/bin/python -c "
import ctypes
from tests.v1.ec_connector._fakes import FakeNixlWrapper, reset_agents
reset_agents()
src = (ctypes.c_ubyte * 16)(*range(16))
dst = (ctypes.c_ubyte * 16)()
p = FakeNixlWrapper('P'); c = FakeNixlWrapper('C')
p_reg = p.get_reg_descs([(ctypes.addressof(src), 16, 0, '')], 'DRAM'); p.register_memory(p_reg)
c_reg = c.get_reg_descs([(ctypes.addressof(dst), 16, 0, '')], 'DRAM'); c.register_memory(c_reg)
c_name = p.add_remote_agent(c.get_agent_metadata())
p_descs = p.get_xfer_descs([(ctypes.addressof(src), 16, 0)], 'DRAM')
c_descs = c.get_xfer_descs([(ctypes.addressof(dst), 16, 0)], 'DRAM')
ld = p.prep_xfer_dlist('NIXL_INIT_AGENT', p_descs)
rd = p.prep_xfer_dlist(c_name, c_descs)
h = p.make_prepped_xfer('WRITE', ld, [0], rd, [0]); p.transfer(h)
print(p.check_xfer_state(h))
print(list(dst))
"
```

Expected output: `DONE` followed by `[0, 1, 2, ..., 15]`.

- [ ] **Step 4: Lint + commit**

```bash
pre-commit run --files tests/v1/ec_connector/_fakes.py
git add tests/v1/ec_connector/__init__.py tests/v1/ec_connector/unit/__init__.py \
  tests/v1/ec_connector/_fakes.py
git commit -m "$(cat <<'EOF'
Add FakeNixlWrapper for EC connector tests

Implements the NixlWrapper subset the connector uses (register_memory,
add_remote_agent, prep_xfer_dlist, make_prepped_xfer, transfer,
check_xfer_state, ...).  WRITE xfers synchronously memmove bytes
between registered regions via ctypes, so tests can assert real data
arrival without a real NIXL runtime.

Co-authored-by: Claude
EOF
)"
```

---

## Task 6: Consumer scheduler `__init__` wiring — NIXL agent, mmap registration, state

**Files:**

- Modify: `vllm/distributed/ec_transfer/ec_connector/cpu_connector.py`
- Create: `tests/v1/ec_connector/unit/test_ec_consumer_scheduler.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/v1/ec_connector/unit/test_ec_consumer_scheduler.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the consumer (is_consumer=True) side of ECCPUConnector."""

import uuid
from unittest.mock import patch

import pytest
import torch

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
    engine_id = ""              # filled per test
    ec_role = "ec_consumer"
    ec_connector_extra_config: dict = {}

    @property
    def is_ec_producer(self) -> bool: return self.ec_role == "ec_producer" or self.ec_role == "ec_both"
    @property
    def is_ec_consumer(self) -> bool: return self.ec_role == "ec_consumer" or self.ec_role == "ec_both"

    def get_from_extra_config(self, key, default):
        return self.ec_connector_extra_config.get(key, default)


class _VllmConfig:
    def __init__(self, engine_id: str, extra: dict | None = None, role: str = "ec_consumer"):
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
    return _VllmConfig(engine_id=f"test-{uuid.uuid4()}",
                       extra={"default_encoder_node": "127.0.0.1:65000",
                              "num_ec_blocks": 4})


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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/v1/ec_connector/unit/test_ec_consumer_scheduler.py -v
```

Expected: FAIL on `AttributeError` for `_region` / `_nixl` / `_encoding_map` — none of the scheduler-side init exists yet.

- [ ] **Step 3: Implement consumer scheduler `__init__`**

In `vllm/distributed/ec_transfer/ec_connector/cpu_connector.py`, update imports at the top:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pickle
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import zmq

from vllm import envs
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
```

Replace the existing `__init__` with:

```python
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
        self._nixl: NixlWrapper | None = None

        self._compat_hash = compute_ec_compatibility_hash(
            vllm_version=_vllm_version(),
            model=vllm_config.model_config.model,
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
        self._nixl = NixlWrapper(str(uuid.uuid4()), nixl_agent_config(
            num_threads=1, capture_telemetry=False))
        mem_type = current_platform.get_nixl_memory_type()
        reg_tuples = [
            (self._region.base_ptr, self._region.total_size_bytes, 0, "")
        ]
        self._reg_descs = self._nixl.get_reg_descs(reg_tuples, mem_type)
        self._nixl.register_memory(self._reg_descs)
        self._mem_type = mem_type

        # xfer_descs over every block slot, for producer peers to address.
        xfer_tuples = [
            (self._region.base_ptr + i * self._block_size_bytes,
             self._block_size_bytes, 0)
            for i in range(num_ec_blocks)
        ]
        self._xfer_descs = self._nixl.get_xfer_descs(xfer_tuples, mem_type)

        self._agent_metadata: bytes = self._nixl.get_agent_metadata()
        # Serialize the descs object — pickle is safe here because the
        # consumer emits bytes that only its peer producers (same trusted
        # deployment) ever decode.  Producer rebuilds via get_xfer_descs
        # over the same (base_ptr, block_size_bytes, num_blocks) tuple we
        # encode here; no arbitrary-object exec path on the wire.
        self._mem_descriptor_bytes: bytes = pickle.dumps(
            xfer_tuples, protocol=pickle.HIGHEST_PROTOCOL
        )

        # Scheduler state.
        self._encoding_map: dict[str, list[int]] = {}
        self._ready: set[str] = set()
        self._dealers: dict[tuple[str, int], zmq.Socket] = {}
        self._zmq_ctx = zmq.Context.instance()
        self._nixl_agent_name = self._nixl.name

    def _init_producer(
        self,
        vllm_config: "VllmConfig",
        ec_config,
        num_ec_blocks: int,
    ) -> None:
        # Producer-side init — scaffolded here; listener thread added in Task 7.
        self._encodings: dict[str, torch.Tensor] = {}
        self._waiting: dict[str, tuple[bytes, XferReq]] = {}
        self._nixl = NixlWrapper(str(uuid.uuid4()), nixl_agent_config(
            num_threads=1, capture_telemetry=False))
        self._mem_type = current_platform.get_nixl_memory_type()
        self._zmq_ctx = zmq.Context.instance()
        self._router: zmq.Socket | None = None
        self._listener_thread = None
        self._stop_event = None
```

Also add a tiny helper at module top (after `logger = init_logger(__name__)`):

```python
def _vllm_version() -> str:
    from vllm import __version__
    return __version__
```

Also update `shutdown` on the connector (add at bottom of class):

```python
    def shutdown(self) -> None:
        if self.role != ECConnectorRole.SCHEDULER:
            return
        if self.is_producer and getattr(self, "_stop_event", None) is not None:
            self._stop_event.set()
            if self._listener_thread is not None:
                self._listener_thread.join(timeout=5)
            if self._router is not None:
                self._router.close(linger=0)
        if self.is_consumer:
            for sock in getattr(self, "_dealers", {}).values():
                sock.close(linger=0)
        if self._nixl is not None:
            try:
                self._nixl.deregister_memory(self._reg_descs)
            except AttributeError:
                pass                               # producer has no _reg_descs
        if self._region is not None:
            self._region.cleanup()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/v1/ec_connector/unit/test_ec_consumer_scheduler.py -v
```

Expected: both tests PASS.

- [ ] **Step 5: Verify existing worker tests still pass**

```bash
.venv/bin/python -m pytest tests/v1/ec_connector/unit/test_ec_cpu_connector.py -v
```

Expected: all existing tests PASS (the worker-role init was refactored into `_init_worker` but behaves identically).

- [ ] **Step 6: Lint + commit**

```bash
pre-commit run --files \
  vllm/distributed/ec_transfer/ec_connector/cpu_connector.py \
  tests/v1/ec_connector/unit/test_ec_consumer_scheduler.py
git add vllm/distributed/ec_transfer/ec_connector/cpu_connector.py \
  tests/v1/ec_connector/unit/test_ec_consumer_scheduler.py
git commit -m "$(cat <<'EOF'
Wire ECCPUConnector consumer scheduler init

Opens the same mmap workers use, creates a NIXL agent, registers the
region, and builds an xfer_descs handle + serialized peer descriptor
bytes for inclusion in XferReq messages.  State dicts (_encoding_map,
_ready, _dealers) are initialized empty.  Producer scaffolding is
added but the listener thread arrives in the next task.

Co-authored-by: Claude
EOF
)"
```

---

## Task 7: Producer scheduler — ROUTER listener thread

**Files:**

- Modify: `vllm/distributed/ec_transfer/ec_connector/cpu_connector.py`
- Create: `tests/v1/ec_connector/unit/test_ec_producer_scheduler.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/v1/ec_connector/unit/test_ec_producer_scheduler.py`:

```python
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
    XferAck,
    XferReq,
    compute_ec_compatibility_hash,
)


class _MMConfig:
    def get_inputs_embeds_size(self) -> int: return 32
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
    def is_ec_producer(self) -> bool: return self.ec_role in ("ec_producer", "ec_both")
    @property
    def is_ec_consumer(self) -> bool: return self.ec_role in ("ec_consumer", "ec_both")
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
    cfg = _VllmConfig(engine_id=f"prod-{uuid.uuid4()}", role="ec_producer",
                      extra={"num_ec_blocks": 4})
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
            # Build a client DEALER and send an XferReq for a hash the producer hasn't got.
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/v1/ec_connector/unit/test_ec_producer_scheduler.py -v
```

Expected: FAIL — listener thread isn't started yet.

- [ ] **Step 3: Implement the listener thread on producer**

In `cpu_connector.py`, add at the top:

```python
import threading
```

Replace the body of `_init_producer` with:

```python
    def _init_producer(
        self,
        vllm_config: "VllmConfig",
        ec_config,
        num_ec_blocks: int,
    ) -> None:
        self._encodings: dict[str, torch.Tensor] = {}
        self._waiting: dict[str, tuple[bytes, XferReq]] = {}
        self._nixl = NixlWrapper(str(uuid.uuid4()), nixl_agent_config(
            num_threads=1, capture_telemetry=False))
        self._mem_type = current_platform.get_nixl_memory_type()
        self._device_id = 0

        self._zmq_ctx = zmq.Context.instance()
        host = envs.VLLM_EC_SIDE_CHANNEL_HOST
        port = envs.VLLM_EC_SIDE_CHANNEL_PORT
        path = make_zmq_path("tcp", host, port)
        self._router = make_zmq_socket(
            self._zmq_ctx, path, zmq.ROUTER, bind=True
        )
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
                identity, _, payload = self._router.recv_multipart()
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
        try:
            self._router.send_multipart([identity, b"", payload])
        except zmq.ContextTerminated:
            pass

    def _do_nixl_xfer(self, req: XferReq, tensor: torch.Tensor) -> bool:
        # Stub: implemented in Task 8.  Returns False until then.
        _ = (req, tensor)
        return False
```

Also add `import msgspec` at module top if not already present.

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/v1/ec_connector/unit/test_ec_producer_scheduler.py -v
```

Expected: both tests PASS.

- [ ] **Step 5: Lint + commit**

```bash
pre-commit run --files \
  vllm/distributed/ec_transfer/ec_connector/cpu_connector.py \
  tests/v1/ec_connector/unit/test_ec_producer_scheduler.py
git add vllm/distributed/ec_transfer/ec_connector/cpu_connector.py \
  tests/v1/ec_connector/unit/test_ec_producer_scheduler.py
git commit -m "$(cat <<'EOF'
Start producer ROUTER listener thread for EC NIXL transfer

Listener decodes XferReq messages, validates the compatibility hash,
and either dispatches (when the encoding is locally available) or
parks in _waiting for the future save_caches path to drain.  NIXL
transfer itself is stubbed; implemented in the next task.

Co-authored-by: Claude
EOF
)"
```

---

## Task 8: Producer — implement `_do_nixl_xfer`

**Files:**

- Modify: `vllm/distributed/ec_transfer/ec_connector/cpu_connector.py`
- Modify: `tests/v1/ec_connector/unit/test_ec_producer_scheduler.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/v1/ec_connector/unit/test_ec_producer_scheduler.py`:

```python
def test_producer_dispatches_when_encoding_present(producer_config):
    import ctypes
    cfg, port = producer_config
    with patch(
        "vllm.distributed.ec_transfer.ec_connector.cpu_connector.NixlWrapper",
        FakeNixlWrapper,
    ):
        conn = ECCPUConnector(cfg, ECConnectorRole.SCHEDULER)
        try:
            # Plant an encoding tensor whose bytes we can verify in the fake's dst.
            n_blocks = 3
            block_size = conn._block_size_bytes
            tensor = torch.arange(n_blocks * block_size, dtype=torch.uint8).contiguous()
            conn._encodings["have-it"] = tensor

            # Create a "consumer" FakeNixlWrapper and a destination buffer
            # the producer will WRITE into.
            consumer = FakeNixlWrapper("consumer-agent")
            dst_buf = (ctypes.c_ubyte * (n_blocks * block_size))()
            dst_addr = ctypes.addressof(dst_buf)
            dst_tuples = [
                (dst_addr + i * block_size, block_size, 0)
                for i in range(n_blocks)
            ]
            import pickle
            consumer_mem_descriptor = pickle.dumps(dst_tuples)

            ctx = zmq.Context.instance()
            dealer = ctx.socket(zmq.DEALER)
            dealer.connect(f"tcp://127.0.0.1:{port}")

            req = XferReq(
                mm_hash="have-it",
                dst_block_indices=list(range(n_blocks)),
                consumer_agent_name=consumer.name,
                consumer_nixl_metadata=consumer.get_agent_metadata(),
                consumer_mem_descriptor=consumer_mem_descriptor,
                compatibility_hash=conn._compat_hash,
            )
            dealer.send(msgspec.msgpack.encode(req))

            dealer.setsockopt(zmq.RCVTIMEO, 5000)
            ack_bytes = dealer.recv()
            ack = msgspec.msgpack.decode(ack_bytes, type=XferAck)
            assert ack.mm_hash == "have-it"
            assert ack.ok is True

            # Destination now holds the source tensor bytes.
            assert list(dst_buf) == list(tensor.tolist())
            dealer.close(linger=0)
        finally:
            conn.shutdown()
```

- [ ] **Step 2: Run test to see it fail**

```bash
.venv/bin/python -m pytest tests/v1/ec_connector/unit/test_ec_producer_scheduler.py::test_producer_dispatches_when_encoding_present -v
```

Expected: FAIL — `ok` is `False` because `_do_nixl_xfer` is still stubbed.

- [ ] **Step 3: Implement `_do_nixl_xfer`**

In `cpu_connector.py`, replace the stubbed `_do_nixl_xfer` with:

```python
    def _do_nixl_xfer(self, req: XferReq, tensor: torch.Tensor) -> bool:
        assert self._nixl is not None
        n = len(req.dst_block_indices)
        expected = n * self._block_size_bytes
        if tensor.nbytes != expected:
            logger.warning(
                "ec: size mismatch for mm_hash=%s: tensor=%d expected=%d",
                req.mm_hash, tensor.nbytes, expected,
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
            remote_agent_name = self._nixl.add_remote_agent(
                req.consumer_nixl_metadata
            )

            local_tuples = [
                (tensor.data_ptr() + i * self._block_size_bytes,
                 self._block_size_bytes, self._device_id)
                for i in range(n)
            ]
            local_descs = self._nixl.get_xfer_descs(local_tuples, self._mem_type)
            local_dlist = self._nixl.prep_xfer_dlist("NIXL_INIT_AGENT", local_descs)

            remote_tuples = pickle.loads(req.consumer_mem_descriptor)
            # Rebuild xfer_descs on our side from the raw tuples the consumer
            # advertised.  We only trust the tuple schema (addr,size,dev_id)
            # — ``pickle.loads`` here deserializes a list[tuple[int,int,int]]
            # produced exclusively by consumers in this connector.
            remote_descs = self._nixl.get_xfer_descs(remote_tuples, self._mem_type)
            remote_dlist = self._nixl.prep_xfer_dlist(
                remote_agent_name, remote_descs
            )

            xfer_handle = self._nixl.make_prepped_xfer(
                "WRITE",
                local_dlist, list(range(n)),
                remote_dlist, req.dst_block_indices,
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
                        state, req.mm_hash,
                    )
                    return False
                time.sleep(0.0005)
        except Exception as e:
            logger.warning(
                "ec: NIXL xfer failed for mm_hash=%s: %s", req.mm_hash, e
            )
            return False
        finally:
            if xfer_handle is not None:
                try:
                    self._nixl.release_xfer_handle(xfer_handle)
                except Exception:
                    logger.debug("ec: release_xfer_handle raised", exc_info=True)
            if local_dlist is not None:
                try:
                    self._nixl.release_dlist_handle(local_dlist)
                except Exception:
                    logger.debug("ec: release_dlist_handle(local) raised",
                                 exc_info=True)
            if remote_dlist is not None:
                try:
                    self._nixl.release_dlist_handle(remote_dlist)
                except Exception:
                    logger.debug("ec: release_dlist_handle(remote) raised",
                                 exc_info=True)
            if remote_agent_name is not None:
                try:
                    self._nixl.remove_remote_agent(remote_agent_name)
                except Exception:
                    logger.debug("ec: remove_remote_agent raised", exc_info=True)
            try:
                self._nixl.deregister_memory(reg_descs)
            except Exception:
                logger.debug("ec: deregister_memory raised", exc_info=True)
```

Also add `import time` at module top.

- [ ] **Step 4: Run test to see it pass**

```bash
.venv/bin/python -m pytest tests/v1/ec_connector/unit/test_ec_producer_scheduler.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Lint + commit**

```bash
pre-commit run --files vllm/distributed/ec_transfer/ec_connector/cpu_connector.py \
  tests/v1/ec_connector/unit/test_ec_producer_scheduler.py
git add vllm/distributed/ec_transfer/ec_connector/cpu_connector.py \
  tests/v1/ec_connector/unit/test_ec_producer_scheduler.py
git commit -m "$(cat <<'EOF'
Implement _do_nixl_xfer on EC producer

Per-transfer lifecycle: register source tensor, add remote agent, prep
local/remote dlists, make_prepped_xfer + transfer, poll for DONE,
release everything on return.  Consumer's memory descriptor arrives as
a pickled list[tuple[int,int,int]] and is rebuilt on our side via
get_xfer_descs — we never deserialize an opaque NIXL object.

Co-authored-by: Claude
EOF
)"
```

---

## Task 9: Consumer — implement `has_cache_item`, `_drain_acks`, `_complete`

**Files:**

- Modify: `vllm/distributed/ec_transfer/ec_connector/cpu_connector.py`
- Modify: `tests/v1/ec_connector/unit/test_ec_consumer_scheduler.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/v1/ec_connector/unit/test_ec_consumer_scheduler.py`:

```python
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
            # Manually mark an entry as pending.
            conn._encoding_map["h"] = [0, 1, 2]
            assert conn.has_cache_item("h") is None
            # Simulate ack arrival by calling _complete directly.
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
            # Blocks returned to the region.
            assert conn._region.alloc(3) == indices
        finally:
            conn.shutdown()
```

- [ ] **Step 2: Run tests to see them fail**

```bash
.venv/bin/python -m pytest tests/v1/ec_connector/unit/test_ec_consumer_scheduler.py -v -k "cache_item or _complete"
```

Expected: FAIL on `AttributeError` / `NotImplementedError` on `has_cache_item`, `_complete`.

- [ ] **Step 3: Implement the three methods**

In `cpu_connector.py`, replace the `has_cache_item` stub with:

```python
    def has_cache_item(self, identifier: str) -> bool | None:
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
            return                                 # stale / duplicate ack
        if ok:
            self._ready.add(mm_hash)
        else:
            indices = self._encoding_map.pop(mm_hash)
            self._region.free(indices)
```

- [ ] **Step 4: Run tests to see them pass**

```bash
.venv/bin/python -m pytest tests/v1/ec_connector/unit/test_ec_consumer_scheduler.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Lint + commit**

```bash
pre-commit run --files vllm/distributed/ec_transfer/ec_connector/cpu_connector.py \
  tests/v1/ec_connector/unit/test_ec_consumer_scheduler.py
git add vllm/distributed/ec_transfer/ec_connector/cpu_connector.py \
  tests/v1/ec_connector/unit/test_ec_consumer_scheduler.py
git commit -m "$(cat <<'EOF'
Implement EC consumer has_cache_item + ack drain + completion

has_cache_item is the engine's readiness probe: True if the xfer
completed, None if still in flight, False if we've never heard of the
hash.  _drain_acks polls every DEALER non-blocking and routes each
ack to _complete.  Success adds to _ready; failure frees the blocks.

Co-authored-by: Claude
EOF
)"
```

---

## Task 10: Consumer — implement `update_state_after_alloc` and dealer management

**Files:**

- Modify: `vllm/distributed/ec_transfer/ec_connector/cpu_connector.py`
- Modify: `tests/v1/ec_connector/unit/test_ec_consumer_scheduler.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/v1/ec_connector/unit/test_ec_consumer_scheduler.py`:

```python
class _FakeMMFeature:
    def __init__(self, mm_hash: str, length: int):
        self.mm_hash = mm_hash
        self.identifier = mm_hash
        class _Pos: ...
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
            ctx = zmq.Context.instance()
            router = ctx.socket(zmq.ROUTER)
            router.bind("tcp://127.0.0.1:65000")
            router.setsockopt(zmq.RCVTIMEO, 2000)

            req = _FakeRequest([_FakeMMFeature("hash-1", length=3)])
            conn.update_state_after_alloc(req, 0)

            identity, _, payload = router.recv_multipart()
            decoded = msgspec.msgpack.decode(payload, type=XferReq)
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
                "default_encoder_node"] = "127.0.0.1:65001"

            feat = _FakeMMFeature("dup", length=2)
            req = _FakeRequest([feat])
            conn.update_state_after_alloc(req, 0)
            conn.update_state_after_alloc(req, 0)                # second call: no-op

            identity, _, payload = router.recv_multipart()
            with pytest.raises(zmq.Again):
                router.recv_multipart()                          # only one XferReq
            router.close(linger=0)
        finally:
            conn.shutdown()
```

Also update the `vllm_config` fixture above to point `default_encoder_node` at port 65000 (already done in Task 6's fixture).

- [ ] **Step 2: Run to see them fail**

```bash
.venv/bin/python -m pytest tests/v1/ec_connector/unit/test_ec_consumer_scheduler.py -v -k "update_state"
```

Expected: FAIL — `update_state_after_alloc` still raises `NotImplementedError`.

- [ ] **Step 3: Implement**

In `cpu_connector.py`, replace `update_state_after_alloc` and add helpers:

```python
    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        if not self.is_consumer:
            return
        feature = request.mm_features[index]
        mm_hash = feature.mm_hash or feature.identifier
        if mm_hash in self._encoding_map:
            return

        n = feature.mm_position.length
        indices = self._region.alloc(n)
        self._encoding_map[mm_hash] = indices

        addr = self._resolve_encoder_address(request, feature)
        dealer = self._get_or_create_dealer(addr)
        xfer = XferReq(
            mm_hash=mm_hash,
            dst_block_indices=indices,
            consumer_agent_name=self._nixl_agent_name,
            consumer_nixl_metadata=self._agent_metadata,
            consumer_mem_descriptor=self._mem_descriptor_bytes,
            compatibility_hash=self._compat_hash,
        )
        dealer.send(msgspec.msgpack.encode(xfer))

    def _resolve_encoder_address(
        self, request: "Request", feature
    ) -> tuple[str, int]:
        # Preferred path (future PR): the request carries it in kv_transfer_params.
        params = getattr(request, "kv_transfer_params", None) or {}
        raw = params.get("encoder_node_address")
        if raw is None:
            raw = self._vllm_config.ec_transfer_config.get_from_extra_config(
                "default_encoder_node", None
            )
        if raw is None:
            raise RuntimeError(
                "EC consumer has no encoder node address: set "
                "ec_connector_extra_config.default_encoder_node or populate "
                "kv_transfer_params.encoder_node_address on the request."
            )
        host, _, port = raw.rpartition(":")
        return host, int(port)

    def _get_or_create_dealer(self, addr: tuple[str, int]) -> zmq.Socket:
        sock = self._dealers.get(addr)
        if sock is not None:
            return sock
        path = make_zmq_path("tcp", addr[0], addr[1])
        sock = make_zmq_socket(self._zmq_ctx, path, zmq.DEALER, bind=False)
        self._dealers[addr] = sock
        return sock
```

- [ ] **Step 4: Run to see them pass**

```bash
.venv/bin/python -m pytest tests/v1/ec_connector/unit/test_ec_consumer_scheduler.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Lint + commit**

```bash
pre-commit run --files vllm/distributed/ec_transfer/ec_connector/cpu_connector.py \
  tests/v1/ec_connector/unit/test_ec_consumer_scheduler.py
git add vllm/distributed/ec_transfer/ec_connector/cpu_connector.py \
  tests/v1/ec_connector/unit/test_ec_consumer_scheduler.py
git commit -m "$(cat <<'EOF'
Implement EC consumer update_state_after_alloc + dealer mgmt

Allocates N blocks from the region, records the mapping in
_encoding_map, sends XferReq to the E node.  Encoder node address is
resolved from kv_transfer_params (future PR) or falls back to
ec_connector_extra_config.default_encoder_node.  DEALER sockets are
cached per (host,port) so we don't churn connections.

Co-authored-by: Claude
EOF
)"
```

---

## Task 11: Consumer — implement `build_connector_meta`

**Files:**

- Modify: `vllm/distributed/ec_transfer/ec_connector/cpu_connector.py`
- Modify: `tests/v1/ec_connector/unit/test_ec_consumer_scheduler.py`

- [ ] **Step 1: Add failing test**

```python
def test_build_connector_meta_hands_off_ready_entries(vllm_config):
    with patch(
        "vllm.distributed.ec_transfer.ec_connector.cpu_connector.NixlWrapper",
        FakeNixlWrapper,
    ):
        conn = ECCPUConnector(vllm_config, ECConnectorRole.SCHEDULER)
        try:
            conn._encoding_map["a"] = [0, 1]
            conn._encoding_map["b"] = [2, 3]
            conn._ready = {"a"}                  # only 'a' is ready

            class _SO: pass
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
```

- [ ] **Step 2: Run to see it fail**

```bash
.venv/bin/python -m pytest tests/v1/ec_connector/unit/test_ec_consumer_scheduler.py::test_build_connector_meta_hands_off_ready_entries -v
```

Expected: FAIL — existing `build_connector_meta` reads `_encoding_map` unconditionally.

- [ ] **Step 3: Replace `build_connector_meta`**

```python
    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> ECCPUConnectorMetadata:
        if self.is_consumer:
            self._drain_acks()
            mapping = {h: self._encoding_map.pop(h) for h in self._ready}
            self._ready.clear()
            return ECCPUConnectorMetadata(mm_hash_to_cpu_blocks=mapping)
        # Producer-only role never has anything to hand to local workers.
        return ECCPUConnectorMetadata(mm_hash_to_cpu_blocks={})
```

- [ ] **Step 4: Run to see it pass**

```bash
.venv/bin/python -m pytest tests/v1/ec_connector/unit/test_ec_consumer_scheduler.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Lint + commit**

```bash
pre-commit run --files vllm/distributed/ec_transfer/ec_connector/cpu_connector.py \
  tests/v1/ec_connector/unit/test_ec_consumer_scheduler.py
git add vllm/distributed/ec_transfer/ec_connector/cpu_connector.py \
  tests/v1/ec_connector/unit/test_ec_consumer_scheduler.py
git commit -m "$(cat <<'EOF'
Implement EC consumer build_connector_meta hand-off

Drains any pending acks, then pops every ready mm_hash from
_encoding_map into the ECCPUConnectorMetadata the worker receives.
Worker's start_load_caches CPU→GPU-copies from the block indices.
Hand-off-and-forget: clears _ready each step.

Co-authored-by: Claude
EOF
)"
```

---

## Task 12: Producer — `_drain_waiting` hook (for future `save_caches`)

**Files:**

- Modify: `vllm/distributed/ec_transfer/ec_connector/cpu_connector.py`
- Modify: `tests/v1/ec_connector/unit/test_ec_producer_scheduler.py`

- [ ] **Step 1: Add failing test**

```python
def test_drain_waiting_fires_parked_request(producer_config):
    import ctypes, pickle
    cfg, port = producer_config
    with patch(
        "vllm.distributed.ec_transfer.ec_connector.cpu_connector.NixlWrapper",
        FakeNixlWrapper,
    ):
        conn = ECCPUConnector(cfg, ECConnectorRole.SCHEDULER)
        try:
            # Client sends XferReq before producer has the encoding.
            consumer = FakeNixlWrapper("c-agent")
            n = 2
            block_size = conn._block_size_bytes
            dst_buf = (ctypes.c_ubyte * (n * block_size))()
            dst_addr = ctypes.addressof(dst_buf)
            dst_tuples = [
                (dst_addr + i * block_size, block_size, 0) for i in range(n)
            ]

            ctx = zmq.Context.instance()
            dealer = ctx.socket(zmq.DEALER)
            dealer.connect(f"tcp://127.0.0.1:{port}")

            req = XferReq(
                mm_hash="later",
                dst_block_indices=list(range(n)),
                consumer_agent_name=consumer.name,
                consumer_nixl_metadata=consumer.get_agent_metadata(),
                consumer_mem_descriptor=pickle.dumps(dst_tuples),
                compatibility_hash=conn._compat_hash,
            )
            dealer.send(msgspec.msgpack.encode(req))

            # Wait for listener to park it.
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and "later" not in conn._waiting:
                time.sleep(0.01)
            assert "later" in conn._waiting

            # Populate encoding and trigger drain.
            tensor = torch.full((n * block_size,), 7, dtype=torch.uint8).contiguous()
            conn._encodings["later"] = tensor
            conn._drain_waiting("later")

            dealer.setsockopt(zmq.RCVTIMEO, 2000)
            ack = msgspec.msgpack.decode(dealer.recv(), type=XferAck)
            assert ack.ok is True
            assert ack.mm_hash == "later"
            assert list(dst_buf) == [7] * (n * block_size)
            assert "later" not in conn._waiting
            dealer.close(linger=0)
        finally:
            conn.shutdown()
```

- [ ] **Step 2: Run to see it fail**

```bash
.venv/bin/python -m pytest tests/v1/ec_connector/unit/test_ec_producer_scheduler.py::test_drain_waiting_fires_parked_request -v
```

Expected: FAIL — `_drain_waiting` doesn't exist.

- [ ] **Step 3: Implement `_drain_waiting`**

Add to `cpu_connector.py` on the connector class:

```python
    def _drain_waiting(self, mm_hash: str) -> None:
        """Triggered by future save_caches after populating _encodings."""
        entry = self._waiting.pop(mm_hash, None)
        if entry is None:
            return
        identity, req = entry
        tensor = self._encodings.get(mm_hash)
        if tensor is None:
            self._send_ack(identity, mm_hash, ok=False)
            return
        ok = self._do_nixl_xfer(req, tensor)
        self._send_ack(identity, mm_hash, ok)
```

- [ ] **Step 4: Run to see it pass**

```bash
.venv/bin/python -m pytest tests/v1/ec_connector/unit/test_ec_producer_scheduler.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Lint + commit**

```bash
pre-commit run --files vllm/distributed/ec_transfer/ec_connector/cpu_connector.py \
  tests/v1/ec_connector/unit/test_ec_producer_scheduler.py
git add vllm/distributed/ec_transfer/ec_connector/cpu_connector.py \
  tests/v1/ec_connector/unit/test_ec_producer_scheduler.py
git commit -m "$(cat <<'EOF'
Add EC producer _drain_waiting hook for late-arriving encodings

Future save_caches populates _encodings[mm_hash] then calls
_drain_waiting(mm_hash) so any XferReqs that arrived early get served.
Tested via a direct call until save_caches lands in a follow-up PR.

Co-authored-by: Claude
EOF
)"
```

---

## Task 13: Integration test — end-to-end over real ZMQ with `FakeNixlWrapper`

**Files:**

- Create: `tests/v1/ec_connector/integration/__init__.py`
- Create: `tests/v1/ec_connector/integration/test_ec_zmq_loopback.py`

- [ ] **Step 1: Ensure package init exists**

```bash
touch tests/v1/ec_connector/integration/__init__.py
```

- [ ] **Step 2: Write the integration test**

Create `tests/v1/ec_connector/integration/test_ec_zmq_loopback.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end test with real ZMQ sockets and FakeNixlWrapper.

Wires one producer ECCPUConnector and one consumer ECCPUConnector
inside a single process, drives an XferReq through the full pipeline,
and asserts the consumer's mmap receives the producer's tensor bytes.
"""

import pickle
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
    def get_inputs_embeds_size(self): return 16
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
    def is_ec_producer(self): return self.ec_role in ("ec_producer", "ec_both")
    @property
    def is_ec_consumer(self): return self.ec_role in ("ec_consumer", "ec_both")
    def get_from_extra_config(self, k, d): return self.ec_connector_extra_config.get(k, d)

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
    s = socket.socket(); s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]; s.close(); return p


class _FakeFeat:
    def __init__(self, h, L):
        self.mm_hash = h; self.identifier = h
        class _Pos: ...
        self.mm_position = _Pos(); self.mm_position.length = L

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
        f"c-{uuid.uuid4()}", "ec_consumer",
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

            assert _poll_until(lambda: cons.has_cache_item("h") is True), \
                "consumer never saw transfer complete"

            class _SO: pass
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
        f"c-{uuid.uuid4()}", "ec_consumer",
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

            assert _poll_until(lambda: "late" in prod._waiting), \
                "producer never parked request"

            # Now populate and drain.
            n, bs = 2, prod._block_size_bytes
            tensor = torch.full((n * bs,), 5, dtype=torch.uint8).contiguous()
            prod._encodings["late"] = tensor
            prod._drain_waiting("late")

            assert _poll_until(lambda: cons.has_cache_item("late") is True)
            class _SO: pass
            meta = cons.build_connector_meta(_SO())
            idx = meta.mm_hash_to_cpu_blocks["late"]
            arrived = cons._region.blocks[idx].flatten()
            assert torch.all(arrived == 5)
        finally:
            cons.shutdown()
            prod.shutdown()
```

- [ ] **Step 3: Run tests**

```bash
.venv/bin/python -m pytest tests/v1/ec_connector/integration/test_ec_zmq_loopback.py -v
```

Expected: both tests PASS.

- [ ] **Step 4: Full test sweep to ensure no regressions**

```bash
.venv/bin/python -m pytest tests/v1/ec_connector/ -v
```

Expected: every test (unit + integration) PASSES.

- [ ] **Step 5: Lint + commit**

```bash
pre-commit run --files \
  tests/v1/ec_connector/integration/__init__.py \
  tests/v1/ec_connector/integration/test_ec_zmq_loopback.py
git add tests/v1/ec_connector/integration/__init__.py \
  tests/v1/ec_connector/integration/test_ec_zmq_loopback.py
git commit -m "$(cat <<'EOF'
Add EC NIXL transfer end-to-end integration tests

Two connectors (producer + consumer) wired over real ZMQ sockets with
FakeNixlWrapper standing in for NIXL's data plane.  Asserts the
consumer's mmap actually receives the producer's tensor bytes, and
that the wait-then-drain path works for requests that arrive before
the encoding is ready.

Co-authored-by: Claude
EOF
)"
```

---

## Self-review

This plan covers:

- **Spec §ECSharedRegion change** → Task 1.
- **Spec §env vars** → Task 2.
- **Spec §base.shutdown** → Task 3.
- **Spec §messages.py schemas + compat hash** → Task 4.
- **Spec §FakeNixlWrapper** → Task 5.
- **Spec §consumer `__init__`** → Task 6.
- **Spec §producer listener** → Tasks 7, 8.
- **Spec §consumer has_cache_item/drain/complete** → Task 9.
- **Spec §consumer update_state_after_alloc / encoder address resolution / dealer lifecycle** → Task 10.
- **Spec §consumer build_connector_meta** → Task 11.
- **Spec §producer _drain_waiting** → Task 12.
- **Spec §integration/loopback tests** → Task 13.

Method/property names verified consistent across tasks: `_region`, `_nixl`, `_encoding_map`, `_ready`, `_dealers`, `_waiting`, `_encodings`, `_router`, `_listener_thread`, `_stop_event`, `_compat_hash`, `_agent_metadata`, `_mem_descriptor_bytes`, `_block_size_bytes`, `_mem_type`, `_device_id`, `_resolve_encoder_address`, `_get_or_create_dealer`, `_send_ack`, `_do_nixl_xfer`, `_drain_acks`, `_drain_waiting`, `_complete`.

No placeholder strings, all commands explicit, every code step is shown in full.

**Out of plan but noted in spec's Follow-up work (no task here — deliberate):**

- Producer worker `save_caches` (future PR).
- Encoder node address via `kv_transfer_params` (future PR).
- Block eviction / free-on-request-finished.
- Real NIXL smoke test (optional; guarded by HAS_NIXL).
- Ack-timeout sweeper.
- Producer pooled-registration optimization.
