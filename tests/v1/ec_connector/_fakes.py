# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-process fake of the NixlWrapper subset the EC connector uses.

WRITE transfers synchronously memcpy bytes between registered regions
so end-to-end tests can assert real data arrival without a real NIXL
runtime.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

# Global registry so producer fake and consumer fake can "see" each other.
_AGENTS: dict[str, FakeNixlWrapper] = {}


@dataclass
class _RegEntry:
    base_addr: int
    size: int


@dataclass
class _XferDescList:
    entries: list[tuple[int, int, int]]  # (addr, size, device_id)


@dataclass
class _Dlist:
    agent_name: str  # owner of the memory
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

    def get_reg_descs(self, data, mem_type):  # noqa: ARG002 — mem_type unused in fake
        return list(data)  # opaque pass-through

    def register_memory(self, descs, backends=None):  # noqa: ARG002
        for addr, size, _dev, _lbl in descs:
            self._registrations.append(_RegEntry(base_addr=addr, size=size))

    def deregister_memory(self, descs) -> None:
        for addr, size, _dev, _lbl in descs:
            self._registrations = [
                r
                for r in self._registrations
                if not (r.base_addr == addr and r.size == size)
            ]

    # --- xfer descriptors + dlists --------------------------------------

    def get_xfer_descs(self, blocks_data, mem_type):  # noqa: ARG002
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
        except Exception as e:  # noqa: BLE001 — mirror real NIXL error surface
            xfer.state = "ERR"
            xfer.error = str(e)

    def check_xfer_state(self, xfer_handle: int) -> str:
        return self._xfers[xfer_handle].state

    def release_xfer_handle(self, xfer_handle: int) -> None:
        self._xfers.pop(xfer_handle, None)


def reset_agents() -> None:
    """Clear the cross-agent registry between tests."""
    _AGENTS.clear()
