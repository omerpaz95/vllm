# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prove a NIXL READ between two pods crosses RDMA, the way ECCPUConnector
issues it.

The connector registers a host (DRAM) region with NIXL's UCX backend and the
consumer READs the producer's blocks into its own, then sends the producer a
notification. This script does exactly that with one large region per side:

    python rdma_probe.py serve --port 5600            # on the producer pod
    python rdma_probe.py read --peer <producer-ip> --port 5600   # consumer pod

Metadata travels over a plain TCP socket (the connector uses ZMQ for the
same thing); only the bulk READ is NIXL's. The reader prints one JSON line
with the measured bandwidth and whether the bytes arrived intact; with
UCX_PROTO_INFO=y, UCX also prints on stderr the protocol and transport each
operation selected (e.g. `rc_mlx5/mlx5_0:1`, or `tcp/eth0`).

Run it with the same UCX_TLS on both sides, once restricted to RDMA
transports (`rc,cuda_copy,cuda_ipc`) and once to tcp (`tcp,cuda_copy`): NIXL
rejects a UCX_TLS with no CUDA transport in a CUDA image, GPU or not. A
real RDMA path names an mlx5 device in that output, still succeeds with tcp
excluded, and is several times faster than the tcp control.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import mmap
import os
import socket
import struct
import subprocess
import sys
import time

_NIXL_DRAM = "DRAM"


def _send(sock: socket.socket, payload: bytes) -> None:
    sock.sendall(struct.pack("!Q", len(payload)) + payload)


def _recv(sock: socket.socket) -> bytes:
    (size,) = struct.unpack("!Q", _recv_exact(sock, 8))
    return _recv_exact(sock, size)


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    buf = bytearray()
    while len(buf) < size:
        chunk = sock.recv(min(size - len(buf), 1 << 20))
        if not chunk:
            raise ConnectionError("peer closed the metadata socket")
        buf.extend(chunk)
    return bytes(buf)


def _region(size: int, fill: int | None) -> tuple[mmap.mmap, int]:
    """An anonymous host buffer and its address, like the connector's mmap."""
    buf = mmap.mmap(-1, size)
    if fill is not None:
        pattern = bytes((fill + i) % 251 for i in range(1 << 20))
        for off in range(0, size, len(pattern)):
            buf[off : off + len(pattern)] = pattern[: size - off]
    addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
    return buf, addr


def _agent(name: str):
    from nixl._api import nixl_agent, nixl_agent_config

    return nixl_agent(name, nixl_agent_config(backends=["UCX"]))


def _ucx_info() -> str:
    """The transports UCX can see in this pod, for the report."""
    try:
        out = subprocess.run(
            ["ucx_info", "-d"], capture_output=True, text=True, timeout=30
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return "ucx_info unavailable"
    lines = [ln.strip() for ln in out.splitlines() if "Transport:" in ln]
    return ", ".join(sorted({ln.split(":", 1)[1].strip() for ln in lines}))


def serve(args: argparse.Namespace) -> int:
    size = args.size_mib << 20
    buf, addr = _region(size, fill=7)
    agent = _agent("probe-producer")
    agent.register_memory(agent.get_reg_descs([(addr, size, 0, "")], _NIXL_DRAM))
    with socket.create_server(("0.0.0.0", args.port)) as server:
        print(f"[probe] serving {args.size_mib} MiB on :{args.port}", flush=True)
        conn, peer = server.accept()
        with conn:
            _send(conn, agent.get_agent_metadata())
            _send(conn, json.dumps({"addr": addr, "size": size}).encode())
            agent.add_remote_agent(_recv(conn))
            # The reader announces completion through NIXL, as the connector's
            # consumer does; wait for it before releasing the region.
            deadline = time.monotonic() + args.timeout_s
            while time.monotonic() < deadline:
                notifs = agent.get_new_notifs()
                if any(b"probe-done" in m for msgs in notifs.values() for m in msgs):
                    print(f"[probe] {peer[0]} finished reading", flush=True)
                    return 0
                time.sleep(0.01)
    print("[probe] no completion notification before the timeout", file=sys.stderr)
    del buf
    return 1


def read(args: argparse.Namespace) -> int:
    size = args.size_mib << 20
    buf, addr = _region(size, fill=None)
    agent = _agent("probe-consumer")
    local = agent.get_reg_descs([(addr, size, 0, "")], _NIXL_DRAM)
    agent.register_memory(local)
    with socket.create_connection((args.peer, args.port), timeout=60) as conn:
        remote_name = agent.add_remote_agent(_recv(conn))
        remote = json.loads(_recv(conn))
        _send(conn, agent.get_agent_metadata())
        src = agent.get_xfer_descs([(remote["addr"], size, 0)], _NIXL_DRAM)
        dst = agent.get_xfer_descs([(addr, size, 0)], _NIXL_DRAM)
        times = []
        for i in range(args.iters):
            notif = b"probe-done" if i == args.iters - 1 else b"probe-iter"
            handle = agent.initialize_xfer("READ", dst, src, remote_name, notif)
            start = time.perf_counter()
            if agent.transfer(handle) == "ERR":
                raise RuntimeError("NIXL rejected the READ")
            while (state := agent.check_xfer_state(handle)) == "PROC":
                pass
            elapsed = time.perf_counter() - start
            agent.release_xfer_handle(handle)
            if state != "DONE":
                raise RuntimeError(f"NIXL READ ended in state {state}")
            times.append(elapsed)
    expected = bytes((7 + i) % 251 for i in range(4096))
    intact = buf[:4096] == expected and buf[size - 4096 : size] == bytes(
        (7 + i) % 251 for i in range((1 << 20) - 4096, 1 << 20)
    )
    best = min(times[1:] or times)
    print(
        json.dumps(
            {
                "ucx_tls": os.environ.get("UCX_TLS", ""),
                "ucx_net_devices": os.environ.get("UCX_NET_DEVICES", ""),
                "ucx_transports_visible": _ucx_info(),
                "size_mib": args.size_mib,
                "iters": args.iters,
                "best_gbps": round(size / best / 1e9, 2),
                "median_gbps": round(size / sorted(times)[len(times) // 2] / 1e9, 2),
                "data_intact": intact,
            }
        ),
        flush=True,
    )
    return 0 if intact else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    r = sub.add_parser("read")
    for q in (s, r):
        q.add_argument("--port", type=int, default=5600)
        q.add_argument("--size-mib", type=int, default=1024)
    s.add_argument("--timeout-s", type=float, default=600.0)
    r.add_argument("--peer", required=True)
    r.add_argument("--iters", type=int, default=5)
    args = p.parse_args()
    return serve(args) if args.cmd == "serve" else read(args)


if __name__ == "__main__":
    sys.exit(main())
