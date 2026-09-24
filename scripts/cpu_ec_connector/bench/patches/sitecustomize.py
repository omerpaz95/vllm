# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Server-side patches for the EC offload benchmark.

Two independent pieces. The first, always applied, logs the connector's
transfer accounting at INFO so the servers can run at INFO: the harness parses
those lines out of the server log, and vLLM's remaining debug output costs
throughput and buries them in roughly a million lines per server.

The second is descriptor-count instrumentation, applied only when
`EC_BENCH_FRAG_FILE` is set.

Wraps `_coalesce_runs` -- the function that decides how many descriptors an
entry's blocks collapse into -- and records `blocks in` against `descriptors
out`. That ratio is the fragmentation metric: a contiguous entry yields one
descriptor for hundreds of blocks, a scattered one yields nearly as many
descriptors as blocks. Effective bandwidth alone cannot distinguish
fragmentation from PCIe contention or NUMA placement; this can.

The connector is not modified. Both call sites resolve `_coalesce_runs`
through module globals at call time, so replacing the module attribute is
enough.

This file lives in its own directory precisely so it is NOT auto-imported by
the benchmark scripts one level up: Python puts a script's own directory on
`sys.path`, and any `sitecustomize` found there loads automatically. Put
*this* directory on PYTHONPATH only for the server process.

A leaked PYTHONPATH therefore raises two loggers' accounting to INFO in
whatever interpreter picks this up. Descriptor counting stays off until
`EC_BENCH_FRAG_FILE` is set.

Descriptor counting adds a Python frame and a dict update per call -- about
1 us against a ~40 us descriptor build. Small, but do not report latencies
from a process running it: run the timed arms with `EC_BENCH_FRAG_FILE` unset
and diagnose fragmentation in a separate run.
"""

import atexit
import json
import logging
import os
import sys
import time

_OUT_PATH = os.environ.get("EC_BENCH_FRAG_FILE")
_FLUSH_INTERVAL_S = float(os.environ.get("EC_BENCH_FRAG_FLUSH_S", "10"))

_ACCOUNTING_MARKER = "[ec-bench] EC accounting logged at INFO"
# Every logger carrying a line `ec_log_stats` parses at DEBUG: the per-transfer
# accounting on the CPU connector's worker, and the hit lines on the example
# connector.
_ACCOUNTING_LOGGERS = (
    "vllm.distributed.ec_transfer.ec_connector.cpu.worker",
    "vllm.distributed.ec_transfer.ec_connector.example_connector",
)


def _accounting_to_info() -> None:
    """Send these loggers' debug records through `info` instead.

    `logging.getLogger` returns the same object `vllm.logger.init_logger` hands
    the connector, so this holds whether or not vllm has been imported yet --
    under vLLM's own logging config, which leaves `disable_existing_loggers`
    false and so keeps loggers created before it ran.

    Every debug record on these two loggers is promoted, not just the accounting
    ones: the CPU worker's region-cleanup failure carries `exc_info`, so that
    one now prints a traceback at INFO. The harness's patterns match any level
    word, so the promotion itself needs no change on the parsing side.

    The marker goes to stderr because logging is not configured yet; `verify()`
    requires it in a fresh server log, so a PYTHONPATH that never reached the
    server fails as itself instead of as an arm that transferred nothing.
    """
    for name in _ACCOUNTING_LOGGERS:
        logger = logging.getLogger(name)
        logger.debug = logger.info
    print(_ACCOUNTING_MARKER, file=sys.stderr)


_accounting_to_info()


def _install(out_path):
    from vllm.distributed.ec_transfer.ec_connector.cpu import worker as worker_mod

    original = worker_mod._coalesce_runs
    # (elapsed_second, caller) -> [calls, blocks_in, descriptors_out]
    buckets: dict[tuple[int, str], list[int]] = {}
    started = time.time()
    state = {"last_flush": 0.0}
    pid = os.getpid()

    def flush() -> None:
        if not buckets:
            return
        rows = [
            {
                "pid": pid,
                "t_s": second,
                "caller": caller,
                "calls": calls,
                "blocks": blocks,
                "descriptors": descriptors,
                # 1.0 means every block needed its own descriptor (worst case);
                # blocks/descriptors is the mean run length.
                "blocks_per_descriptor": round(blocks / descriptors, 2)
                if descriptors
                else 0,
            }
            for (second, caller), (calls, blocks, descriptors) in sorted(
                buckets.items()
            )
        ]
        buckets.clear()
        try:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(
                    "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows)
                )
        except OSError:
            pass  # never let instrumentation break the server

    def wrapped(block_ids):
        slots, first_blocks, num_blocks = original(block_ids)
        # The caller's name separates the save path from the load path without
        # threading a flag through the connector.
        caller = sys._getframe(1).f_code.co_name
        elapsed = time.time() - started
        entry = buckets.get((int(elapsed), caller))
        if entry is None:
            entry = buckets[(int(elapsed), caller)] = [0, 0, 0]
        entry[0] += 1
        entry[1] += len(block_ids)
        entry[2] += int(slots.size)
        # Periodic rather than signal-driven: vLLM owns SIGTERM, and a flushed
        # prefix survives even a SIGKILL.
        if elapsed - state["last_flush"] >= _FLUSH_INTERVAL_S:
            state["last_flush"] = elapsed
            flush()
        return slots, first_blocks, num_blocks

    worker_mod._coalesce_runs = wrapped
    atexit.register(flush)
    print(f"[ec-bench] descriptor counting active -> {out_path}", file=sys.stderr)


if _OUT_PATH:
    try:
        _install(_OUT_PATH)
    except ImportError as exc:
        # Probe interpreters that never load vllm land here; nothing to patch.
        print(f"[ec-bench] skipping instrumentation ({exc})", file=sys.stderr)
