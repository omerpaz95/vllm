# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Auto-loaded by Python at interpreter startup (the `site` module imports
`sitecustomize` if it is importable) when this directory is on PYTHONPATH.

Patches `ECExampleConnector` -- the debug/reference EC connector used by the
`example-grid`/`example-data` arms -- to defer a request instead of routing
it to local encode when a remote encoding hasn't landed yet.

`ECExampleConnector.has_cache_item` is a bare `os.path.exists` check with no
wait and no retry: the scheduler calls it once per step to decide whether an
item loads from the connector or falls to local encode
(`vllm/v1/core/sched/scheduler.py`, the `_try_schedule_encoder_inputs` loop).
A miss commits the item to local encode for that step. For a decode instance
run with `--enable-mm-embeds`, there is no raw pixel data to fall back on --
the proxy already rewrote the item to a metadata-only reference -- so a miss
during the brief window between the encoder starting its save and the file
appearing on shared storage crashes the whole engine
(`AssertionError` in `vllm.v1.worker.utils.sanity_check_mm_encoder_outputs`,
"but got len(mm_embeddings)=0 instead").

This mirrors the `ECCPUConnector`'s own fix for the equivalent race
(https://github.com/vllm-project/vllm/pull/55290, "Fail the request, not the
engine, when a remote encoding cannot arrive") by hooking the same two
connector-agnostic scheduler entry points that fix relies on:
  * `ensure_cache_available` -- checked before the load-vs-encode commit; a
    `False` return defers the request to a later step instead.
  * `take_unavailable_requests` -- drained by the scheduler every step to
    fail specific requests without touching the engine.
`ECExampleConnector` inherits only the base class's no-op defaults for both
(`ensure_cache_available` always True, `take_unavailable_requests` always
empty), which is what leaves it exposed to the race.

The wait is bounded: a request stuck past `_DEFER_TIMEOUT_S` is added to the
connector's own unavailable set (drained above) rather than deferred forever.

Installs `internvl_ec_patch`, which lets an InternVL (`-HF`) EC consumer take
a metadata-only image so the grid arms run for that model; it touches the
model module only if a server imports it.

Installs `encoder_budget_patch`, which leaves a modality limited to 0 items
out of the GPU encoder cache's sizing, so a connector server's cache is the
same size as the baseline's.

Also drops the debug output of `_QUIET_LOGGERS`. The pods run at DEBUG for the
connector's accounting, and these two log a line per forward pass and per
loaded expert: hundreds of MB per server, which the harness's `oc exec` log
reads cannot move within their timeout.
"""

import logging
import time

_DEFER_TIMEOUT_S = 10.0
_QUIET_LOGGERS = (
    "vllm.v1.attention.backends.flash_attn",
    "vllm.model_executor.layers.fused_moe.routed_experts",
)


def _quiet_debug_floods() -> None:
    # An instance attribute, not setLevel: vLLM's dictConfig resets the level
    # of every existing `vllm.*` logger when it configures logging.
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).debug = lambda *args, **kwargs: None


def _patch_example_connector() -> None:
    try:
        from vllm.distributed.ec_transfer.ec_connector import example_connector as ec
    except ImportError:
        return

    connector_cls = ec.ECExampleConnector
    logger = ec.logger

    def ensure_cache_available(self, request, num_computed_tokens: int) -> bool:
        if not request.ec_transfer_params:
            return True
        deferred_since: dict[tuple[str, str], float] = self.__dict__.setdefault(
            "_defer_since", {}
        )
        unavailable: set[str] = self.__dict__.setdefault("_defer_unavailable", set())
        now = time.monotonic()
        for feature in request.mm_features:
            pos = feature.mm_position
            mm_hash = feature.identifier
            key = (request.request_id, mm_hash)
            if pos.offset + pos.length <= num_computed_tokens:
                # Already computed; this connector no longer needs it.
                deferred_since.pop(key, None)
                continue
            if self.has_cache_item(mm_hash):
                deferred_since.pop(key, None)
                continue
            since = deferred_since.setdefault(key, now)
            if now - since <= _DEFER_TIMEOUT_S:
                # Still within budget: defer the whole request this step.
                return False
            deferred_since.pop(key, None)
            unavailable.add(request.request_id)
            fail_count = self.__dict__["_defer_fail_count"] = (
                self.__dict__.get("_defer_fail_count", 0) + 1
            )
            logger.error(
                "EC consumer: request %s mm_hash=%s unavailable after %.0fs "
                "(deferred-fail tally=%d)",
                request.request_id,
                mm_hash,
                now - since,
                fail_count,
            )
            return False
        return True

    def take_unavailable_requests(self) -> set[str]:
        unavailable: set[str] = self.__dict__.setdefault("_defer_unavailable", set())
        ids, unavailable = unavailable, set()
        self.__dict__["_defer_unavailable"] = unavailable
        return ids

    connector_cls.ensure_cache_available = ensure_cache_available
    connector_cls.take_unavailable_requests = take_unavailable_requests


def _install_internvl_patch() -> None:
    try:
        import internvl_ec_patch
    except ImportError:
        return
    internvl_ec_patch.install()


def _install_encoder_budget_patch() -> None:
    try:
        import encoder_budget_patch
    except ImportError:
        return
    encoder_budget_patch.install()


_quiet_debug_floods()
_patch_example_connector()
_install_internvl_patch()
_install_encoder_budget_patch()
