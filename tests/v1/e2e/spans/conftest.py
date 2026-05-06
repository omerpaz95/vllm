# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared fixtures and helpers for spans / Legolink e2e tests.

Mode names:
    FR      - VLLM_V1_SPANS_ENABLED=False, no gap policy (full recompute baseline).
    SPANS   - VLLM_V1_SPANS_ENABLED=True, no gap policy.
    LL-16   - Legolink: span-aware gap policy with gap_length == block_size,
              prefix caching enabled.
    LL-FULL - Legolink: span-aware gap policy with gap_length >> prompt length,
              prefix caching enabled. On a cache hit this forces a full
              recompute of the cached prefix, which means run #2 ≡ FR.
"""
import gc

import pytest
import torch

import vllm.envs as envs
from vllm import LLM

BLOCK_SIZE = 16
SPAN_TOKEN_PLUS = 10
SPAN_TOKEN_CROSS = 31
HUGE_GAP_LENGTH = 1_000_000

MODELS = ["Qwen/Qwen3-0.6B", "NousResearch/Meta-Llama-3.1-8B-Instruct"]
LARGE_MODELS = {"NousResearch/Meta-Llama-3.1-8B-Instruct"}
LARGE_MODEL_MIN_GIB = 24


def _has_enough_gpu_for(model: str) -> bool:
    if not torch.cuda.is_available():
        return False
    if model not in LARGE_MODELS:
        return True
    total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return total_gib >= LARGE_MODEL_MIN_GIB


@pytest.fixture(params=MODELS, ids=["qwen3_0_6b", "llama3_1_8b"])
def model(request) -> str:
    name = request.param
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for spans e2e tests")
    if not _has_enough_gpu_for(name):
        pytest.skip(
            f"{name} needs >= {LARGE_MODEL_MIN_GIB} GiB GPU memory; "
            f"have {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB"
        )
    return name


def _set_spans_env(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    monkeypatch.setattr(envs, "VLLM_V1_SPANS_ENABLED", enabled)


def build_llm(
    model: str,
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> LLM:
    """Construct an LLM configured for one of FR / SPANS / LL-16 / LL-FULL."""
    if mode == "FR":
        spans_enabled = False
        gap_policy_name = None
        gap_policy_config = None
        enable_prefix_caching = False
    elif mode == "SPANS":
        spans_enabled = True
        gap_policy_name = None
        gap_policy_config = None
        enable_prefix_caching = False
    elif mode == "LL-16":
        spans_enabled = True
        gap_policy_name = "span_aware"
        gap_policy_config = {
            "gap_length": BLOCK_SIZE,
            "block_size": BLOCK_SIZE,
        }
        enable_prefix_caching = True
    elif mode == "LL-FULL":
        spans_enabled = True
        gap_policy_name = "span_aware"
        gap_policy_config = {
            "gap_length": HUGE_GAP_LENGTH,
            "block_size": BLOCK_SIZE,
        }
        enable_prefix_caching = True
    else:
        raise ValueError(f"unknown mode: {mode}")

    _set_spans_env(monkeypatch, spans_enabled)

    extra: dict = {}
    if gap_policy_name is not None:
        extra["gap_policy_name"] = gap_policy_name
        extra["gap_policy_config"] = gap_policy_config

    return LLM(
        model=model,
        tensor_parallel_size=1,
        kv_transfer_config=None,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        block_size=BLOCK_SIZE,
        enable_prefix_caching=enable_prefix_caching,
        async_scheduling=False,
        attention_backend="TRITON_ATTN",
        **extra,
    )


def cleanup(llm: LLM | None) -> None:
    if llm is not None:
        del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def extract_step0_topk(out, topk: int = 10) -> list[tuple[int, float]]:
    """Step-0 top-K logprobs as a stably-sorted list. Bit-exact equality across
    runs ⇔ bit-exact match of the top-K distribution.
    """
    if out.logprobs is None or len(out.logprobs) == 0:
        return []
    items = [(tid, float(lp.logprob)) for tid, lp in out.logprobs[0].items()]
    items.sort(key=lambda x: (-x[1], x[0]))
    return items[:topk]
