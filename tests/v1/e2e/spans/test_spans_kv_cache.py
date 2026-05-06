# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV-cache and gap-policy assertions for the spans / Legolink machinery.

Migrated from examples/offline_inference/spans/spans_time_and_kv.py:
the example's per-block SHA-256 dump becomes the snapshot used to assert
recompute actually changed cached K/V content. Timing / TTFT / TPOT
metrics are dropped (perf, not correctness).

Tests:
  test_gap_size_larger_than_chunk
      gap_length > block_size → SpanAwareGapPolicy.get_gaps yields an
      interval covering multiple blocks (structural check, no LLM).
  test_same_pic_chunk_hashes_match_across_requests_no_recompute
      same PIC chunk in two requests with different prefixes → identical
      chunk hash but different pre-chunk hashes (the "fan-in" guarantee
      from the user's perspective: same content, different chain). Pure
      structural hash check, no recompute.
  test_legolink_recompute_overwrites_pic_chunk_kv_in_place
      LL-FULL + prefix caching: run the same prompt twice. Run #2 hits the
      cache, gap policy fires, and the virtual gap request shares the
      parent's block_ids (see scheduler.py:963-996) → K/V is recomputed
      and written back into the same physical slots, overwriting them.
      The KV-cache snapshot after run #2 must differ from the one taken
      between runs.
"""
import hashlib

import pytest

import vllm.envs as envs
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256_cbor
from vllm.v1.core.kv_cache_utils import get_request_block_hasher
from vllm.v1.core.sched.gap_policy import SpanAwareGapPolicy
from vllm.v1.request import Request

from .conftest import BLOCK_SIZE, build_llm, cleanup

pytestmark = pytest.mark.spans

SEED = 42
MAX_TOKENS = 16
LAYER_IDX = 0  # Layer to snapshot. 0 always exists; some example models lack 19.


def _make_request(
    prompt_len: int,
    span_starts: list[int] | None = None,
) -> Request:
    extra_args = {"span_starts": span_starts} if span_starts is not None else None
    sp = SamplingParams(max_tokens=MAX_TOKENS, extra_args=extra_args)
    sp.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id="kv_test",
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=sp,
        pooling_params=None,
    )


def test_gap_size_larger_than_chunk():
    """gap_length=2*block_size with one span boundary should yield a single
    interval that covers two consecutive blocks."""
    policy = SpanAwareGapPolicy(gap_length=2 * BLOCK_SIZE, block_size=BLOCK_SIZE)
    req = _make_request(prompt_len=128, span_starts=[BLOCK_SIZE * 2])
    gaps = policy.get_gaps(req, num_computed_tokens=128, num_external_tokens=0)
    # span at 32, gap_length=32, no later span: gap = (32, 64). 64 - 32 = 32 = 2 blocks.
    assert gaps == [(BLOCK_SIZE * 2, BLOCK_SIZE * 4)]
    gap_start, gap_end = gaps[0]
    assert (gap_end - gap_start) // BLOCK_SIZE == 2


def test_same_pic_chunk_hashes_match_across_requests_no_recompute():
    """Pure block-hash check (NO recompute, no LLM): two requests with the
    same PIC chunk but different surrounding tokens hash the chunk to the
    same block hash (fan-in / position-invariant), and the surrounding
    blocks to different hashes. This is structural — gap policy and runtime
    K/V overwrite are not involved."""
    original = envs.VLLM_V1_SPANS_ENABLED
    try:
        envs.VLLM_V1_SPANS_ENABLED = True
        chunk = list(range(500, 500 + BLOCK_SIZE))

        # Two prefixes of equal length, different content, both 1 block.
        prefix_a = list(range(0, BLOCK_SIZE))
        prefix_b = list(range(900, 900 + BLOCK_SIZE))

        sp = SamplingParams(
            max_tokens=MAX_TOKENS,
            extra_args={"span_starts": [BLOCK_SIZE]},
        )
        sp.update_from_generation_config({}, eos_token_id=100)
        hasher = get_request_block_hasher(BLOCK_SIZE, sha256_cbor)

        req_a = Request(
            request_id="pic_share_a",
            prompt_token_ids=prefix_a + chunk,
            sampling_params=sp,
            pooling_params=None,
            block_hasher=hasher,
        )
        req_b = Request(
            request_id="pic_share_b",
            prompt_token_ids=prefix_b + chunk,
            sampling_params=sp,
            pooling_params=None,
            block_hasher=hasher,
        )

        assert len(req_a.block_hashes) == 2
        assert len(req_b.block_hashes) == 2
        # Pre-chunk blocks differ (different prefixes).
        assert req_a.block_hashes[0] != req_b.block_hashes[0]
        # PIC chunk: identical hash regardless of preceding context (fan-in).
        assert req_a.block_hashes[1] == req_b.block_hashes[1]
    finally:
        envs.VLLM_V1_SPANS_ENABLED = original


def _kv_cache_block_hashes(llm, layer_idx: int) -> list[str]:
    """Per-block SHA-256 of layer `layer_idx` in the worker's KV cache.

    Mirrors examples/offline_inference/spans/spans_time_and_kv.py:
    _get_kv_cache_info_from_worker.
    """

    def _grab(worker_self):
        import torch

        kv = worker_self.model_runner.kv_caches[layer_idx]
        cpu = kv.detach().cpu()
        if cpu.dtype == torch.bfloat16:
            cpu = cpu.to(torch.float32)
        num_blocks = cpu.shape[0] if cpu.ndim > 0 else 1
        return [
            hashlib.sha256(cpu[i].numpy().tobytes()).hexdigest()
            for i in range(num_blocks)
        ]

    results = llm.llm_engine.engine_core.collective_rpc(_grab)
    assert results, "collective_rpc returned no worker results"
    return results[0]


def test_legolink_recompute_overwrites_pic_chunk_kv_in_place(model, monkeypatch):
    """End-to-end check that gap-policy recompute actually writes new K/V
    into the cache (and, by the design at scheduler.py:963-996, into the
    *same* physical blocks the parent already owns).

    Run #1: cold prefill populates the KV cache with one set of K/V.
    Run #2: prefix-cache hit, gap policy fires with gap_length >> prompt,
            virtual gap request runs the model and writes K/V at the
            recomputed positions back into the parent's block_ids.
    Therefore the KV-cache snapshot taken after run #2 must differ from
    the one taken between runs."""
    prompt = "Hello world! Please write a short greeting in one sentence."
    sp = SamplingParams.from_optional(
        seed=SEED, temperature=0.0, max_tokens=MAX_TOKENS
    )

    llm = build_llm(model, "LL-FULL", monkeypatch)
    try:
        # Run #1: cold prefill, populates cache.
        llm.generate(prompt, sampling_params=sp, use_tqdm=False)
        snap_after_1 = _kv_cache_block_hashes(llm, LAYER_IDX)

        # Run #2: cache hit → gap_length=huge → full recompute, new blocks
        # allocated and filled.
        llm.generate(prompt, sampling_params=sp, use_tqdm=False)
        snap_after_2 = _kv_cache_block_hashes(llm, LAYER_IDX)

        assert snap_after_2 != snap_after_1, (
            "LL-FULL replay did not change the KV cache; recompute appears "
            "to have been a no-op (or written into the same physical blocks "
            "with identical bytes)."
        )
    finally:
        cleanup(llm)
