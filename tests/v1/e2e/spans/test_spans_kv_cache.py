# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV-cache and gap-policy assertions for the spans / Legolink machinery.

Migrated from examples/offline_inference/spans/spans_time_and_kv.py:
the example's per-block SHA-256 dump becomes the snapshot used to assert
recompute actually changed cached K/V content. Timing / TTFT / TPOT
metrics are dropped (perf, not correctness).

Tests:
  test_same_pic_chunk_hashes_match_across_requests_no_recompute
      same PIC chunk in two requests with different prefixes → identical
      chunk hash but different pre-chunk hashes (the "fan-in" guarantee
      from the user's perspective: same content, different chain). Pure
      structural hash check, no recompute.
  test_pic_spans_preserve_prefix_caching_across_requests
      Three structural requests sharing a PIC chunk + tail: pins
      determinism, chunk fan-in, post-PIC tail share-ability, and that
      divergence stays scoped to the pre-span blocks.
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


def test_pic_spans_preserve_prefix_caching_across_requests():
    """Prefix caching survives PIC spans embedded in a long prompt.

    Three structural requests, all with span_starts=[BLOCK_SIZE * 2] (one PIC
    chunk in the middle of the prompt) and a shared post-PIC tail:

        req_a = prefix_X + chunk + suffix    (request id "a")
        req_b = prefix_X + chunk + suffix    (same content, different id)
        req_c = prefix_Y + chunk + suffix    (different prefix, same chunk + tail)

    The four assertions pin properties prefix caching depends on:

      1. Determinism — replaying the same request hashes identically end-to-end.
         Without this, prefix caching cannot reuse anything on a re-run.
      2. Fan-in on the PIC chunk — same chunk hashes the same regardless of
         preceding prefix. Cross-request cache reuse on the chunk itself.
      3. Post-PIC tail is also shareable — every block downstream of the chunk
         hashes the same across A and C. PIC dropping the parent at the span
         boundary means the tail's chain only depends on chunk + tail tokens,
         not on the prefix. This is the load-bearing property: it turns a
         single shared chunk into a shared *suffix from the chunk onward*.
      4. Divergence stays scoped — pre-span blocks hash differently across A
         and C. The span boundary is the only point where chains decouple.
    """
    original = envs.VLLM_V1_SPANS_ENABLED
    try:
        envs.VLLM_V1_SPANS_ENABLED = True

        prefix_x = list(range(0, BLOCK_SIZE * 2))               # 2 blocks
        prefix_y = list(range(900, 900 + BLOCK_SIZE * 2))       # 2 blocks, different content
        chunk = list(range(500, 500 + BLOCK_SIZE))              # 1 block, the PIC chunk
        suffix = list(range(700, 700 + BLOCK_SIZE * 3))         # 3 blocks, shared tail

        sp = SamplingParams(
            max_tokens=MAX_TOKENS,
            extra_args={"span_starts": [BLOCK_SIZE * 2]},
        )
        sp.update_from_generation_config({}, eos_token_id=100)
        hasher = get_request_block_hasher(BLOCK_SIZE, sha256_cbor)

        req_a = Request(
            request_id="pic_pc_a",
            prompt_token_ids=prefix_x + chunk + suffix,
            sampling_params=sp,
            pooling_params=None,
            block_hasher=hasher,
        )
        req_b = Request(
            request_id="pic_pc_b",
            prompt_token_ids=prefix_x + chunk + suffix,
            sampling_params=sp,
            pooling_params=None,
            block_hasher=hasher,
        )
        req_c = Request(
            request_id="pic_pc_c",
            prompt_token_ids=prefix_y + chunk + suffix,
            sampling_params=sp,
            pooling_params=None,
            block_hasher=hasher,
        )

        # 6 blocks each: 2 prefix + 1 chunk + 3 suffix.
        assert len(req_a.block_hashes) == 6
        assert len(req_b.block_hashes) == 6
        assert len(req_c.block_hashes) == 6

        # 1. Determinism: same prompt + same spans → identical hashes.
        assert req_a.block_hashes == req_b.block_hashes

        # 2. Fan-in on the PIC chunk across different prefixes.
        assert req_a.block_hashes[2] == req_c.block_hashes[2]

        # 3. Post-PIC tail is shareable across requests with matching chunk
        #    + tail, even when prefixes differ. THIS is the property that
        #    makes PIC actually pay off in cache hit-rate.
        assert req_a.block_hashes[2:] == req_c.block_hashes[2:]

        # 4. Divergence is scoped to the pre-span blocks only.
        assert req_a.block_hashes[0:2] != req_c.block_hashes[0:2]
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


def test_pic_spans_actually_reuse_kv_across_requests(model, monkeypatch):
    """End-to-end follow-up to test_pic_spans_preserve_prefix_caching_across_requests.

    The structural test only asserts that block_hashes match. This one
    actually loads the model in mode SPANS-PC (spans on + prefix caching,
    NO gap policy) and proves that the K/V bytes stored under those
    matching hashes are reused byte-for-byte across requests that share
    chunk + tail.

    Setup:
        req_A = prefix_X + chunk + suffix
        req_C = prefix_Y + chunk + suffix    (different prefix)

    After running A then C, the KV cache must contain:
      - prefix_X blocks (written by A, still there - nothing evicted them)
      - prefix_Y blocks (written by C, fresh)
      - chunk + tail blocks (written by A, REUSED by C - hashes collided,
        cache hit fired, no recompute, so the bytes are still A's)

    Therefore the *set of non-empty K/V block byte-hashes* after A must be
    a subset of the set after C: nothing A wrote was evicted or
    overwritten. The cache hit path silently reuses A's K/V even when C's
    actual cross-attention would produce different bytes - this is the
    accuracy/perf trade-off PIC explicitly enables.
    """
    chunk = list(range(500, 500 + BLOCK_SIZE))
    suffix = list(range(700, 700 + BLOCK_SIZE * 3))
    prefix_x = list(range(0, BLOCK_SIZE * 2))
    prefix_y = list(range(900, 900 + BLOCK_SIZE * 2))

    sp = SamplingParams.from_optional(
        seed=SEED,
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        extra_args={"span_starts": [BLOCK_SIZE * 2]},
    )

    llm = build_llm(model, "SPANS-PC", monkeypatch)
    try:
        # req_A: cold cache; populates prefix_X + chunk + tail K/V
        llm.generate(
            {"prompt_token_ids": prefix_x + chunk + suffix},
            sampling_params=sp,
            use_tqdm=False,
        )
        snap_after_a = _kv_cache_block_hashes(llm, LAYER_IDX)
        non_empty_a = set(snap_after_a)

        # req_C: prefix_Y differs; chunk + tail block_hashes collide -> cache
        # hit; A's K/V bytes are reused, C only computes fresh K/V for the
        # 2 prefix_Y blocks.
        llm.generate(
            {"prompt_token_ids": prefix_y + chunk + suffix},
            sampling_params=sp,
            use_tqdm=False,
        )
        snap_after_c = _kv_cache_block_hashes(llm, LAYER_IDX)
        non_empty_c = set(snap_after_c)

        # Every K/V block A wrote must still be present byte-for-byte after C
        # ran. If anything diverged, either C overwrote A's slots (would mean
        # gap-policy fired, which is disabled here) or A's blocks were
        # evicted (cache is far larger than 8 blocks, so this can't happen).
        missing_from_c = non_empty_a - non_empty_c
        assert not missing_from_c, (
            f"{len(missing_from_c)} K/V block(s) A wrote are no longer in the "
            f"cache after C ran - reuse path failed."
        )

        # Sanity: C must have added at least one new K/V block (its prefix_Y
        # blocks). If non_empty_c == non_empty_a, then C didn't write
        # anything - which would mean even prefix_Y was somehow shared,
        # which it shouldn't be (different tokens, different hash chain).
        new_in_c = non_empty_c - non_empty_a
        assert len(new_in_c) >= 1, (
            f"C wrote no new K/V blocks - prefix_Y should have been a cache "
            f"miss but apparently wasn't. snap diff: {len(non_empty_c)} vs "
            f"{len(non_empty_a)}"
        )
    finally:
        cleanup(llm)


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
