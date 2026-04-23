# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests that span blocks remain immutable when gap recomputation runs.

Gap requests must write to freshly allocated blocks, not overwrite the
parent's (immutable) span blocks. The parent's block_ids are patched
in-place so both parent and gap request share the same fresh blocks
during the forward pass.
"""

import pytest

import vllm.envs as envs
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.request import Request

from .utils import EOS_TOKEN_ID, create_scheduler

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 16


def _init_hashing():
    init_none_hash(sha256)


def _make_span_request(
    req_id: str,
    prompt_len: int,
    span_starts: list[int],
) -> Request:
    extra_args = {"span_starts": span_starts}
    sampling_params = SamplingParams(
        max_tokens=17,
        extra_args=extra_args,
    )
    sampling_params.update_from_generation_config({}, eos_token_id=EOS_TOKEN_ID)
    block_hasher = get_request_block_hasher(BLOCK_SIZE, sha256)
    return Request(
        request_id=req_id,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=block_hasher,
    )


def _make_scheduler_with_gap_policy():
    from vllm.v1.core.sched.gap_policy import SpanAwareGapPolicy

    scheduler = create_scheduler(
        block_size=BLOCK_SIZE,
        enable_prefix_caching=True,
        max_num_batched_tokens=8192,
        enable_chunked_prefill=True,
    )
    scheduler.gap_policy = SpanAwareGapPolicy(
        gap_length=BLOCK_SIZE, block_size=BLOCK_SIZE
    )
    return scheduler


class TestImmutableSpanBlocks:
    def setup_method(self):
        self._original_spans = envs.VLLM_V1_SPANS_ENABLED
        envs.VLLM_V1_SPANS_ENABLED = True

    def teardown_method(self):
        envs.VLLM_V1_SPANS_ENABLED = self._original_spans

    def test_parent_and_gap_share_fresh_blocks(self):
        """Parent's block_ids are patched in-place so both parent and
        gap request point to the same fresh blocks at gap positions."""
        _init_hashing()
        scheduler = _make_scheduler_with_gap_policy()

        req = _make_span_request("req-0", 128, span_starts=[64])
        scheduler.add_request(req)

        output = scheduler.schedule()

        parent_nrd = None
        gap_nrds = []
        for nrd in output.scheduled_new_reqs:
            if nrd.is_gap_recompute:
                gap_nrds.append(nrd)
            elif nrd.req_id == "req-0":
                parent_nrd = nrd

        assert parent_nrd is not None, "Parent request not found"
        assert len(gap_nrds) >= 1, "No gap requests created"

        gap_nrd = gap_nrds[0]
        start_block = 64 // BLOCK_SIZE
        gap_len_blocks = 1  # gap_length == block_size -> 1 block

        # Parent and gap must point to the SAME fresh blocks at gap
        # positions — this is how the forward pass stays correct.
        for gid in range(len(parent_nrd.block_ids)):
            parent_gap = parent_nrd.block_ids[gid][
                start_block : start_block + gap_len_blocks
            ]
            gap_fresh = gap_nrd.block_ids[gid][
                start_block : start_block + gap_len_blocks
            ]
            assert parent_gap == gap_fresh, (
                f"Group {gid}: parent and gap request must share the "
                f"same fresh blocks at gap positions"
            )

    def test_prefix_blocks_shared(self):
        """Blocks before the gap position must be identical between
        parent and gap request (shared span/prefix blocks)."""
        _init_hashing()
        scheduler = _make_scheduler_with_gap_policy()

        req = _make_span_request("req-1", 128, span_starts=[64])
        scheduler.add_request(req)

        output = scheduler.schedule()

        parent_nrd = None
        gap_nrd = None
        for nrd in output.scheduled_new_reqs:
            if nrd.is_gap_recompute:
                gap_nrd = nrd
            elif nrd.req_id == "req-1":
                parent_nrd = nrd

        assert parent_nrd is not None
        assert gap_nrd is not None

        start_block = 64 // BLOCK_SIZE

        for gid in range(len(parent_nrd.block_ids)):
            parent_prefix = parent_nrd.block_ids[gid][:start_block]
            gap_prefix = gap_nrd.block_ids[gid][:start_block]
            assert parent_prefix == gap_prefix, (
                f"Group {gid}: prefix blocks must be shared"
            )

    def test_multiple_gaps_allocate_unique_blocks(self):
        """Each gap gets its own fresh blocks, no collisions."""
        _init_hashing()
        scheduler = _make_scheduler_with_gap_policy()

        req = _make_span_request("req-2", 256, span_starts=[64, 128])
        scheduler.add_request(req)

        output = scheduler.schedule()

        gap_nrds = [nrd for nrd in output.scheduled_new_reqs if nrd.is_gap_recompute]
        assert len(gap_nrds) == 2, "Expected 2 gap requests for 2 span starts"

        all_gap_block_ids: list[int] = []
        for nrd in gap_nrds:
            start_block = nrd.gap_start // BLOCK_SIZE
            for gid in range(len(nrd.block_ids)):
                all_gap_block_ids.append(nrd.block_ids[gid][start_block])

        assert len(all_gap_block_ids) == len(set(all_gap_block_ids)), (
            "Gap blocks must be unique across gaps"
        )

    def test_gap_blocks_registered_for_cleanup(self):
        """Fresh gap blocks are registered with the parent in the KV
        cache manager so they're freed when the parent finishes."""
        _init_hashing()
        scheduler = _make_scheduler_with_gap_policy()

        req = _make_span_request("req-3", 128, span_starts=[64])
        scheduler.add_request(req)

        blocks_before = set()
        for mgr in scheduler.kv_cache_manager.coordinator.single_type_managers:
            for blk in mgr.req_to_blocks.get("req-3", []):
                blocks_before.add(blk.block_id)

        output = scheduler.schedule()

        blocks_after = set()
        for mgr in scheduler.kv_cache_manager.coordinator.single_type_managers:
            for blk in mgr.req_to_blocks.get("req-3", []):
                blocks_after.add(blk.block_id)

        new_blocks = blocks_after - blocks_before
        assert len(new_blocks) >= 1, (
            "Gap blocks should be registered with parent request"
        )
