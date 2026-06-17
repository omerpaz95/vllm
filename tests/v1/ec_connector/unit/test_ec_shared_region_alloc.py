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
    # free list after free(a): [3,4,5,6,7, 0,1,2]
    b = region.alloc(5)
    assert b == [3, 4, 5, 6, 7]
    # free list now: [0,1,2]  (the freed ones)
    c = region.alloc(3)
    assert sorted(c) == [0, 1, 2]  # the freed ones come back


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
        assert b.alloc(2) == [0, 1]  # independent, not [2, 3]
    finally:
        a.cleanup()
        b.cleanup()
