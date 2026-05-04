# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for ECCPUConnector.
"""

from unittest.mock import Mock, patch

import pytest
import torch

from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
from vllm.distributed.ec_transfer.ec_connector.cpu_connector import (
    ECCPUConnector,
    ECCPUConnectorMetadata,
)
from vllm.v1.core.sched.output import SchedulerOutput

NUM_BLOCKS = 16
HIDDEN_DIM = 32
DTYPE = torch.float32
ELEMENT_SIZE = 4  # float32
BLOCK_SIZE_BYTES = HIDDEN_DIM * ELEMENT_SIZE


def _make_mock_config(
    num_ec_blocks: int = NUM_BLOCKS,
) -> Mock:
    config = Mock(spec=VllmConfig)
    config.ec_transfer_config = Mock()
    config.ec_transfer_config.is_ec_producer = False
    config.ec_transfer_config.is_ec_consumer = True
    config.ec_transfer_config.engine_id = "test-engine-id"
    config.ec_transfer_config.get_from_extra_config = lambda key, default: {
        "num_ec_blocks": num_ec_blocks,
    }.get(key, default)
    config.parallel_config = Mock()
    config.parallel_config.rank = 0
    config.model_config = Mock()
    config.model_config.get_inputs_embeds_size.return_value = HIDDEN_DIM
    config.model_config.dtype = DTYPE
    return config


def _make_worker_connector(
    num_ec_blocks: int = NUM_BLOCKS,
) -> ECCPUConnector:
    """Create a worker connector with ECSharedRegion patched out."""
    config = _make_mock_config(num_ec_blocks=num_ec_blocks)

    cpu_blocks = torch.arange(
        num_ec_blocks * BLOCK_SIZE_BYTES,
        dtype=torch.uint8,
    ).reshape(num_ec_blocks, BLOCK_SIZE_BYTES)

    with patch(
        "vllm.distributed.ec_transfer.ec_connector.cpu_connector.ECSharedRegion"
    ) as mock_region_cls:
        mock_region_cls.return_value.blocks = cpu_blocks
        connector = ECCPUConnector(
            vllm_config=config,
            role=ECConnectorRole.WORKER,
        )
    return connector


@pytest.fixture
def scheduler_connector():
    config = _make_mock_config()
    return ECCPUConnector(
        vllm_config=config,
        role=ECConnectorRole.SCHEDULER,
    )


@pytest.fixture
def worker_connector():
    return _make_worker_connector()


class TestBuildConnectorMeta:
    def test_returns_metadata_with_encoding_map(self, scheduler_connector):
        scheduler_connector._encoding_map = {
            "hash_a": [0, 1, 2],
            "hash_b": [3, 4],
        }
        scheduler_output = Mock(spec=SchedulerOutput)

        meta = scheduler_connector.build_connector_meta(scheduler_output)

        assert isinstance(meta, ECCPUConnectorMetadata)
        assert meta.mm_hash_to_cpu_blocks == {
            "hash_a": [0, 1, 2],
            "hash_b": [3, 4],
        }

    def test_resets_encoding_map_after_build(self, scheduler_connector):
        scheduler_connector._encoding_map = {
            "hash_a": [0, 1],
            "hash_b": [2, 3],
        }
        scheduler_output = Mock(spec=SchedulerOutput)

        scheduler_connector.build_connector_meta(scheduler_output)

        assert scheduler_connector._encoding_map == {}

    def test_consecutive_builds_are_independent(self, scheduler_connector):
        scheduler_output = Mock(spec=SchedulerOutput)

        scheduler_connector._encoding_map = {"hash_a": [0, 1]}
        meta1 = scheduler_connector.build_connector_meta(scheduler_output)

        scheduler_connector._encoding_map = {"hash_b": [5, 6]}
        meta2 = scheduler_connector.build_connector_meta(scheduler_output)

        assert meta1.mm_hash_to_cpu_blocks == {"hash_a": [0, 1]}
        assert meta2.mm_hash_to_cpu_blocks == {"hash_b": [5, 6]}

    def test_empty_encoding_map(self, scheduler_connector):
        scheduler_output = Mock(spec=SchedulerOutput)

        meta = scheduler_connector.build_connector_meta(scheduler_output)

        assert isinstance(meta, ECCPUConnectorMetadata)
        assert meta.mm_hash_to_cpu_blocks == {}


class TestStartLoadCaches:
    @patch("vllm.distributed.ec_transfer.ec_connector.cpu_connector.current_platform")
    def test_gathers_correct_blocks(self, mock_platform, worker_connector):
        mock_platform.device_type = "cpu"

        metadata = ECCPUConnectorMetadata(mm_hash_to_cpu_blocks={"hash_a": [2, 5]})
        worker_connector.bind_connector_metadata(metadata)

        encoder_cache: dict[str, torch.Tensor] = {}
        worker_connector.start_load_caches(encoder_cache=encoder_cache)

        assert "hash_a" in encoder_cache
        expected = worker_connector._cpu_blocks[[2, 5]]
        assert torch.equal(encoder_cache["hash_a"], expected)

    @patch("vllm.distributed.ec_transfer.ec_connector.cpu_connector.current_platform")
    def test_skips_existing_hash_in_encoder_cache(
        self, mock_platform, worker_connector
    ):
        mock_platform.device_type = "cpu"

        existing_tensor = torch.zeros(2, BLOCK_SIZE_BYTES)
        encoder_cache: dict[str, torch.Tensor] = {"hash_a": existing_tensor}

        metadata = ECCPUConnectorMetadata(mm_hash_to_cpu_blocks={"hash_a": [0, 1]})
        worker_connector.bind_connector_metadata(metadata)
        worker_connector.start_load_caches(encoder_cache=encoder_cache)

        assert torch.equal(encoder_cache["hash_a"], existing_tensor)

    def test_empty_metadata(self, worker_connector):
        metadata = ECCPUConnectorMetadata(mm_hash_to_cpu_blocks={})
        worker_connector.bind_connector_metadata(metadata)

        encoder_cache: dict[str, torch.Tensor] = {}
        worker_connector.start_load_caches(encoder_cache=encoder_cache)

        assert len(encoder_cache) == 0

    @patch("vllm.distributed.ec_transfer.ec_connector.cpu_connector.current_platform")
    def test_loads_multiple_encodings(self, mock_platform, worker_connector):
        mock_platform.device_type = "cpu"

        metadata = ECCPUConnectorMetadata(
            mm_hash_to_cpu_blocks={
                "hash_a": [0, 1],
                "hash_b": [4, 5, 6],
                "hash_c": [10],
            }
        )
        worker_connector.bind_connector_metadata(metadata)

        encoder_cache: dict[str, torch.Tensor] = {}
        worker_connector.start_load_caches(encoder_cache=encoder_cache)

        assert len(encoder_cache) == 3
        assert torch.equal(
            encoder_cache["hash_a"],
            worker_connector._cpu_blocks[[0, 1]],
        )
        assert torch.equal(
            encoder_cache["hash_b"],
            worker_connector._cpu_blocks[[4, 5, 6]],
        )
        assert torch.equal(
            encoder_cache["hash_c"],
            worker_connector._cpu_blocks[[10]],
        )


class TestMetadataLifecycle:
    def test_bind_and_retrieve(self, worker_connector):
        metadata = ECCPUConnectorMetadata(mm_hash_to_cpu_blocks={"h": [0, 1]})
        worker_connector.bind_connector_metadata(metadata)

        assert worker_connector._get_connector_metadata() is metadata

    def test_clear(self, worker_connector):
        metadata = ECCPUConnectorMetadata(mm_hash_to_cpu_blocks={})
        worker_connector.bind_connector_metadata(metadata)
        worker_connector.clear_connector_metadata()

        assert worker_connector._connector_metadata is None

    def test_get_without_bind_raises(self, worker_connector):
        with pytest.raises(AssertionError):
            worker_connector._get_connector_metadata()
