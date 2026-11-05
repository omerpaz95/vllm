# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check `encoder_budget_patch` against vLLM's own `MultiModalBudget`.

Builds the GPU encoder cache budget the way a server does, for InternVL with
the benchmark's `--limit-mm-per-prompt '{"video":0}'`: once as the baseline
(no connector, so `enable_mm_embeds` stays off) and once as an EC consumer
(`enable_mm_embeds` on). Unpatched, the consumer's cache is sized for a
maximal video no request carries; patched, the two caches are equal. No
weights and no GPU: only the checkpoint's config and processor files are read.

    INTERNVL_HF_DIR=/path/to/InternVL3_5-241B-A28B-HF-configs \\
        python -m pytest test_encoder_budget_patch.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

MODEL_DIR = os.environ.get("INTERNVL_HF_DIR", "")
pytestmark = pytest.mark.skipif(
    not MODEL_DIR, reason="set INTERNVL_HF_DIR to a -HF checkpoint's config files"
)


def _cache_size(consumer: bool, limits: dict) -> int:
    from vllm.config import ModelConfig, SchedulerConfig, VllmConfig
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.encoder_budget import MultiModalBudget

    model = ModelConfig(
        MODEL_DIR,
        tokenizer=MODEL_DIR,
        dtype="bfloat16",
        mm_processor_kwargs={"crop_to_patches": True},
        limit_mm_per_prompt=limits,
        enable_mm_embeds=consumer,
        max_model_len=32768,
    )
    scheduler = SchedulerConfig(
        max_num_batched_tokens=8192, max_model_len=32768, is_encoder_decoder=False
    )
    config = VllmConfig(model_config=model, scheduler_config=scheduler)
    return MultiModalBudget(config, MULTIMODAL_REGISTRY).encoder_cache_size


@pytest.fixture
def patched():
    import encoder_budget_patch

    import vllm.multimodal.encoder_budget as mod

    saved = (mod.MultiModalBudget.__init__, mod.compute_mm_encoder_budget)
    encoder_budget_patch.install()
    yield
    mod.MultiModalBudget.__init__, mod.compute_mm_encoder_budget = saved
    delattr(mod, "_ec_bench_budget_patch")


NO_VIDEO = {"image": 8, "video": 0}


def test_unpatched_consumer_is_sized_for_a_video() -> None:
    baseline = _cache_size(consumer=False, limits=NO_VIDEO)
    consumer = _cache_size(consumer=True, limits=NO_VIDEO)
    assert consumer > baseline, (baseline, consumer)


def test_patched_consumer_matches_the_baseline(patched) -> None:
    assert _cache_size(consumer=True, limits=NO_VIDEO) == _cache_size(
        consumer=False, limits=NO_VIDEO
    )


WITH_VIDEO = {"image": 8, "video": 1}


@pytest.fixture
def unpatched_with_video() -> int:
    return _cache_size(consumer=True, limits=WITH_VIDEO)


def test_a_modality_a_request_can_carry_still_counts(
    unpatched_with_video, patched
) -> None:
    # With video allowed, the patch leaves vLLM's sizing as it is.
    assert _cache_size(consumer=True, limits=WITH_VIDEO) == unpatched_with_video
