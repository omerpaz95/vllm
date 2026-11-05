# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check `internvl_ec_patch` against vLLM's own InternVL processor.

Runs the real multimodal processor twice, the way an EPD pair does: once on
an image (the encoder) and once on the metadata the encoder publishes (the
decode, which gets no pixels). The placeholder ranges must match exactly, or
the connector's load lands in the wrong tokens. No weights and no GPU: only
the checkpoint's config, tokenizer and processor files are read.

    INTERNVL_HF_DIR=/path/to/InternVL3_5-241B-A28B-HF-configs \\
        python -m pytest test_internvl_ec_patch.py
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
TILED = {"crop_to_patches": True}


def _processor(consumer: bool):
    import internvl_ec_patch

    internvl_ec_patch.install()
    from vllm.config import ModelConfig
    from vllm.multimodal import MULTIMODAL_REGISTRY

    config = ModelConfig(
        MODEL_DIR,
        tokenizer=MODEL_DIR,
        dtype="bfloat16",
        mm_processor_kwargs=TILED,
        limit_mm_per_prompt={"image": 2, "video": 0},
        enable_mm_embeds=consumer,
        skip_tokenizer_init=False,
    )
    # What VllmConfig sets on an EC consumer: the embeddings arrive through
    # the connector, so a request may carry only the placeholder metadata.
    config.multimodal_config.allow_missing_mm_embeddings = consumer
    return MULTIMODAL_REGISTRY.create_processor(config)


def _prompt(processor, n_images: int) -> str:
    image = processor.info.get_hf_processor().image_token
    return "Question " + " ".join(["<IMG_CONTEXT>"] * n_images).replace(
        "<IMG_CONTEXT>", image
    )


@pytest.mark.parametrize("sizes", [[(2048, 2048)], [(1700, 2200), (500, 300)]])
def test_metadata_only_matches_pixel_placeholders(sizes):
    from PIL import Image

    from vllm.distributed.ec_transfer.ec_connector.utils import (
        PlaceholderMetadataResolver,
    )

    encoder = _processor(consumer=False)
    images = [Image.new("RGB", wh, (120, 80, 40)) for wh in sizes]
    prompt = _prompt(encoder, len(images))
    enc_out = encoder(
        prompt,
        encoder.info.data_parser.parse_mm_data({"image": images}),
        hf_processor_mm_kwargs=TILED,
    )
    enc_ranges = enc_out["mm_placeholders"]["image"]

    # What the encoder publishes for the decode: the declared metadata fields.
    fields = PlaceholderMetadataResolver(encoder.info.ctx.model_config).fields_for(
        "image"
    )
    assert fields == {"image_num_patches"}, fields
    published = [
        item.get_data()["image_num_patches"] for item in enc_out["mm_kwargs"]["image"]
    ]

    decode = _processor(consumer=True)
    import torch

    meta = {"image_num_patches": torch.stack([torch.as_tensor(p) for p in published])}
    dec_out = decode(
        prompt,
        decode.info.data_parser.parse_mm_data({"image": meta}),
        hf_processor_mm_kwargs=TILED,
    )
    dec_ranges = dec_out["mm_placeholders"]["image"]

    assert [(r.offset, r.length) for r in dec_ranges] == [
        (r.offset, r.length) for r in enc_ranges
    ]
    assert enc_out["prompt_token_ids"] == dec_out["prompt_token_ids"]
    assert sum(r.get_num_embeds() for r in enc_ranges) == sum(
        int(p) * 256 for p in published
    )


@pytest.mark.parametrize("sizes", [[(2048, 2048)], [(1700, 2200), (500, 300)]])
def test_api_metadata_merges_like_the_server(sizes):
    """The chat API merges each image's metadata itself, from what arrives
    in the request: one 1-element tensor per image, exactly what the EPD
    proxy sends. Its field check must pass, or the decode answers 400."""
    import torch
    import transformers
    from PIL import Image
    from vllm.renderers.chat_utils import _merge_embeds

    encoder = _processor(consumer=False)
    images = [Image.new("RGB", wh, (120, 80, 40)) for wh in sizes]
    enc_out = encoder(
        _prompt(encoder, len(images)),
        encoder.info.data_parser.parse_mm_data({"image": images}),
        hf_processor_mm_kwargs=TILED,
    )
    published = [
        int(item.get_data()["image_num_patches"])
        for item in enc_out["mm_kwargs"]["image"]
    ]

    decode = _processor(consumer=True)
    items = [{"image_num_patches": torch.tensor([p])} for p in published]
    merged = _merge_embeds(items, decode)
    configs = decode._get_mm_fields_config(transformers.BatchFeature(merged), {})
    assert set(configs) >= {"image_embeds", "image_num_patches"}
    dec_out = decode(
        _prompt(decode, len(images)),
        decode.info.data_parser.parse_mm_data({"image": merged}),
        hf_processor_mm_kwargs=TILED,
    )
    assert [(r.offset, r.length) for r in dec_out["mm_placeholders"]["image"]] == [
        (r.offset, r.length) for r in enc_out["mm_placeholders"]["image"]
    ]
