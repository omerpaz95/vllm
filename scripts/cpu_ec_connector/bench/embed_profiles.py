# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""How many encoder outputs an image becomes, and how many bytes each takes in
the EC connector's CPU region, per model family.

The workload generators size the region from these numbers, so they must match
what the connector actually stores: `ECCPUConnector` allocates one block per
placeholder token, `block_size_bytes = hidden_dim * element_size`, with
`hidden_dim = out_hidden_size * (1 + len(deepstack_visual_indexes))` for a
deepstack model (vllm/distributed/ec_transfer/ec_connector/cpu/common.py).

Two families cover the models this benchmark runs:

  * `qwen`: Qwen2-VL's `smart_resize`. Sides round to multiples of
    `patch_size * merge_size` within a pixel budget; one placeholder per
    stride x stride block.
  * `internvl`: dynamic tiling. The image is cut into up to `max_tiles`
    448 px tiles on the closest aspect-ratio grid, plus one thumbnail tile
    when there is more than one; every tile yields `tokens_per_tile`
    placeholders (InternVL3.5 and Intern-S1: 256).

Embedding counts are a function of the processor's resize rule alone, so a
profile carries only the processor parameters and the per-embedding bytes.
"""

from __future__ import annotations

import math
from typing import NamedTuple

# Qwen2-VL's processor refuses anything more elongated than this.
MAX_ASPECT_RATIO = 200


class Profile(NamedTuple):
    """One model's resize rule and per-embedding footprint in the region."""

    family: str  # "qwen" or "internvl"
    bytes_per_embed: int
    # qwen: patch_size * merge_size, and the pixel budget.
    stride: int = 0
    min_pixels: int = 0
    max_pixels: int = 0
    # internvl: tile side, tile budget, placeholders per tile, thumbnail.
    tile: int = 0
    min_tiles: int = 1
    max_tiles: int = 0
    tokens_per_tile: int = 0
    thumbnail: bool = True


_BF16 = 2

# Qwen2.5-VL-7B: hidden 3584, patch 14 x merge 2, transformers' default budget.
QWEN2_5_VL_7B = Profile(
    "qwen", 3584 * _BF16, stride=28, min_pixels=3136, max_pixels=12845056
)
# Qwen3-VL-235B-A22B: out_hidden 4096 plus 3 deepstack layers (4 x 4096), patch
# 16 x merge 2. The budget is preprocessor_config.json's shortest_edge and
# longest_edge, which Qwen2VLImageProcessorFast reads as min and max pixels.
QWEN3_VL_235B = Profile(
    "qwen", 4 * 4096 * _BF16, stride=32, min_pixels=65536, max_pixels=16777216
)
# InternVL3.5-241B-A28B and Intern-S1: InternViT-6B into Qwen3-235B-A22B
# (hidden 4096), 448 px tiles, up to 12 plus a thumbnail, 256 tokens a tile
# after the 0.5 pixel shuffle.
INTERNVL3_5_241B = Profile(
    "internvl",
    4096 * _BF16,
    tile=448,
    max_tiles=12,
    tokens_per_tile=256,
)
# InternVL3.5-GPT-OSS-20B-A4B: InternViT-300M into GPT-OSS-20B (hidden 2880).
INTERNVL3_5_GPT_OSS_20B = INTERNVL3_5_241B._replace(bytes_per_embed=2880 * _BF16)

# Matched as a case-insensitive substring of the model name, first hit wins.
PROFILES: tuple[tuple[str, Profile], ...] = (
    ("qwen3-vl-235b", QWEN3_VL_235B),
    ("qwen2.5-vl-7b", QWEN2_5_VL_7B),
    ("internvl3_5-gpt-oss", INTERNVL3_5_GPT_OSS_20B),
    # The `-HF` release ships `crop_to_patches: false` (one tile per image);
    # the bench serves it with `--mm-processor-kwargs '{"crop_to_patches":
    # true}'`, which tiles exactly as the original-format release does.
    ("internvl3_5-241b", INTERNVL3_5_241B),
    ("intern-s1", INTERNVL3_5_241B),
)
DEFAULT_PROFILE = "Qwen/Qwen2.5-VL-7B-Instruct"


def profile_for(model: str) -> Profile:
    """The profile for `model`, matched by name.

    Raises:
        SystemExit: If no profile names this model. A guessed footprint is how
            a region ends up sized for a different model.
    """
    name = model.lower()
    for key, profile in PROFILES:
        if key in name:
            return profile
    known = ", ".join(key for key, _ in PROFILES)
    raise SystemExit(
        f"[workload] no embedding profile for {model!r} (known: {known}); add "
        "one to embed_profiles.py from the model's config.json and "
        "preprocessor_config.json"
    )


def smart_resize(
    height: int, width: int, factor: int, min_pixels: int, max_pixels: int
) -> tuple[int, int]:
    """Qwen2-VL's resize rule, mirroring transformers' image processor.

    Args:
        height: Source height in pixels.
        width: Source width in pixels.
        factor: Patch size times spatial merge size; sides become multiples
            of this.
        min_pixels: Smallest resized area, enlarged to reach it.
        max_pixels: Largest resized area, shrunk to fit it.

    Returns:
        The `(height, width)` the vision tower will actually see.

    Raises:
        ValueError: If the aspect ratio exceeds what the processor accepts.
    """
    if max(height, width) / max(1, min(height, width)) > MAX_ASPECT_RATIO:
        raise ValueError(
            f"aspect ratio {max(height, width) / max(1, min(height, width)):.1f} "
            f"exceeds {MAX_ASPECT_RATIO}"
        )
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return max(factor, h_bar), max(factor, w_bar)


def _tile_grid(width: int, height: int, profile: Profile) -> tuple[int, int]:
    """InternVL's closest-aspect-ratio tile grid, as vLLM's processor picks it
    (vllm/transformers_utils/processors/internvl.py)."""
    ratios = sorted(
        {
            (i, j)
            for n in range(profile.min_tiles, profile.max_tiles + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if profile.min_tiles <= i * j <= profile.max_tiles
        },
        key=lambda r: r[0] * r[1],
    )
    aspect = width / height
    area = width * height
    best, best_diff = (1, 1), float("inf")
    for ratio in ratios:
        diff = abs(aspect - ratio[0] / ratio[1])
        if diff < best_diff:
            best, best_diff = ratio, diff
        elif (
            diff == best_diff
            and area > 0.5 * profile.tile * profile.tile * ratio[0] * ratio[1]
        ):
            best = ratio
    return best


def embeds_for_image(width: int, height: int, profile: Profile) -> int:
    """Placeholder tokens one image occupies, which is what the connector
    allocates blocks for.

    Raises:
        ValueError: If the image's shape is one the processor rejects.
    """
    if profile.family == "qwen":
        h_bar, w_bar = smart_resize(
            height, width, profile.stride, profile.min_pixels, profile.max_pixels
        )
        return (h_bar // profile.stride) * (w_bar // profile.stride)
    cols, rows = _tile_grid(width, height, profile)
    tiles = cols * rows
    if profile.thumbnail and tiles != 1:
        tiles += 1
    return tiles * profile.tokens_per_tile
