#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Convert a Hugging Face multimodal dataset into a `gen_workload.py` directory.

Emits the same layout `gen_workload.py` does -- `pool/`, `workload.jsonl`,
`manifest.json` -- so `run_bench.py` consumes the result unchanged. What
changes is where the images and the questions come from: real dataset rows
instead of synthesized photos and a fixed prompt.

The two load-bearing properties of the synthetic generator are preserved:

  * A unique text nonce comes BEFORE the images in every request. Without it
    the KV prefix cache would serve a repeated image outright and the encoder
    output would never be wanted, so there would be nothing to reload.
  * `manifest.json["expected"]` states what the sequence should cost, which is
    what makes the server's own accounting checkable.

Reuse -- the property the EC connector exists to exploit -- is measured, not
assumed. Images are deduplicated by the sha256 of their decoded pixels, so a
photo two rows share becomes one pool file and two references, and the
manifest's `max_hit_rate` reports exactly how much reuse the dataset actually
carries. A dataset whose rows never share an image lands at 0% and is a poor
choice for the offload arms, whatever else it is good for.

Embedding counts follow Qwen2-VL's `smart_resize`: sides are rounded to
multiples of 28 and the pixel budget clamped, then one embedding is charged
per 28x28 block. That is the quantity the connector stores, so the region
sizes in the manifest are the ones the server will actually want.

Examples:

    python hf_workload.py --dataset MUIRBENCH/MUIRBENCH --split test \
        --out-dir /data/wl-muir --max-samples 600

    python hf_workload.py --dataset lmarena-ai/VisionArena-Chat \
        --split train --out-dir /data/wl-va --expand-turns
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gen_workload import (  # noqa: E402
    DEFAULT_ELEMENT_SIZE,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_MERGE_STRIDE,
    PoolImage,
    _nonce_text,
    build_manifest,
    self_check,
)

# transformers' Qwen2VLImageProcessor defaults.
DEFAULT_MIN_PIXELS = 3136
DEFAULT_MAX_PIXELS = 12845056
MAX_ASPECT_RATIO = 200

DEFAULT_WARMUP_REQUESTS = 8
# Only used when the dataset cannot spare rows for a warmup that shares no
# image with the workload.
_SYNTHETIC_WARMUP_SIZE = (896, 896)

# What a dataset's columns are called, when it is one we know. Anything else is
# detected from the first row, or named with --image-column/--question-column.
_KNOWN_DATASETS: dict[str, dict[str, Any]] = {
    "MUIRBENCH/MUIRBENCH": {
        "split": "test",
        "image_columns": ("image_list", "images", "image"),
        "question_columns": ("question",),
        "options_column": "options",
    },
    "lmarena-ai/VisionArena-Chat": {
        "split": "train",
        "image_columns": ("images",),
        "conversation_column": "conversation",
    },
    "lmarena-ai/vision-arena-bench-v0.1": {
        "split": "train",
        "image_columns": ("images",),
        "conversation_column": "turns",
    },
    "Lin-Chen/MMStar": {
        "split": "val",
        "image_columns": ("image",),
        "question_columns": ("question",),
    },
    "lmms-lab/DocVQA": {
        "split": "validation",
        "image_columns": ("image",),
        "question_columns": ("question",),
    },
}

_QUESTION_COLUMNS = ("question", "prompt", "query", "instruction", "text")


class Sample(NamedTuple):
    """One dataset row: its images and the request text(s) they carry."""

    images: list[Any]
    texts: list[str]


def smart_resize(
    height: int,
    width: int,
    factor: int = DEFAULT_MERGE_STRIDE,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    max_pixels: int = DEFAULT_MAX_PIXELS,
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


def embeds_for_image(
    width: int, height: int, stride: int, min_pixels: int, max_pixels: int
) -> int:
    """Encoder outputs one image occupies after the spatial merge."""
    h_bar, w_bar = smart_resize(height, width, stride, min_pixels, max_pixels)
    return (h_bar // stride) * (w_bar // stride)


# ---------------------------------------------------------------------------
# Dataset rows -> images and question text
# ---------------------------------------------------------------------------


def _as_pil(value: Any) -> Any:
    from PIL import Image

    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict) and value.get("bytes"):
        return Image.open(io.BytesIO(value["bytes"]))
    if isinstance(value, str) and value and Path(value).is_file():
        return Image.open(value)
    return None


def _images_of(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        found = [_as_pil(item) for item in value]
        return [img for img in found if img is not None]
    single = _as_pil(value)
    return [single] if single is not None else []


def detect_image_column(record: dict, preferred: Sequence[str]) -> str:
    """Name of the column holding this row's image(s)."""
    for name in preferred:
        if name in record and _images_of(record[name]):
            return name
    for name, value in record.items():
        if _images_of(value):
            return name
    raise SystemExit(
        f"[hf] no image column found; row has {sorted(record)}. Pass --image-column."
    )


def detect_question_column(record: dict, preferred: Sequence[str]) -> str | None:
    for name in preferred:
        if isinstance(record.get(name), str) and record[name].strip():
            return name
    return None


def _conversation_texts(value: Any) -> list[str]:
    """User turns of a VisionArena-style conversation column."""
    texts: list[str] = []
    if not isinstance(value, (list, tuple)):
        return texts
    for turn in value:
        message = turn[0] if isinstance(turn, (list, tuple)) and turn else turn
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            texts.append(message["content"])
    return texts


def _options_text(value: Any) -> str:
    if isinstance(value, (list, tuple)) and value:
        letters = "ABCDEFGHIJKLMNOP"
        joined = " ".join(
            f"{letters[i]}. {opt}" for i, opt in enumerate(value) if i < len(letters)
        )
        return f" Options: {joined}"
    if isinstance(value, str) and value.strip():
        return f" Options: {value.strip()}"
    return ""


def parse_sample(record: dict, cfg: dict[str, Any], expand_turns: bool) -> Sample:
    """Turn one dataset row into its images and one text per request."""
    images = _images_of(record.get(cfg["image_column"]))
    conversation = cfg.get("conversation_column")
    if conversation and conversation in record:
        texts = _conversation_texts(record[conversation])
        texts = texts if expand_turns else texts[:1]
    else:
        column = cfg.get("question_column")
        question = record.get(column) if column else None
        texts = [str(question).strip()] if isinstance(question, str) else []
        options = cfg.get("options_column")
        if texts and options and options in record:
            texts = [texts[0] + _options_text(record[options])]
    return Sample(images, [t for t in texts if t.strip()])


def load_records(args: argparse.Namespace) -> list[dict]:
    """Pull up to `--max-samples` rows, streaming so nothing extra downloads."""
    from datasets import load_dataset

    data = load_dataset(
        args.dataset,
        name=args.subset,
        split=args.split,
        streaming=not args.no_stream,
    )
    records: list[dict] = []
    for record in data:
        records.append(dict(record))
        if args.max_samples and len(records) >= args.max_samples:
            break
    if not records:
        raise SystemExit(f"[hf] {args.dataset} split {args.split!r} yielded no rows")
    return records


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------


class Pool:
    """Content-addressed image pool: one file per distinct image."""

    def __init__(self, out_dir: Path, args: argparse.Namespace, bytes_per_embed: int):
        self.dir = out_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.args = args
        self.bytes_per_embed = bytes_per_embed
        self.images: list[PoolImage] = []
        self.by_hash: dict[str, int] = {}
        self.names: set[str] = set()
        self.rejected = 0

    def add(self, image: Any, skip_hashes: set[str] | None = None) -> int | None:
        """Index of this image in the pool, saving it on first sight."""
        rgb = image.convert("RGB")
        digest = hashlib.sha256(rgb.tobytes()).hexdigest()
        if skip_hashes and digest in skip_hashes:
            return None
        if digest in self.by_hash:
            return self.by_hash[digest]
        width, height = rgb.size
        try:
            embeds = embeds_for_image(
                width,
                height,
                self.args.merge_stride,
                self.args.min_pixels,
                self.args.max_pixels,
            )
        except ValueError as exc:
            print(f"[hf] skipping {width}x{height} image: {exc}", file=sys.stderr)
            self.rejected += 1
            return None
        name = f"{width}x{height}_{digest[:8]}.jpg"
        length = 8
        while name in self.names:
            length += 4
            name = f"{width}x{height}_{digest[:length]}.jpg"
        path = self.dir / name
        rgb.save(path, format="JPEG", quality=self.args.jpeg_quality)
        self.names.add(name)
        self.by_hash[digest] = len(self.images)
        self.images.append(
            PoolImage(path, width, height, embeds, embeds * self.bytes_per_embed)
        )
        return self.by_hash[digest]


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


def _record_for(
    index: int,
    question: str,
    chosen: Sequence[int],
    pool: Pool,
    rng: random.Random,
    prefix_tokens: int,
) -> dict:
    nonce = f"[req {index} n{rng.getrandbits(48):012x}] " + _nonce_text(
        rng, prefix_tokens
    )
    content: list[dict] = [{"type": "text", "text": nonce}]
    for idx in chosen:
        content.append(
            {"type": "image_url", "image_url": {"url": str(pool.images[idx].path)}}
        )
    content.append({"type": "text", "text": question})
    return {"content": content}


def build_requests(
    samples: Iterable[Sample],
    pool: Pool,
    rng: random.Random,
    *,
    prefix_tokens: int,
    limit: int,
    skip_hashes: set[str] | None = None,
    max_images: int = 0,
    max_embeds: int = 0,
    oversized: list[int] | None = None,
) -> tuple[list[dict], list[list[int]]]:
    """Return `(jsonl_records, per_request_image_indices)`.

    A row whose images are all rejected or all held back is dropped: a
    text-only request would count as a request that could never reuse. A
    row over `max_images` or `max_embeds` is dropped too, and counted in
    `oversized`: such a request outgrows the context window on every arm
    alike, and one failure stops the benchmark's completion gate.
    """
    records: list[dict] = []
    per_request: list[list[int]] = []
    for sample in samples:
        if limit and len(records) >= limit:
            break
        if max_images and len(sample.images) > max_images:
            if oversized is not None:
                oversized.append(len(sample.images))
            continue
        chosen = [
            index
            for index in (pool.add(img, skip_hashes) for img in sample.images)
            if index is not None
        ]
        if not chosen:
            continue
        embeds = sum(pool.images[i].embeds for i in chosen)
        if max_embeds and embeds > max_embeds:
            if oversized is not None:
                oversized.append(embeds)
            continue
        for text in sample.texts:
            if limit and len(records) >= limit:
                break
            records.append(
                _record_for(len(records), text, chosen, pool, rng, prefix_tokens)
            )
            per_request.append(list(chosen))
    return records, per_request


def synthetic_images(count: int, size: tuple[int, int]) -> list[Any]:
    """Filler images for a warmup the dataset was too small to supply."""
    from PIL import Image, ImageChops

    width, height = size
    out = []
    for index in range(count):
        bands = []
        for channel in range(3):
            gradient = Image.linear_gradient("L").rotate(90 * (channel + index) % 360)
            bands.append(
                ImageChops.add(
                    gradient.resize((width, height)),
                    Image.effect_noise((width, height), 48),
                    scale=1.0,
                    offset=-64,
                )
            )
        out.append(Image.merge("RGB", bands))
    return out


def write_jsonl(path: Path, records: Sequence[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def resolve_columns(record: dict, args: argparse.Namespace) -> dict[str, Any]:
    cfg = dict(_KNOWN_DATASETS.get(args.dataset, {}))
    cfg["image_column"] = args.image_column or detect_image_column(
        record, cfg.get("image_columns", ())
    )
    if args.question_column:
        cfg["question_column"] = args.question_column
        cfg.pop("conversation_column", None)
    elif not cfg.get("conversation_column"):
        cfg["question_column"] = detect_question_column(
            record, tuple(cfg.get("question_columns", ())) + _QUESTION_COLUMNS
        )
        if cfg["question_column"] is None and not cfg.get("conversation_column"):
            raise SystemExit(
                f"[hf] no question column found; row has {sorted(record)}. "
                "Pass --question-column."
            )
    return cfg


def convert(records: list[dict], args: argparse.Namespace) -> int:
    """Build the workload directory from already-loaded dataset rows."""
    rng = random.Random(args.seed)
    bytes_per_embed = args.hidden_dim * args.element_size
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = resolve_columns(records[0], args)
    print(
        f"[hf] {args.dataset} ({args.split}): {len(records)} rows, images from "
        f"{cfg['image_column']!r}, text from "
        f"{cfg.get('conversation_column') or cfg.get('question_column')!r}"
    )

    holdout_size = min(args.warmup_requests, len(records) // 4)
    holdout = records[len(records) - holdout_size :] if holdout_size else []
    kept = records[: len(records) - holdout_size]
    if args.order == "shuffle":
        rng.shuffle(kept)

    samples = [parse_sample(r, cfg, args.expand_turns) for r in kept]
    samples = [s for s in samples if s.images and s.texts]
    if not samples:
        raise SystemExit("[hf] no row carried both an image and a question")

    pool = Pool(args.out_dir / "pool", args, bytes_per_embed)
    oversized: list[int] = []
    workload, per_request = build_requests(
        samples,
        pool,
        rng,
        prefix_tokens=args.prefix_tokens,
        limit=args.num_requests,
        max_images=args.max_images_per_request,
        max_embeds=args.max_embeds_per_request,
        oversized=oversized,
    )
    if oversized:
        print(
            f"[hf] dropped {len(oversized)} rows over --max-images-per-request/"
            f"--max-embeds-per-request (largest: {max(oversized)})"
        )
    if not workload:
        raise SystemExit("[hf] every row was dropped; nothing to replay")
    write_jsonl(args.out_dir / "workload.jsonl", workload)

    warm_pool = Pool(args.out_dir / "warmup_pool", args, bytes_per_embed)
    warm_samples = [parse_sample(r, cfg, args.expand_turns) for r in holdout]
    warm_samples = [s for s in warm_samples if s.images and s.texts]
    warmup, _ = build_requests(
        warm_samples,
        warm_pool,
        rng,
        prefix_tokens=args.prefix_tokens,
        limit=args.warmup_requests,
        skip_hashes=set(pool.by_hash),
    )
    if len(warmup) < args.warmup_requests:
        missing = args.warmup_requests - len(warmup)
        print(
            f"[hf] warning: the dataset could spare only {len(warmup)} rows whose "
            f"images the workload does not also use; filling {missing} warmup "
            "requests with synthetic images",
            file=sys.stderr,
        )
        filler = [
            Sample([img], ["Describe this image in one short sentence."])
            for img in synthetic_images(missing, _SYNTHETIC_WARMUP_SIZE)
        ]
        extra, _ = build_requests(
            filler, warm_pool, rng, prefix_tokens=args.prefix_tokens, limit=missing
        )
        warmup.extend(extra)
    write_jsonl(args.out_dir / "warmup.jsonl", warmup)

    args.images_per_request = max(len(chosen) for chosen in per_request)
    manifest = build_manifest(
        pool.images, per_request, bytes_per_embed=bytes_per_embed, args=args
    )
    manifest["source"] = {
        "dataset": args.dataset,
        "split": args.split,
        "samples": len(samples),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    seq, exp = manifest["sequence"], manifest["expected"]
    megapixels = sum(p.width * p.height for p in pool.images) / 1e6
    disk_mb = sum(p.path.stat().st_size for p in pool.images) / 1e6
    print(
        f"[hf] pool: {len(pool.images)} distinct images, {pool.rejected} rejected, "
        f"{disk_mb:.1f} MB on disk ({disk_mb / max(megapixels, 1e-9):.2f} MB/MP)"
    )
    print(
        f"[hf] {args.out_dir / 'workload.jsonl'}: {seq['requests']} requests, "
        f"{seq['image_references']} image refs over "
        f"{seq['distinct_images_used']} distinct images, images per request "
        f"{seq['image_count_histogram']}"
    )
    print(
        f"[hf] working set {exp['working_set_bytes'] / 1024**3:.2f} GiB, "
        f"max hit rate {exp['max_hit_rate'] * 100:.1f}%, max embeds in one "
        f"request {exp['max_embeds_per_request']}, suggested ec_cpu_bytes "
        f"{exp['suggested_ec_cpu_bytes']}"
    )
    if not exp["reuses"]:
        print(
            "[hf] warning: no image is referenced twice, so the connector has "
            "nothing to reload and the offload arms cannot differ from baseline",
            file=sys.stderr,
        )
    print(f"[hf] warmup: {len(warmup)} requests in {args.out_dir / 'warmup.jsonl'}")

    if args.self_check:
        self_check(args.out_dir / "workload.jsonl", len(workload))
        self_check(args.out_dir / "warmup.jsonl", len(warmup))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", default="MUIRBENCH/MUIRBENCH")
    p.add_argument(
        "--subset", default=None, help="HF config name, if the dataset has one"
    )
    p.add_argument("--split", default=None, help="default: the dataset's known split")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument(
        "--num-requests", type=int, default=0, help="0 replays every usable row"
    )
    p.add_argument("--max-samples", type=int, default=0, help="0 pulls the whole split")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--jpeg-quality", type=int, default=85)
    p.add_argument("--image-column", default=None)
    p.add_argument("--question-column", default=None)
    p.add_argument(
        "--expand-turns",
        action="store_true",
        help="one request per user turn of a conversation dataset, all sharing "
        "the row's images -- reuse a single-turn conversion throws away",
    )
    p.add_argument("--order", choices=("dataset", "shuffle"), default="dataset")
    p.add_argument("--prefix-tokens", type=int, default=32)
    p.add_argument("--warmup-requests", type=int, default=DEFAULT_WARMUP_REQUESTS)
    p.add_argument("--no-stream", action="store_true")
    p.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    p.add_argument("--element-size", type=int, default=DEFAULT_ELEMENT_SIZE)
    p.add_argument("--merge-stride", type=int, default=DEFAULT_MERGE_STRIDE)
    p.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS)
    p.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    p.add_argument(
        "--max-images-per-request",
        type=int,
        default=0,
        help="drop rows with more images than this; 0 = no cap",
    )
    p.add_argument(
        "--max-embeds-per-request",
        type=int,
        default=0,
        help="drop rows whose images together exceed this many embeddings, so "
        "every request fits the decode instance's --max-model-len (32768 by "
        "default: 28000 leaves room for the text); 0 = no cap",
    )
    p.add_argument("--self-check", action="store_true")
    args = p.parse_args(argv)
    if args.split is None:
        args.split = _KNOWN_DATASETS.get(args.dataset, {}).get("split", "train")
    return args


def main() -> int:
    args = parse_args()
    return convert(load_records(args), args)


if __name__ == "__main__":
    sys.exit(main())
