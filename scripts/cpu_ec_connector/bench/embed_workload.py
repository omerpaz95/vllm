#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Write a copy of a workload with every image inlined as a data URL.

`vllm bench serve --custom-ensure-client-side-data` loads every referenced
image and re-encodes it (PNG, base64) before the first request of each load
point, single-threaded: minutes per point on a document workload. This does
that encoding once, per distinct image and in parallel, through the same
`process_image` the client calls, so the requests carry the same bytes. The
client keeps a `data:image/` URL as it is.

    python embed_workload.py --workload-dir /bench/wl-qwen3vl-docvqa

writes `workload-embedded.jsonl` beside `workload.jsonl` (and the same for
`warmup.jsonl` and each per-point `workload-pN.jsonl`). A copy newer than its
source is kept.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import regex as re

SOURCES = ("workload.jsonl", "warmup.jsonl")
_IMAGE_TYPES = ("image", "image_url")
# hf_workload.py --points writes one of these per load point.
_POINT_FILE_RE = re.compile(r"workload-p(\d+)\.jsonl")


def point_file_name(point: int) -> str:
    """The requests load point `point` replays (hf_workload.py --points)."""
    return f"workload-p{point}.jsonl"


def embedded_name(source: str) -> str:
    stem, _, ext = source.rpartition(".")
    return f"{stem}-embedded.{ext}"


def _image_ref(part: dict[str, Any]) -> str | None:
    if part.get("type") == "image":
        return part.get("image")
    if part.get("type") == "image_url":
        url = part.get("image_url")
        return url.get("url") if isinstance(url, dict) else url
    return None


def _set_image_ref(part: dict[str, Any], data_url: str) -> None:
    if part["type"] == "image":
        part["image"] = data_url
    elif isinstance(part["image_url"], dict):
        part["image_url"]["url"] = data_url
    else:
        part["image_url"] = data_url


def _encode(ref: str) -> str:
    from vllm.benchmarks.datasets.datasets import process_image

    return process_image(ref, ensure_client_side_data=True)["image_url"]["url"]


def embed(source: Path, dest: Path, workers: int) -> tuple[int, int]:
    """(image references, distinct images) written from `source` to `dest`."""
    records = [
        json.loads(line) for line in source.read_text().splitlines() if line.strip()
    ]
    refs = [
        ref
        for record in records
        for part in record.get("content", [])
        if (ref := _image_ref(part)) and not ref.startswith("data:image/")
    ]
    distinct = sorted(set(refs))
    # Spawned, not forked: a caller that has imported vLLM holds threads.
    spawn = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=spawn) as pool:
        encoded = dict(zip(distinct, pool.map(_encode, distinct, chunksize=4)))
    # A unique name, so two runs embedding the same workload at once never
    # interleave; the rename makes whichever finishes last the copy.
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
    with open(tmp, "w") as out:
        for record in records:
            for part in record.get("content", []):
                ref = _image_ref(part)
                if ref in encoded:
                    _set_image_ref(part, encoded[ref])
            out.write(json.dumps(record) + "\n")
    tmp.replace(dest)
    return len(refs), len(distinct)


def is_fresh(dest: Path, source: Path, pool: Path) -> bool:
    """`dest` is newer than `source` and than every image in `pool`: an
    image rewritten in place changes no JSONL, only the pool."""
    if not dest.exists():
        return False
    built = dest.stat().st_mtime
    if built <= source.stat().st_mtime:
        return False
    return not pool.is_dir() or all(
        f.stat().st_mtime < built for f in pool.iterdir() if f.is_file()
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--workload-dir", type=Path, required=True)
    p.add_argument(
        "--workers",
        type=int,
        default=min(16, len(os.sched_getaffinity(0))),
        help="encoding processes",
    )
    p.add_argument("--force", action="store_true", help="rebuild fresh copies too")
    args = p.parse_args()
    points = sorted(
        (
            f.name
            for f in args.workload_dir.glob("workload-p*.jsonl")
            if _POINT_FILE_RE.fullmatch(f.name)
        ),
        key=lambda n: int(_POINT_FILE_RE.fullmatch(n).group(1)),
    )
    for name in (*SOURCES, *points):
        source = args.workload_dir / name
        dest = args.workload_dir / embedded_name(name)
        if not source.exists():
            continue
        if not args.force and is_fresh(dest, source, args.workload_dir / "pool"):
            print(f"[embed] {dest} is up to date")
            continue
        refs, distinct = embed(source, dest, args.workers)
        print(
            f"[embed] {dest}: {refs} image refs ({distinct} distinct), "
            f"{dest.stat().st_size / 1e9:.2f} GB"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
