# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check the encode fan-out workload: `hf_workload.py --no-reuse --points N`
and `run_bench.py --disjoint-points`.

A fan-out point measures something only if every image in it is new to every
cache, so each is encoded and sent to the decode within that point. These
tests pin that property on a DocVQA-shaped dataset (several questions per
page, one page repeated under a different key) and the gate that refuses a
point where some encodings came from a cache instead.

    python -m pytest test_fanout_workload.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

pytest.importorskip("PIL")

MODEL = "OpenGVLab/InternVL3_5-241B-A28B-HF"


def _page(seed: int):
    from PIL import Image

    rng = random.Random(seed)
    colour = tuple(rng.randrange(256) for _ in range(3))
    return Image.new("RGB", (900 + seed, 1200), colour)


def _rows(pages: int, questions: int, repeat: tuple[int, int]) -> list[dict]:
    """`questions` rows per page; page `repeat[0]` carries page `repeat[1]`'s
    pixels under its own key, the way a dataset can hold one scan twice."""
    rows = []
    for page in range(pages):
        pixels = repeat[1] if page == repeat[0] else page
        for q in range(questions):
            rows.append(
                {
                    "image": _page(pixels),
                    "question": f"question {q} about page {page}",
                    "doc": f"d{page}",
                    "page": "1",
                }
            )
    return rows


def _build(tmp_path: Path, rows: list[dict], points: int) -> Path:
    import hf_workload

    args = hf_workload.parse_args(
        [
            "--dataset",
            "lmms-lab/DocVQA",
            "--out-dir",
            str(tmp_path),
            "--model",
            MODEL,
            "--no-reuse",
            "--points",
            str(points),
            "--warmup-requests",
            "4",
        ]
    )
    # load_records' --distinct-key filter, without streaming a dataset.
    seen, kept = set(), []
    for row in rows:
        key = (row["doc"], row["page"])
        if key not in seen:
            seen.add(key)
            kept.append(row)
    hf_workload.convert(kept, args)
    return tmp_path


def _refs(path: Path) -> list[str]:
    return [
        part["image_url"]["url"]
        for line in path.read_text().splitlines()
        if line.strip()
        for part in json.loads(line)["content"]
        if part.get("type") == "image_url"
    ]


def test_no_image_repeats_within_or_across_points(tmp_path: Path) -> None:
    out = _build(tmp_path, _rows(pages=40, questions=3, repeat=(30, 7)), points=3)
    blocks = [_refs(out / f"workload-p{p}.jsonl") for p in range(3)]
    assert all(blocks) and len({len(b) for b in blocks}) == 1
    every = [ref for block in blocks for ref in block]
    assert len(every) == len(set(every)), "an image appears twice"
    # The warmup must not seed a cache any point then hits.
    assert not set(_refs(out / "warmup.jsonl")) & set(every)
    expected = json.loads((out / "manifest.json").read_text())["expected"]
    assert expected["reuses"] == 0


def test_too_few_requests_for_the_points_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        _build(tmp_path, _rows(pages=6, questions=1, repeat=(0, 0)), points=8)


def _point_args(tmp_path: Path, points: int) -> argparse.Namespace:
    return argparse.Namespace(
        workload_dir=str(tmp_path),
        load_points=[(str(r), 0) for r in range(1, points + 1)],
    )


def test_disjoint_points_reads_each_block(tmp_path: Path) -> None:
    import run_bench

    out = _build(tmp_path, _rows(pages=40, questions=3, repeat=(30, 7)), points=3)
    args = _point_args(out, 3)
    run_bench.prepare_point_files(run_bench.Target(None), args)
    assert args.point_files == [f"workload-p{p}.jsonl" for p in range(3)]
    assert args.point_prompts == [len(_refs(out / f)) for f in args.point_files]
    assert args.point_image_refs == args.point_prompts


def test_disjoint_points_needs_a_file_per_point(tmp_path: Path) -> None:
    import run_bench

    out = _build(tmp_path, _rows(pages=40, questions=3, repeat=(30, 7)), points=3)
    with pytest.raises(SystemExit, match="workload-p3.jsonl"):
        run_bench.prepare_point_files(run_bench.Target(None), _point_args(out, 4))


def test_a_point_served_from_a_cache_is_refused() -> None:
    import run_bench

    run_bench.check_all_transferred("cpu-grid", "rate=1", {"nixl_reads": 400}, 400)
    with pytest.raises(run_bench.ServerMismatchError, match="from a cache"):
        run_bench.check_all_transferred("cpu-grid", "rate=3", {"nixl_reads": 12}, 400)


def test_nixl_reads_are_counted_per_item() -> None:
    from ec_log_stats import summarize

    line = (
        "(Worker_TP0 pid=1) DEBUG 11-01 01:40:00 [cpu/session.py:168] "
        "EC consumer: NIXL READ completed mm_hash={}"
    )
    # The producer logs its own completion line, which is not a read.
    text = "\n".join(
        [
            line.format("a"),
            line.format("b"),
            "EC producer: NIXL READ completed mm_hash=a key=k",
        ]
    )
    assert summarize(text)["nixl_reads"] == 2
