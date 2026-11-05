# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check that an embedded workload sends the client exactly what the original
does: `vllm bench serve`'s own CustomImageDataset samples both, with
`--custom-ensure-client-side-data`, and the requests must be identical.

Needs a vLLM install (the client's dataset code); no GPU.

    python -m pytest test_embed_workload.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

pytest.importorskip("vllm.benchmarks.datasets.datasets")


def _workload(tmp_path: Path) -> Path:
    from PIL import Image

    pool = tmp_path / "pool"
    pool.mkdir()
    paths = []
    for i, (size, mode) in enumerate(
        [((64, 48), "RGB"), ((33, 17), "L"), ((40, 40), "RGBA")]
    ):
        path = pool / f"img{i}.jpg" if mode != "RGBA" else pool / f"img{i}.png"
        Image.new(mode, size, color=(i * 70) % 256 if mode == "L" else None).save(path)
        paths.append(str(path))
    lines = [
        {
            "content": [
                {"type": "text", "text": "Compare "},
                {"type": "image_url", "image_url": {"url": paths[0]}},
                {"type": "text", "text": " and "},
                {"type": "image_url", "image_url": {"url": paths[1]}},
            ]
        },
        {"content": [{"type": "image", "image": paths[2]}]},
        {"content": [{"type": "image_url", "image_url": paths[0]}]},
        {"content": [{"type": "text", "text": "text only"}]},
    ]
    (tmp_path / "workload.jsonl").write_text(
        "".join(json.dumps(line) + "\n" for line in lines)
    )
    return tmp_path


def _sample(path: Path) -> list:
    from vllm.benchmarks.datasets.datasets import CustomImageDataset

    dataset = CustomImageDataset(dataset_path=str(path), disable_shuffle=True)
    return [
        r.prompt
        for r in dataset.sample(
            tokenizer=None,
            num_requests=0,
            output_len=8,
            ensure_client_side_data=True,
        )
    ]


def test_embedded_copy_sends_identical_requests(tmp_path: Path) -> None:
    from embed_workload import embed

    wl = _workload(tmp_path)
    refs, distinct = embed(wl / "workload.jsonl", wl / "workload-embedded.jsonl", 2)
    assert (refs, distinct) == (4, 3)
    original = _sample(wl / "workload.jsonl")
    embedded = _sample(wl / "workload-embedded.jsonl")
    assert embedded == original
    assert "data:image/png;base64," in json.dumps(embedded)


def test_main_keeps_a_fresh_copy(tmp_path: Path, monkeypatch, capsys) -> None:
    import embed_workload

    wl = _workload(tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["embed_workload.py", "--workload-dir", str(wl), "--workers", "1"]
    )
    embed_workload.main()
    copy = wl / "workload-embedded.jsonl"
    first = copy.stat().st_mtime_ns
    embed_workload.main()
    assert copy.stat().st_mtime_ns == first
    assert "up to date" in capsys.readouterr().out
    # An edited source makes the copy stale.
    source = wl / "workload.jsonl"
    os.utime(source, ns=(first + 10**9, first + 10**9))
    embed_workload.main()
    second = copy.stat().st_mtime_ns
    assert second != first
    # So does an image rewritten in place, which changes no JSONL.
    image = next((wl / "pool").iterdir())
    os.utime(image, ns=(second + 10**9, second + 10**9))
    embed_workload.main()
    assert copy.stat().st_mtime_ns != second
