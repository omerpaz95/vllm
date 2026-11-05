# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Size the GPU encoder cache from the modalities a server can be sent.

An EC consumer runs with `enable_mm_embeds` (`VllmConfig` turns it on for any
EC or KV consumer). With it on, `MultiModalBudget` (vLLM v0.32.0,
`vllm/multimodal/encoder_budget.py`) counts a modality whose
`--limit-mm-per-prompt` is 0 as "embedding-only" and includes its largest item
when it sizes the GPU encoder cache. The benchmark sets `{"video": 0}`, so no
request ever carries video, yet the cache is sized for one maximal video.

For InternVL that video is the whole context: the cache grows from
`max_num_batched_tokens` (8192, about two images) to 29440 tokens (about eight
images) on every connector server, while the baseline, which has no consumer
role, keeps 8192. A single-node `offload` arm then holds four times the
baseline's recently used images on the GPU, and serves repeats from there
rather than from the CPU region the arm exists to measure.

This patch removes modalities whose limit is 0 from the sizing, so a
connector server's cache matches the baseline's. Their embedding inputs are
still accepted; only the sizing changes. A modality a request can carry is
counted as before.

Loaded from `sitecustomize.py`; it touches `MultiModalBudget` only once a
server imports the module.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import sys

_TARGET = "vllm.multimodal.encoder_budget"
_MARKER = "_ec_bench_budget_patch"


def _patch(mod) -> None:
    if getattr(mod, _MARKER, False):
        return
    original = mod.compute_mm_encoder_budget
    budget_cls = mod.MultiModalBudget
    original_init = budget_cls.__init__

    def __init__(self, vllm_config, *args, **kwargs):
        mm_config = vllm_config.model_config.get_multimodal_config()

        def sized(scheduler_config, mm_max_toks_per_item):
            # Keep the modalities a request can carry; a limit of 0 means it
            # never does. With nothing left, size as vLLM would.
            kept = {
                m: toks
                for m, toks in mm_max_toks_per_item.items()
                if mm_config is None or mm_config.get_limit_per_prompt(m) != 0
            }
            return original(scheduler_config, kept or mm_max_toks_per_item)

        mod.compute_mm_encoder_budget = sized
        try:
            original_init(self, vllm_config, *args, **kwargs)
        finally:
            mod.compute_mm_encoder_budget = original

    budget_cls.__init__ = __init__
    setattr(mod, _MARKER, True)


class _PatchOnImport(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Runs `_patch` on the target module right after it executes."""

    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        self._inner = spec.loader
        spec.loader = self
        return spec

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        _patch(module)


def install() -> None:
    if _TARGET in sys.modules:
        _patch(sys.modules[_TARGET])
    elif not any(isinstance(f, _PatchOnImport) for f in sys.meta_path):
        sys.meta_path.insert(0, _PatchOnImport())
