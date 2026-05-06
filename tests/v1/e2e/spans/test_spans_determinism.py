# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-mode output equivalence for spans / Legolink.

Migrated from examples/offline_inference/spans/basic_spans_determinism.py:
the example's diagnostics (top-K logprob equivalence, multi-seed drift,
gap-policy replay) become the assertions of these tests.

The contract:
  * On a prompt with no PIC chunk, FR == SPANS == LL-16 == LL-FULL
    (text + top-K logprobs bit-identical at temp=0).
  * On a prompt whose first 16 tokens are a PIC chunk and no preload
    (cache empty), the four modes still agree because no cache hit means
    the gap policy never fires.
  * LL-FULL with prefix-caching ON, run twice: run #2 hits the cache,
    gap policy with gap_length >> prompt forces a full recompute, output
    must equal a clean FR reference.
"""
import pytest

from vllm import SamplingParams

from .conftest import (
    BLOCK_SIZE,
    build_llm,
    cleanup,
    extract_step0_topk,
)

pytestmark = pytest.mark.spans

SEED = 42
MAX_TOKENS = 16
LOGPROBS_TOPK = 10
PLAIN_PROMPT = "Hello world! Please write a short greeting in one sentence."

ALL_MODES = ("FR", "SPANS", "LL-16", "LL-FULL")


def _greedy_params(extra_args: dict | None = None) -> SamplingParams:
    return SamplingParams.from_optional(
        seed=SEED,
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        logprobs=LOGPROBS_TOPK,
        extra_args=extra_args,
    )


def _run(llm, prompt: str, extra_args: dict | None = None):
    res = llm.generate(prompt, sampling_params=_greedy_params(extra_args), use_tqdm=False)
    out = res[0].outputs[0]
    return out.text, extract_step0_topk(out, LOGPROBS_TOPK)


def _reference_results(model: str, prompt: str, monkeypatch, extra_args=None):
    """Run the prompt under each mode and return {mode: (text, top10)}.

    Each mode gets its own fresh LLM (modes change global env + LLM kwargs)."""
    out: dict[str, tuple[str, list]] = {}
    for mode in ALL_MODES:
        llm = build_llm(model, mode, monkeypatch)
        try:
            out[mode] = _run(llm, prompt, extra_args)
        finally:
            cleanup(llm)
    return out


def test_no_pic_all_modes_match(model, monkeypatch):
    """Plain prompt, no PIC: FR == SPANS == LL-16 == LL-FULL."""
    results = _reference_results(model, PLAIN_PROMPT, monkeypatch, extra_args=None)
    fr_text, fr_top = results["FR"]
    for mode in ALL_MODES:
        text, top = results[mode]
        assert text == fr_text, f"{mode} text drifted vs FR"
        assert top == fr_top, f"{mode} top-{LOGPROBS_TOPK} logprobs drifted vs FR"


def test_pic_at_start_all_modes_match(model, monkeypatch):
    """PIC chunk at position 0 (<= 16 tokens), no preload → no cache hit, so
    even Legolink modes match FR.
    """
    extra_args = {"span_starts": [0]}
    results = _reference_results(model, PLAIN_PROMPT, monkeypatch, extra_args=extra_args)
    fr_text, fr_top = results["FR"]
    for mode in ALL_MODES:
        text, top = results[mode]
        assert text == fr_text, f"{mode} text drifted vs FR with PIC at start"
        assert top == fr_top, f"{mode} top-{LOGPROBS_TOPK} drifted vs FR with PIC at start"


def test_legolink_gap_huge_equals_full_recompute(model, monkeypatch):
    """LL-FULL with prefix caching: cold prefill populates the cache, second
    run hits it and (because gap_length >> prompt) re-prefills everything.
    Output must equal clean FR.
    """
    # Reference: clean FR, same prompt, same sampling params.
    fr_llm = build_llm(model, "FR", monkeypatch)
    try:
        ref_text, ref_top = _run(fr_llm, PLAIN_PROMPT)
    finally:
        cleanup(fr_llm)

    ll_llm = build_llm(model, "LL-FULL", monkeypatch)
    try:
        # Run #1: cold prefill, populates cache.
        _run(ll_llm, PLAIN_PROMPT)
        # Run #2: every block is now a cache hit → gap policy forces full
        # recompute over the cached prefix.
        replay_text, replay_top = _run(ll_llm, PLAIN_PROMPT)
    finally:
        cleanup(ll_llm)

    assert replay_text == ref_text, "LL-FULL replay diverged from FR"
    assert replay_top == ref_top, (
        f"LL-FULL replay top-{LOGPROBS_TOPK} logprobs diverged from FR"
    )


def test_block_size_constant_matches_conftest():
    """Defensive: tests assume block_size == 16. If conftest changes, tests
    that rely on PIC alignment must be revisited."""
    assert BLOCK_SIZE == 16
