# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Isolation test: does VLLM_V1_SPANS_ENABLED=True perturb generation on a plain
prompt that contains no span markers and no preloaded KV blocks?

Two modes (recompute vs spans_no_kv), same seed, two temperatures, 3 reps each.

Plus two diagnostics that distinguish the remaining hypotheses for the t1 drift
seen in earlier runs:

  Hypothesis A: the forward pass is numerically non-bit-identical between
      modes (layer-RoPE vs kernel-RoPE path difference). Argmax is stable,
      sampling is not.

  Hypothesis B: the forward pass is bit-identical, but spans mode advances the
      RNG state by an extra step (e.g. an RNG-touching op runs on the spans
      codepath). Argmax is unaffected, sampling draws from a shifted state.

Diagnostics:

  LOGPROB EQUIVALENCE TEST (primary) - LOGPROBS at t=0, max_tokens=1, logprobs=10.
      If top-10 logprobs are byte-identical across modes: A is refuted -> B.
      If they differ: A is confirmed.

  MULTI-SEED DRIFT TEST (secondary) - at t=1, max_tokens=16, seeds=[42, 43, 100, 7].
      Cross-checks: under B you'd expect every seed to produce a different
      result between modes; under A you'd expect a mix (some seeds hit a
      probability boundary and flip, others don't).
"""

import difflib
import gc
import os

from vllm import LLM, SamplingParams

MODEL_NAMES = [
    "ldsjmdy/Tulu3-Block-FT",
    "ldsjmdy/Tulu3-RAG",
    "ibm-granite/granite-3.1-8b-instruct",
    "/vllm-workspace/hub/models--meta-llama--Llama-4-Maverick-17B-128E-Instruct-FP8/snapshots/94125d2bd83076b21eed33119525e29eaf3894f4",
    "zai-org/GLM-4.7-FP8",
    "mistralai/Mistral-Large-3-675B-Instruct-2512",
    "NousResearch/Meta-Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-8B",
]
MODEL_INDEX = int(os.environ.get("VLLM_SPANS_MODEL_INDEX", "6"))
TP = int(os.environ.get("VLLM_SPANS_TP", "1"))

SEED = 42
MAX_TOKENS = 64
PROMPT = "Hello world! Please write a short greeting in one sentence."
NUM_REPS = 3
TEMPS = [0.0, 1.0]
LOGPROBS_TOPK = 10
MULTI_SEEDS = [42, 43, 100, 7]
MULTI_SEED_MAX_TOKENS = 16

SPAN_TOKEN_PLUS = 10
SPAN_TOKEN_CROSS = 31


def enable_spans() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "False"
    os.environ["VLLM_V1_SPANS_ENABLED"] = "True"
    os.environ["VLLM_V1_SPANS_TOKEN_PLUS"] = str(SPAN_TOKEN_PLUS)
    os.environ["VLLM_V1_SPANS_TOKEN_CROSS"] = str(SPAN_TOKEN_CROSS)
    os.environ["VLLM_V1_SPANS_DEBUG"] = "False"


def disable_spans() -> None:
    os.environ["VLLM_V1_SPANS_ENABLED"] = "False"


# Gap length so large it dwarfs the prompt, so span_aware gap policy (when it
# fires) is forced to recompute the entire cached prefix - making the resulting
# output equivalent to a full cold-start recompute.
GAP_LENGTH_HUGE = 1_000_000

MODES: list[str] = ["recompute", "spans_no_kv", "spans_gap_huge"]
MODE_SPECS: dict[str, tuple[bool, dict | None]] = {
    # name -> (spans_enabled, gap_policy_config)
    "recompute": (False, None),
    "spans_no_kv": (True, None),
    "spans_gap_huge": (
        True,
        {
            "gap_length": GAP_LENGTH_HUGE,
            "span_marker_token_id": SPAN_TOKEN_PLUS,
            "block_size": 16,
        },
    ),
}


def build_llm(
    model: str,
    spans_enabled: bool,
    gap_policy_config: dict | None = None,
) -> LLM:
    if spans_enabled:
        enable_spans()
    else:
        disable_spans()
    extra: dict = {}
    if gap_policy_config is not None:
        extra["gap_policy_name"] = "span_aware"
        extra["gap_policy_config"] = gap_policy_config
    return LLM(
        model=model,
        tensor_parallel_size=TP,
        kv_transfer_config=None,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        block_size=16,
        enable_prefix_caching=False,
        async_scheduling=False,
        attention_backend="TRITON_ATTN",
        **extra,
    )


def _make_sp(
    seed: int,
    temp: float,
    max_toks: int,
    logprobs: int | None = None,
) -> SamplingParams:
    # from_optional() has an explicit signature; direct SamplingParams(...)
    # uses msgspec.Struct metaclass magic which the type checker doesn't
    # resolve, so kwarg-style construction trips it up.
    return SamplingParams.from_optional(
        seed=seed,
        temperature=temp,
        max_tokens=max_toks,
        logprobs=logprobs,
    )


def _extract_step0_topk(out) -> list[tuple[int, float]]:
    """
    Return the step-0 top-K logprob entries as an ORDERED list of
    (token_id, logprob). Ordered by logprob desc, then by token_id for
    stable comparison. Bit-exact list equality between reps == bit-exact
    logprob match.
    """
    if out.logprobs is None or len(out.logprobs) == 0:
        return []
    step0 = out.logprobs[0]
    items = [(tid, float(lp.logprob)) for tid, lp in step0.items()]
    items.sort(key=lambda x: (-x[1], x[0]))
    return items


def run_reps(
    llm: LLM, temp: float
) -> tuple[list[str], list[list[tuple[int, float]]]]:
    """
    Runs NUM_REPS generations. Returns (texts, top10_per_rep) where top10_per_rep[i]
    is the step-0 top-K logprob list for rep i.
    """
    sp = _make_sp(SEED, temp, MAX_TOKENS, logprobs=LOGPROBS_TOPK)
    texts: list[str] = []
    top10s: list[list[tuple[int, float]]] = []
    for _ in range(NUM_REPS):
        res = llm.generate(PROMPT, sampling_params=sp, use_tqdm=False)
        out = res[0].outputs[0]
        texts.append(out.text)
        top10s.append(_extract_step0_topk(out))
    return texts, top10s


def get_first_token_logprobs(
    llm: LLM,
) -> list[tuple[int, str, float]]:
    """
    Returns the top-K logprobs for the first generated token at temperature=0.
    List of (token_id, decoded_text, logprob) sorted by logprob desc.
    """
    sp = _make_sp(SEED, 0.0, 1, logprobs=LOGPROBS_TOPK)
    res = llm.generate(PROMPT, sampling_params=sp, use_tqdm=False)
    out = res[0].outputs[0]
    if out.logprobs is None:
        raise RuntimeError(
            "Expected logprobs on the response but got None - check that "
            "SamplingParams(logprobs=...) was honored."
        )
    # out.logprobs is a list with 1 entry (one generated token);
    # that entry is a dict mapping token_id -> Logprob object.
    step0 = out.logprobs[0]
    items: list[tuple[int, str, float]] = []
    for tok_id, lp in step0.items():
        decoded = lp.decoded_token if lp.decoded_token is not None else ""
        items.append((tok_id, decoded, float(lp.logprob)))
    items.sort(key=lambda x: x[2], reverse=True)
    return items


def run_with_seed(
    llm: LLM, seed: int, temp: float, max_toks: int
) -> tuple[str, list[tuple[int, float]]]:
    sp = _make_sp(seed, temp, max_toks, logprobs=LOGPROBS_TOPK)
    res = llm.generate(PROMPT, sampling_params=sp, use_tqdm=False)
    out = res[0].outputs[0]
    return out.text, _extract_step0_topk(out)


def ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def truncate(s: str, n: int = 150) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "..."


def first_differ_index(a: str, b: str) -> int:
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return -1


def label(mode: str, temp: float) -> str:
    return f"{mode}_t{int(temp)}"


def run_gap_policy_replay_test(
    model: str,
    reference_text: str,
    reference_top10: list[tuple[int, float]],
) -> None:
    """
    Force gap policy's recomputation path to fire and verify it matches
    clean recompute.

    Gap policy only triggers on prefix-cache HITS. The rest of this test
    runs with enable_prefix_caching=False so gap policy is inert. Here we:

      1. Build an LLM with spans=True, gap_policy=span_aware,
         gap_length=GAP_LENGTH_HUGE, and enable_prefix_caching=True.
      2. Run PROMPT once cold (populates the cache).
      3. Run PROMPT a second time. Every block is now a cache hit, which
         hands the request to gap policy. With gap_length >> prompt length,
         gap policy should recompute the entire cached prefix.
      4. Compare the second run's output to the clean recompute-mode
         reference from the main 3-mode loop.

    Expected: second-run output is bit-identical to recompute. That is the
    "huge gap ≡ full recompute" equivalence claim.
    """
    print(
        f"\n{'=' * 72}\n"
        "GAP-POLICY REPLAY TEST (prefix_cache=ON, gap_length=huge)\n"
        f"{'=' * 72}"
    )

    enable_spans()
    llm = LLM(
        model=model,
        tensor_parallel_size=TP,
        kv_transfer_config=None,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        block_size=16,
        enable_prefix_caching=True,
        async_scheduling=False,
        attention_backend="TRITON_ATTN",
        gap_policy_name="span_aware",
        gap_policy_config={
            "gap_length": GAP_LENGTH_HUGE,
            "span_marker_token_id": SPAN_TOKEN_PLUS,
            "block_size": 16,
        },
    )

    sp = _make_sp(SEED, 0.0, MAX_TOKENS, logprobs=LOGPROBS_TOPK)

    print("  run #1 (cold prefill, populates cache)...")
    res1 = llm.generate(PROMPT, sampling_params=sp, use_tqdm=False)
    text1 = res1[0].outputs[0].text
    top10_1 = _extract_step0_topk(res1[0].outputs[0])
    print(f"    output: {truncate(text1)}")

    print("  run #2 (all blocks cached → gap policy fires → recompute)...")
    res2 = llm.generate(PROMPT, sampling_params=sp, use_tqdm=False)
    text2 = res2[0].outputs[0].text
    top10_2 = _extract_step0_topk(res2[0].outputs[0])
    print(f"    output: {truncate(text2)}")

    del llm
    gc.collect()
    try:
        import torch  # type: ignore[import-not-found]
        torch.cuda.empty_cache()
    except Exception:
        pass

    print("\n--- Verdict ---")

    # First: did the replay itself change anything? If run#2 differs from
    # run#1, gap policy did some recomputation that perturbed the output.
    # With deterministic sampling (t=0, same seed) and a correct
    # implementation, run#1 and run#2 should be bit-identical regardless
    # of whether gap policy fired.
    replay_stable = text1 == text2 and top10_1 == top10_2
    if replay_stable:
        print("  [OK]   run #1 == run #2 (replay is deterministic)")
    else:
        print("  [WARN] run #1 != run #2 (prefix cache / gap policy leaks "
              "nondeterminism)")

    # Second: does gap-policy-replay match clean recompute?
    text_match = text2 == reference_text
    top_match = top10_2 == reference_top10

    if text_match and top_match:
        print("  [PASS] replay run #2 == clean recompute (text + "
              f"top-{LOGPROBS_TOPK}-logprobs bit-identical)")
        print("  -> gap_length >= prompt_length ≡ full recompute ✓")
    else:
        print("  [FAIL] replay run #2 != clean recompute")
        if not text_match:
            print(f"    reference (recompute): {truncate(reference_text)}")
            print(f"    replay run #2        : {truncate(text2)}")
        if not top_match:
            ref_map = {tid: lp for tid, lp in reference_top10}
            rep_map = {tid: lp for tid, lp in top10_2}
            common = set(ref_map) & set(rep_map)
            if common:
                max_diff = max(abs(ref_map[t] - rep_map[t]) for t in common)
                print(f"    top-{LOGPROBS_TOPK} logprobs max|Δ| = "
                      f"{max_diff:.3e}")
            else:
                print(f"    top-{LOGPROBS_TOPK} token sets disjoint")


def main() -> None:
    model = MODEL_NAMES[MODEL_INDEX]
    print(f"Model: {model}")
    print(f"TP: {TP}  seed: {SEED}  max_tokens: {MAX_TOKENS}  reps: {NUM_REPS}")
    print(f"Prompt: {PROMPT!r}\n")

    results: dict[tuple[str, float], list[str]] = {}
    results_top10: dict[tuple[str, float], list[list[tuple[int, float]]]] = {}
    logprobs_by_mode: dict[str, list[tuple[int, str, float]]] = {}
    multi_seed_by_mode: dict[
        str, dict[int, tuple[str, list[tuple[int, float]]]]
    ] = {}

    for mode in MODES:
        spans_enabled, gap_policy_config = MODE_SPECS[mode]
        print(f"\n{'=' * 72}\nBuilding LLM: mode={mode}\n{'=' * 72}")
        llm = build_llm(
            model,
            spans_enabled=spans_enabled,
            gap_policy_config=gap_policy_config,
        )

        # --- Determinism matrix ---
        for temp in TEMPS:
            key = (mode, temp)
            print(f"  running {label(*key)} x{NUM_REPS}...")
            texts, top10s = run_reps(llm, temp)
            results[key] = texts
            results_top10[key] = top10s
            for i, r in enumerate(texts):
                print(f"    run#{i}: {truncate(r)}")

        # --- Logprob equivalence test: logprobs at first token, t=0, top-K ---
        print(f"  collecting top-{LOGPROBS_TOPK} logprobs for step 0 (t=0)...")
        logprobs_by_mode[mode] = get_first_token_logprobs(llm)

        # --- Multi-seed drift test: t=1 across several seeds ---
        print(f"  running multi-seed t=1 (seeds={MULTI_SEEDS})...")
        multi_seed_by_mode[mode] = {}
        for s in MULTI_SEEDS:
            multi_seed_by_mode[mode][s] = run_with_seed(
                llm, s, 1.0, MULTI_SEED_MAX_TOKENS
            )

        del llm
        gc.collect()
        try:
            import torch  # type: ignore[import-not-found]
            torch.cuda.empty_cache()
        except Exception:
            pass

    print(f"\n\n{'#' * 72}\n# REPORT\n{'#' * 72}")

    # 1. Within-config determinism
    print("\n=== WITHIN-CONFIG DETERMINISM (3 runs per config) ===")
    for key, runs in results.items():
        lbl = label(*key)
        top10s = results_top10[key]

        # Text comparison
        if runs[0] == runs[1] == runs[2]:
            text_verdict = "text-identical"
        else:
            r01 = ratio(runs[0], runs[1])
            r02 = ratio(runs[0], runs[2])
            text_verdict = f"text-DRIFT (r0-1={r01:.3f}, r0-2={r02:.3f})"

        # Top-K logprob bit-exactness check across the 3 reps.
        # _extract_step0_topk already sorts by (-logprob, token_id), so
        # bit-exact list equality ⇔ bit-exact top-K match (same ids in
        # the same order AND bit-identical float64 logprobs).
        if top10s[0] == top10s[1] == top10s[2]:
            top_verdict = f"top-{LOGPROBS_TOPK}-logprobs-identical"
        else:
            def _max_abs_diff(
                a: list[tuple[int, float]], b: list[tuple[int, float]]
            ) -> float:
                a_map = {tid: lp for tid, lp in a}
                b_map = {tid: lp for tid, lp in b}
                common = set(a_map) & set(b_map)
                if not common:
                    return float("inf")
                return max(abs(a_map[t] - b_map[t]) for t in common)

            d01 = _max_abs_diff(top10s[0], top10s[1])
            d02 = _max_abs_diff(top10s[0], top10s[2])
            top_verdict = (
                f"top-{LOGPROBS_TOPK}-logprobs-DIFFER "
                f"(max|Δ|_0-1={d01:.3e}, max|Δ|_0-2={d02:.3e})"
            )

        tag = "[PASS]" if (
            runs[0] == runs[1] == runs[2] and top10s[0] == top10s[1] == top10s[2]
        ) else "[FAIL]"
        print(f"{tag} {lbl:20s}: {text_verdict}, {top_verdict}")

    # 2. Cross-mode at same temp - compare every non-recompute mode to recompute
    print("\n=== CROSS-MODE vs recompute (same temp, run#0) ===")
    for temp in TEMPS:
        a = results[("recompute", temp)][0]
        a_top = results_top10[("recompute", temp)][0]
        tag = f"t{int(temp)}"

        for other in MODES:
            if other == "recompute":
                continue
            b = results[(other, temp)][0]
            b_top = results_top10[(other, temp)][0]

            text_match = a == b
            if text_match:
                text_part = "text-identical"
            else:
                text_part = (
                    f"text-DIFFER (ratio={ratio(a, b):.3f}, "
                    f"first_differ_char={first_differ_index(a, b)})"
                )

            top_match = a_top == b_top
            if top_match:
                top_part = f"top-{LOGPROBS_TOPK}-logprobs-identical"
            else:
                a_map = {tid: lp for tid, lp in a_top}
                b_map = {tid: lp for tid, lp in b_top}
                common = set(a_map) & set(b_map)
                if common:
                    max_diff = max(abs(a_map[t] - b_map[t]) for t in common)
                    top_part = (
                        f"top-{LOGPROBS_TOPK}-logprobs-DIFFER "
                        f"(max|Δ|={max_diff:.3e})"
                    )
                else:
                    top_part = (
                        f"top-{LOGPROBS_TOPK}-logprobs-DIFFER "
                        f"(no overlap in top-{LOGPROBS_TOPK} ids)"
                    )

            verdict = "[PASS]" if (text_match and top_match) else "[WARN]"
            print(
                f"{verdict} {tag:6s} recompute vs {other:14s}: "
                f"{text_part}, {top_part}"
            )
            if not text_match:
                print(f"         recompute       : {truncate(a)}")
                print(f"         {other:14s}  : {truncate(b)}")

    # 3. Cross-temp within same mode
    print("\n=== CROSS-TEMP (same mode, run#0) ===")
    for mode in MODES:
        a = results[(mode, 0.0)][0]
        b = results[(mode, 1.0)][0]
        r = ratio(a, b)
        if a == b:
            print(f"[WARN] {mode:14s}: t0 == t1 (seed/temp wiring broken?)")
        else:
            print(f"[OK]   {mode:14s}: t0 != t1 as expected (ratio={r:.3f})")

    # =========================================================================
    # Logprob equivalence test: logprobs at step 0, t=0
    # =========================================================================
    print(
        f"\n=== LOGPROB EQUIVALENCE TEST: TOP-{LOGPROBS_TOPK} LOGPROBS AT "
        f"STEP 0 (t=0, max_tokens=1) ==="
    )
    for mode in MODES:
        print(f"\n{mode}:")
        for i, (tid, tok, lp) in enumerate(logprobs_by_mode[mode]):
            print(f"  #{i + 1:2d}  id={tid:6d}  logprob={lp:.10f}  {tok!r}")

    rc = logprobs_by_mode["recompute"]
    rc_ids = [t[0] for t in rc]
    rc_lps = [t[2] for t in rc]
    rc_map = {tid: lp for tid, _, lp in rc}

    print("\n--- Logprob equivalence verdict (vs recompute) ---")
    for other in MODES:
        if other == "recompute":
            continue
        ot = logprobs_by_mode[other]
        ot_ids = [t[0] for t in ot]
        ot_lps = [t[2] for t in ot]
        ot_map = {tid: lp for tid, _, lp in ot}

        same_ids = rc_ids == ot_ids
        same_lps_bit = rc_lps == ot_lps
        max_abs_lp_diff = 0.0
        common_ids = set(rc_map) & set(ot_map)
        if common_ids:
            max_abs_lp_diff = max(
                abs(rc_map[tid] - ot_map[tid]) for tid in common_ids
            )

        verdict = (
            "bit-identical"
            if same_lps_bit and same_ids
            else f"max|Δlogprob|={max_abs_lp_diff:.3e}"
        )
        print(
            f"  recompute vs {other:14s}: ids_in_order={same_ids}  "
            f"lps_bit_identical={same_lps_bit}  {verdict}"
        )

    # =========================================================================
    # Multi-seed drift test: t=1
    # =========================================================================
    print(
        f"\n=== MULTI-SEED TEXT & LOG PROBS DRIFT TEST: t=1, "
        f"max_tokens={MULTI_SEED_MAX_TOKENS} ==="
    )
    matches = 0
    differs = 0
    first_diff_positions: list[int] = []
    for other in MODES:
        if other == "recompute":
            continue
        print(f"\n-- recompute vs {other} --")
        for s in MULTI_SEEDS:
            a_text, a_top = multi_seed_by_mode["recompute"][s]
            b_text, b_top = multi_seed_by_mode[other][s]
            text_same = a_text == b_text
            top_same = a_top == b_top
            fd = first_differ_index(a_text, b_text)
            r = ratio(a_text, b_text)

            if top_same:
                top_part = f"top-{LOGPROBS_TOPK}-logprobs-identical"
            else:
                a_map = {tid: lp for tid, lp in a_top}
                b_map = {tid: lp for tid, lp in b_top}
                common = set(a_map) & set(b_map)
                if common:
                    max_diff = max(abs(a_map[t] - b_map[t]) for t in common)
                    top_part = (
                        f"top-{LOGPROBS_TOPK}-logprobs-DIFFER "
                        f"(max|Δ|={max_diff:.3e})"
                    )
                else:
                    top_part = (
                        f"top-{LOGPROBS_TOPK}-logprobs-DIFFER "
                        f"(no overlap in top-{LOGPROBS_TOPK} ids)"
                    )

            if text_same and top_same:
                matches += 1
                print(
                    f"  seed={s:4d}  [MATCH]   ratio=1.000  {top_part}  "
                    f"{truncate(a_text, 60)!r}"
                )
            else:
                differs += 1
                if not text_same:
                    first_diff_positions.append(fd)
                text_part = (
                    "text-identical"
                    if text_same
                    else f"text-DIFFER ratio={r:.3f} first_differ_char={fd}"
                )
                print(
                    f"  seed={s:4d}  [DIFFER]  {text_part}, {top_part}"
                )
                if not text_same:
                    print(f"    recompute      : {truncate(a_text, 60)}")
                    print(f"    {other:14s} : {truncate(b_text, 60)}")

    total_comparisons = len(MULTI_SEEDS) * (len(MODES) - 1)
    print("\n--- Multi-seed drift verdict ---")
    print(f"seeds that MATCH across modes : {matches}/{total_comparisons}")
    print(f"seeds that DIFFER across modes: {differs}/{total_comparisons}")
    if first_diff_positions:
        print(f"first-differ character positions: {first_diff_positions}")
    if matches == total_comparisons:
        print(
            "\nAll seeds match across modes at t=1. Combined with the "
            "cross-mode t1 drift you saw with seed=42 earlier, this would "
            "be surprising - double-check the test harness."
        )
    elif differs == total_comparisons:
        print(
            "\nAll seeds drift across modes. Consistent with either A or B. "
            "Use the LOGPROB EQUIVALENCE TEST to disambiguate - "
            "the multi-seed drift test alone cannot."
        )
    else:
        print(
            "\nMixed result: some seeds drift, some don't. Suggests A - "
            "logits differ slightly, and only seeds whose RNG draw lands "
            "near a probability boundary end up flipping a token."
        )

    # Pick up the clean recompute baseline at t=0 run#0 for the replay test
    # to compare against.
    reference_text = results[("recompute", 0.0)][0]
    reference_top10 = results_top10[("recompute", 0.0)][0]
    run_gap_policy_replay_test(model, reference_text, reference_top10)

    print(f"\n{'#' * 72}")


if __name__ == "__main__":
    main()
