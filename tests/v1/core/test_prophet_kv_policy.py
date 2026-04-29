# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ProphetKV gap policy integration (spans mode, env-var driven).

Covers:
  - Core algorithm from prophet_kv.py (Algorithm 1 of the ProphetKV paper):
    Stage I (compute_layer_alpha), Stage II (fuse_and_select), interval
    coalescing (indices_to_gap_intervals), end-to-end ProphetKVSelector.
  - ProphetKVGapPolicy get_gaps() contract.
  - GapPolicyFactory env-var driven construction for
    VLLM_V1_SPANS_PROPHET_KV_* knobs.

Run:
  .venv/bin/python -m pytest tests/v1/core/test_prophet_kv_policy.py -v
"""

from __future__ import annotations

import math
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.v1.core.sched.gap_policy import (
    GapPolicy,
    GapPolicyFactory,
    NoGapPolicy,
)
from vllm.v1.core.sched.prophet_kv import (
    ProphetKVConfig,
    ProphetKVGapPolicy,
    ProphetKVSelector,
    compute_layer_alpha,
    fuse_and_select,
    indices_to_gap_intervals,
    prophet_kv_policy_from_env,
    scores_to_gap_intervals,
)


def _qk(h_q=2, h_kv=2, q_len=3, c_len=8, d=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(h_q, q_len, d, generator=g),
        torch.randn(h_kv, c_len, d, generator=g),
    )


# ---- Stage I ------------------------------------------------------------


class TestStageI:
    def test_shape_and_nonneg(self):
        q, k = _qk(h_q=4, h_kv=4, q_len=3, c_len=16, d=8)
        alpha = compute_layer_alpha(q, k, head_dim=8)
        assert alpha.shape == (16,)
        assert torch.all(alpha >= 0)

    def test_total_mass_equals_Hq_times_Qs(self):
        q, k = _qk(h_q=3, h_kv=3, q_len=5, c_len=12, d=8)
        alpha = compute_layer_alpha(q, k, head_dim=8)
        assert math.isclose(alpha.sum().item(), 3 * 5, abs_tol=1e-4)

    def test_matches_hand_computation(self):
        q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        k = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
        d = 2
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
        ref = torch.softmax(logits, dim=-1).sum(dim=(0, 1))
        got = compute_layer_alpha(q, k, head_dim=d)
        torch.testing.assert_close(got, ref, rtol=1e-6, atol=1e-6)

    def test_gqa(self):
        q, k = _qk(h_q=4, h_kv=2, q_len=2, c_len=6, d=4)
        got = compute_layer_alpha(q, k, head_dim=4)
        ref = compute_layer_alpha(q, k.repeat_interleave(2, dim=0), head_dim=4)
        torch.testing.assert_close(got, ref)

    def test_bad_gqa_ratio_raises(self):
        q, k = _qk(h_q=3, h_kv=2)
        with pytest.raises(ValueError, match="divisible"):
            compute_layer_alpha(q, k, head_dim=4)

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError, match="3D"):
            compute_layer_alpha(torch.zeros(2, 3), torch.zeros(2, 3, 4), head_dim=4)


# ---- Stage II -----------------------------------------------------------


class TestFuseAndSelect:
    def test_top_p_budget(self):
        idx = fuse_and_select([torch.arange(100.0)], recomp_ratio=0.2)
        assert len(idx) == 20 and idx.tolist() == list(range(80, 100))

    def test_sum_vs_mean_equivalent_for_top_p(self):
        torch.manual_seed(1)
        alphas = [torch.rand(32) for _ in range(6)]
        idx_mean = fuse_and_select(alphas, recomp_ratio=0.25)
        ref = torch.sort(
            torch.topk(torch.stack(alphas).sum(dim=0), k=8).indices
        ).values
        torch.testing.assert_close(idx_mean, ref)

    def test_min_tokens_floor(self):
        idx = fuse_and_select([torch.arange(10.0)], recomp_ratio=0.1, min_tokens=3)
        assert len(idx) == 3

    def test_empty_and_invalid(self):
        with pytest.raises(ValueError):
            fuse_and_select([], recomp_ratio=0.5)
        with pytest.raises(ValueError):
            fuse_and_select([torch.ones(4)], recomp_ratio=-0.1)
        with pytest.raises(ValueError):
            fuse_and_select([torch.ones(4)], recomp_ratio=1.5)

    def test_zero_budget(self):
        assert fuse_and_select([torch.ones(10)], recomp_ratio=0.0).numel() == 0


# ---- Selector end-to-end -----------------------------------------------


class TestSelector:
    def test_query_aligned_chunk_dominates(self):
        torch.manual_seed(0)
        h, d = 2, 8
        q = torch.randn(h, 1, d)
        q = q / q.norm(dim=-1, keepdim=True)
        chunk = 10
        rnd_a = torch.randn(h, chunk, d)
        aligned = q.expand(h, chunk, d) + 0.005 * torch.randn(h, chunk, d)
        rnd_b = torch.randn(h, chunk, d)
        k = torch.cat([rnd_a, aligned, rnd_b], dim=1)
        sel = ProphetKVSelector(ProphetKVConfig(recomp_ratio=1 / 3))
        idx = sel.select([q], [k], head_dim=d)
        in_aligned = ((idx >= chunk) & (idx < 2 * chunk)).sum().item()
        assert in_aligned >= int(0.8 * len(idx))

    def test_determinism(self):
        torch.manual_seed(7)
        q = [torch.randn(2, 3, 8) for _ in range(4)]
        k = [torch.randn(2, 50, 8) for _ in range(4)]
        sel = ProphetKVSelector(ProphetKVConfig(recomp_ratio=0.2))
        torch.testing.assert_close(
            sel.select(q, k, head_dim=8), sel.select(q, k, head_dim=8)
        )

    def test_gaps_are_contiguous_spans(self):
        torch.manual_seed(3)
        q = [torch.randn(1, 2, 4)]
        k = [torch.randn(1, 64, 4)]
        sel = ProphetKVSelector(ProphetKVConfig(recomp_ratio=0.25, block_size=16))
        for s, e in sel.gaps(q, k, head_dim=4, total_len=64):
            assert 0 <= s < e <= 64

    def test_varying_C_raises(self):
        sel = ProphetKVSelector()
        with pytest.raises(ValueError, match="context length varies"):
            sel.select(
                [torch.randn(1, 1, 4), torch.randn(1, 1, 4)],
                [torch.randn(1, 4, 4), torch.randn(1, 5, 4)],
                head_dim=4,
            )


# ---- indices_to_gap_intervals ------------------------------------------


class TestIntervals:
    def test_empty(self):
        assert indices_to_gap_intervals([], total_len=10) == []

    def test_contiguous_runs(self):
        assert indices_to_gap_intervals([0, 1, 2, 5, 6, 9], total_len=10) == [
            (0, 3), (5, 7), (9, 10),
        ]

    def test_gap_threshold_merges_nearby_spans(self):
        assert indices_to_gap_intervals(
            [1, 2, 5, 6, 10], total_len=16, gap_threshold=2
        ) == [(1, 11)]

    def test_max_num_spans_merges_closest_gaps(self):
        assert indices_to_gap_intervals(
            [1, 4, 8, 20], total_len=32, max_num_spans=2
        ) == [(1, 9), (20, 21)]

    def test_oor_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            indices_to_gap_intervals([10], total_len=10)

    def test_tensor_input(self):
        t = torch.tensor([2, 4, 6], dtype=torch.long)
        assert indices_to_gap_intervals(t, total_len=8) == [
            (2, 3), (4, 5), (6, 7)
        ]

    def test_negative_gap_threshold_raises(self):
        with pytest.raises(ValueError, match="gap_threshold"):
            indices_to_gap_intervals([1], total_len=8, gap_threshold=-1)

    def test_negative_max_num_spans_raises(self):
        with pytest.raises(ValueError, match="max_num_spans"):
            indices_to_gap_intervals([1], total_len=8, max_num_spans=-1)


class TestScoresToGapIntervals:
    def test_token_topk_then_contiguous_spans(self):
        scores = torch.tensor([0.0, 0.9, 0.8, 0.0, 0.7, 0.0])
        assert scores_to_gap_intervals(scores, 6, recomp_ratio=0.5) == [
            (1, 3), (4, 5)
        ]

    def test_span_controls_reduce_fragmentation(self):
        scores = torch.tensor([0.9, 0.0, 0.8, 0.0, 0.7, 0.0, 0.6])
        assert scores_to_gap_intervals(
            scores,
            7,
            recomp_ratio=4 / 7,
            gap_threshold=0,
            max_num_spans=2,
        ) == [(0, 5), (6, 7)]


# ---- ProphetKVGapPolicy -------------------------------------------------


class TestProphetKVGapPolicy:
    def test_is_a_gap_policy(self):
        assert issubclass(ProphetKVGapPolicy, GapPolicy)

    def test_no_scores_falls_back_to_no_gaps(self):
        p = ProphetKVGapPolicy(recomp_ratio=0.2)
        req = SimpleNamespace(kv_transfer_params=None)
        assert p.get_gaps(req, 16, 0) == []

    def test_scores_via_kv_transfer_params(self):
        scores = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.0, 0.95])
        req = SimpleNamespace(
            kv_transfer_params={"prophet_kv_scores": scores}
        )
        p = ProphetKVGapPolicy(recomp_ratio=0.5)
        assert p.get_gaps(req, 6, 0) == [(1, 2), (3, 4), (5, 6)]

    def test_paper_default_20pct_budget(self):
        torch.manual_seed(42)
        scores = torch.rand(100)
        req = SimpleNamespace(
            kv_transfer_params={"prophet_kv_scores": scores}
        )
        gaps = ProphetKVGapPolicy().get_gaps(req, 100, 0)
        assert sum(e - s for s, e in gaps) == 20

    def test_honors_external_tokens(self):
        scores = torch.arange(10.0)
        req = SimpleNamespace(
            kv_transfer_params={"prophet_kv_scores": scores}
        )
        p = ProphetKVGapPolicy(recomp_ratio=0.3)
        # Top 3 over all 10 computed tokens -> indices 7,8,9 -> one run.
        assert p.get_gaps(req, 4, 6) == [(7, 10)]

    def test_gap_threshold(self):
        scores = torch.zeros(10)
        scores[1] = 1.0
        scores[4] = 0.9
        scores[7] = 0.8
        req = SimpleNamespace(
            kv_transfer_params={"prophet_kv_scores": scores}
        )
        p = ProphetKVGapPolicy(recomp_ratio=0.3, gap_threshold=2)
        assert p.get_gaps(req, 10, 0) == [(1, 8)]

    def test_max_num_spans(self):
        scores = torch.tensor([1.0, 0.0, 0.9, 0.0, 0.8, 0.0, 0.7])
        req = SimpleNamespace(
            kv_transfer_params={"prophet_kv_scores": scores}
        )
        p = ProphetKVGapPolicy(recomp_ratio=4 / 7, max_num_spans=2)
        assert p.get_gaps(req, 7, 0) == [(0, 3), (4, 7)]

    def test_custom_score_key(self):
        scores = torch.tensor([1.0, 0.0, 1.0])
        req = SimpleNamespace(kv_transfer_params={"my_key": scores})
        p = ProphetKVGapPolicy(recomp_ratio=2 / 3, score_key="my_key")
        assert p.get_gaps(req, 3, 0) == [(0, 1), (2, 3)]

    def test_score_provider_callback(self):
        scores = torch.tensor([0.0, 1.0, 0.0, 1.0])
        p = ProphetKVGapPolicy(recomp_ratio=0.5, score_provider=lambda r: scores)
        req = SimpleNamespace(kv_transfer_params=None)
        assert p.get_gaps(req, 4, 0) == [(1, 2), (3, 4)]

    def test_bad_ratio_raises(self):
        with pytest.raises(ValueError):
            ProphetKVGapPolicy(recomp_ratio=1.5)


# ---- Factory + env-var path --------------------------------------------


class TestFactoryEnv:
    def test_create_none(self):
        assert GapPolicyFactory.create_policy(None) is None

    def test_create_prophet_kv(self):
        p = GapPolicyFactory.create_policy(
            "prophet_kv", {"recomp_ratio": 0.3, "block_size": 4}
        )
        assert isinstance(p, ProphetKVGapPolicy)
        assert p.recomp_ratio == 0.3 and p.block_size == 4

    def test_unknown_falls_back(self):
        p = GapPolicyFactory.create_policy("bogus")
        assert isinstance(p, NoGapPolicy)

    def test_bad_config_propagates(self):
        # recomp_ratio outside [0,1] raises ValueError in __init__; the
        # vendored factory only catches TypeError, so ValueError bubbles.
        with pytest.raises(ValueError):
            GapPolicyFactory.create_policy("prophet_kv", {"recomp_ratio": 2.0})

    def test_env_var_default_off(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VLLM_V1_SPANS_PROPHET_KV_ENABLED", None)
            assert prophet_kv_policy_from_env() is None

    def test_env_var_turns_on_prophet_kv(self):
        env = {
            "VLLM_V1_SPANS_PROPHET_KV_ENABLED": "True",
            "VLLM_V1_SPANS_PROPHET_KV_RATIO": "0.35",
            "VLLM_V1_SPANS_PROPHET_KV_MIN_TOKENS": "2",
            "VLLM_V1_SPANS_PROPHET_KV_GAP_THRESHOLD": "3",
            "VLLM_V1_SPANS_PROPHET_KV_MAX_NUM_SPANS": "12",
            "VLLM_V1_SPANS_PROPHET_KV_BLOCK_SIZE": "8",
            "VLLM_V1_SPANS_PROPHET_KV_SCORE_KEY": "alpha_bar",
        }
        with patch.dict(os.environ, env, clear=False):
            p = prophet_kv_policy_from_env()
        assert isinstance(p, ProphetKVGapPolicy)
        assert p.recomp_ratio == pytest.approx(0.35)
        assert p.gap_threshold == 3
        assert p.max_num_spans == 12
        assert p.block_size == 8
        assert p.min_tokens == 2
        assert p.score_key == "alpha_bar"

    def test_env_var_zero_block_size_disables_block_fallback(self):
        env = {
            "VLLM_V1_SPANS_PROPHET_KV_ENABLED": "True",
            "VLLM_V1_SPANS_PROPHET_KV_BLOCK_SIZE": "0",
        }
        with patch.dict(os.environ, env, clear=False):
            p = prophet_kv_policy_from_env(default_block_size=16)
        assert isinstance(p, ProphetKVGapPolicy)
        assert p.block_size == 1

    def test_explicit_policy_name_wins_over_env(self):
        env = {"VLLM_V1_SPANS_PROPHET_KV_ENABLED": "True"}
        with patch.dict(os.environ, env, clear=False):
            p = prophet_kv_policy_from_env(policy_name="none")
        assert isinstance(p, NoGapPolicy)

    def test_auto_registered_with_factory(self):
        # Importing prophet_kv triggers GapPolicyFactory.register_policy.
        assert "prophet_kv" in GapPolicyFactory._POLICIES
        assert GapPolicyFactory._POLICIES["prophet_kv"] is ProphetKVGapPolicy

    def test_register_rejects_non_policy(self):
        class NotAPolicy:
            pass

        # The vendored factory's message is "must be a subclass of GapPolicy".
        with pytest.raises(ValueError, match="subclass of GapPolicy"):
            GapPolicyFactory.register_policy("x", NotAPolicy)


# ---- End-to-end: selector -> alpha_bar -> policy -> gaps ---------------


class TestIntegration:
    def test_selector_output_drives_policy(self):
        """Paper Algorithm 1 end-to-end: compute alpha_bar with the selector,
        stash it on the request, read it back via the policy."""
        torch.manual_seed(11)
        h, d, c = 2, 8, 32
        q = torch.randn(h, 1, d)
        q = q / q.norm(dim=-1, keepdim=True)
        k = torch.randn(h, c, d) * 0.5
        # Aligned slice: copies of q with small noise -> highest attention.
        k[:, 10:20, :] = q.expand(h, 10, d) + 0.01 * torch.randn(h, 10, d)

        from vllm.v1.core.sched.prophet_kv import compute_layer_alpha
        alpha = compute_layer_alpha(q, k, head_dim=d)
        req = SimpleNamespace(kv_transfer_params={"prophet_kv_scores": alpha})
        gaps = ProphetKVGapPolicy(recomp_ratio=0.3).get_gaps(req, c, 0)
        covered = sum(max(0, min(e, 20) - max(s, 10)) for s, e in gaps)
        total = sum(e - s for s, e in gaps)
        assert total > 0 and covered / total >= 0.8
