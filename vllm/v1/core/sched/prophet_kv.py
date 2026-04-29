# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
ProphetKV: user-query-driven selective KV cache recomputation.

Algorithm from Wang et al., "ProphetKV: User-Query-Driven Selective
Recomputation for Efficient KV Cache Reuse in Retrieval-Augmented Generation"
(see prophet_kv_paper/body/3_design.tex and appendix.tex Algorithm 1).

  Stage I (per layer l):
      alpha_l(t) = col-sum over q in Q_s of
                     softmax_t(Q_s^(l) K'_{1:C}^(l).T / sqrt(d_k))
  Stage II (uniform fusion, Eq. 8):
      alpha_bar(t) = mean_l alpha_l(t)
      T_p          = Top-p indices by alpha_bar.

This module plugs into vLLM's existing gap policy abstraction
(vllm.v1.core.sched.gap_policy.GapPolicy / GapPolicyFactory). At import time
it registers ProphetKVGapPolicy under the name "prophet_kv" with the
factory, so it works through the same `gap_policy_name` /
`gap_policy_config` plumbing the rest of spans mode already uses.

Turning it on:
  - via env var: VLLM_V1_SPANS_PROPHET_KV_ENABLED=True (other knobs:
    VLLM_V1_SPANS_PROPHET_KV_RATIO / _MIN_TOKENS /
    _GAP_THRESHOLD / _MAX_NUM_SPANS / _BLOCK_SIZE / _SCORE_KEY).
    Call prophet_kv_policy_from_env(...) from the scheduler init.
  - via CLI: --gap-policy-name prophet_kv --gap-policy-config '{"recomp_ratio":0.2}'
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.v1.core.sched.gap_policy import GapPolicy, GapPolicyFactory

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)

PROPHET_KV_PENDING_RECOMPUTE_KEY = "_prophet_kv_pending_recompute"
PROPHET_KV_STAGE_GAP_RECOMPUTE = "gap_recompute"
PROPHET_KV_STAGE_QUERY_RECOMPUTE = "query_recompute"


# --- Stage I / Stage II ---------------------------------------------------


def compute_layer_alpha(
    q_proj: torch.Tensor,
    k_cache: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    """Stage I per layer. Returns alpha_l(t) shaped [C].

    q_proj : [H_q, |Q_s|, D]  - Q projection of the user query tokens.
    k_cache: [H_kv, C, D]     - K' projection of context tokens (imprecise
                                cache; Algorithm 1 line 4).
    GQA is handled by repeat_interleave when H_q != H_kv.
    """
    if q_proj.dim() != 3 or k_cache.dim() != 3:
        raise ValueError(
            f"expected 3D tensors [heads, tokens, dim]; "
            f"got q={tuple(q_proj.shape)}, k={tuple(k_cache.shape)}"
        )
    h_q, _, d_q = q_proj.shape
    h_kv, _, d_k = k_cache.shape
    if d_q != head_dim or d_k != head_dim:
        raise ValueError(f"head_dim mismatch: q={d_q}, k={d_k}, arg={head_dim}")
    if h_q % h_kv != 0:
        raise ValueError(f"H_q ({h_q}) must be divisible by H_kv ({h_kv})")
    if h_q != h_kv:
        k_cache = k_cache.repeat_interleave(h_q // h_kv, dim=0)
    scale = 1.0 / (head_dim**0.5)
    logits = torch.matmul(q_proj, k_cache.transpose(-2, -1)) * scale
    return torch.softmax(logits.float(), dim=-1).sum(dim=(0, 1))


def fuse_and_select(
    per_layer_alpha: Sequence[torch.Tensor],
    recomp_ratio: float,
    min_tokens: int = 0,
) -> torch.Tensor:
    """Stage II: uniform-fuse across layers and return sorted top-p indices."""
    if not 0.0 <= recomp_ratio <= 1.0:
        raise ValueError(f"recomp_ratio must be in [0,1]; got {recomp_ratio}")
    if len(per_layer_alpha) == 0:
        raise ValueError("per_layer_alpha must be non-empty")
    stacked = torch.stack([a.detach().float() for a in per_layer_alpha], dim=0)
    if stacked.dim() != 2:
        raise ValueError(f"per-layer alphas must be 1-D [C]; got {stacked.shape}")
    alpha_bar = stacked.mean(dim=0)
    c_len = alpha_bar.shape[0]
    k = min(max(min_tokens, int(round(recomp_ratio * c_len))), c_len)
    if k == 0:
        return torch.empty(0, dtype=torch.long, device=alpha_bar.device)
    top = torch.topk(alpha_bar, k=k, largest=True, sorted=False).indices
    return torch.sort(top).values


def indices_to_gap_intervals(
    indices: torch.Tensor | Sequence[int],
    total_len: int,
    gap_threshold: int = 0,
    max_num_spans: int = 0,
) -> list[tuple[int, int]]:
    """Coalesce selected indices into half-open (start, end) intervals.

    Output matches the GapPolicy contract: sorted, non-overlapping, half-open,
    all within [0, total_len). Optional `gap_threshold` merges nearby spans to
    reduce fragmentation; optional `max_num_spans` further merges the closest
    adjacent spans until the cap is satisfied.
    """
    if total_len < 0:
        raise ValueError(f"total_len must be non-negative; got {total_len}")
    if gap_threshold < 0:
        raise ValueError(f"gap_threshold must be non-negative; got {gap_threshold}")
    if max_num_spans < 0:
        raise ValueError(f"max_num_spans must be non-negative; got {max_num_spans}")
    if isinstance(indices, torch.Tensor):
        idx = indices.detach().cpu().tolist()
    else:
        idx = list(indices)
    if not idx:
        return []
    idx = sorted(set(int(i) for i in idx))
    if idx[0] < 0 or idx[-1] >= total_len:
        raise ValueError(
            f"indices out of range [0,{total_len}): [{idx[0]}, {idx[-1]}]"
        )

    runs: list[tuple[int, int]] = []
    start = prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            runs.append((start, prev + 1))
            start = prev = i
    runs.append((start, prev + 1))
    if gap_threshold > 0 and len(runs) > 1:
        merged_runs: list[tuple[int, int]] = [runs[0]]
        for s, e in runs[1:]:
            prev_s, prev_e = merged_runs[-1]
            if s - prev_e <= gap_threshold:
                merged_runs[-1] = (prev_s, e)
            else:
                merged_runs.append((s, e))
        runs = merged_runs
    if max_num_spans == 0 or len(runs) <= max_num_spans:
        return runs

    spans = runs[:]
    while len(spans) > max_num_spans:
        merge_idx = min(
            range(len(spans) - 1),
            key=lambda i: (
                spans[i + 1][0] - spans[i][1],
                spans[i + 1][1] - spans[i][0],
                i,
            ),
        )
        spans[merge_idx : merge_idx + 2] = [
            (spans[merge_idx][0], spans[merge_idx + 1][1])
        ]
    return spans


def scores_to_gap_intervals(
    scores: torch.Tensor,
    total_len: int,
    recomp_ratio: float,
    *,
    min_tokens: int = 0,
    gap_threshold: int = 0,
    max_num_spans: int = 0,
) -> list[tuple[int, int]]:
    """Select recompute intervals directly from fused scores.

    ProphetKV selects top-scoring tokens, then emits minimal contiguous spans
    covering those tokens. Optional `gap_threshold` and `max_num_spans`
    reduce fragmentation while preserving coverage.
    """
    if not 0.0 <= recomp_ratio <= 1.0:
        raise ValueError(f"recomp_ratio must be in [0,1]; got {recomp_ratio}")
    if total_len < 0:
        raise ValueError(f"total_len must be non-negative; got {total_len}")
    if total_len == 0:
        return []

    scores = scores.detach().float().reshape(-1)
    if scores.shape[0] < total_len:
        total_len = scores.shape[0]
    elif scores.shape[0] > total_len:
        scores = scores[:total_len]

    k_tokens = min(max(int(min_tokens), int(round(recomp_ratio * total_len))), total_len)
    if k_tokens == 0:
        return []
    idx = torch.topk(scores, k=k_tokens, largest=True, sorted=False).indices
    return indices_to_gap_intervals(
        idx,
        total_len=total_len,
        gap_threshold=gap_threshold,
        max_num_spans=max_num_spans,
    )


# --- Selector orchestrator ------------------------------------------------


@dataclass
class ProphetKVConfig:
    """Tunables for the dual-stage pipeline."""

    recomp_ratio: float = 0.2
    min_tokens: int = 0
    gap_threshold: int = 0
    max_num_spans: int = 0
    block_size: int = 1
    score_key: str = "prophet_kv_scores"


class ProphetKVSelector:
    """Stateless orchestrator: per-layer Q, K' -> selected indices / gaps."""

    def __init__(self, config: ProphetKVConfig | None = None) -> None:
        self.config = config or ProphetKVConfig()

    def select(
        self,
        q_per_layer: Sequence[torch.Tensor],
        k_per_layer: Sequence[torch.Tensor],
        head_dim: int,
    ) -> torch.Tensor:
        if len(q_per_layer) != len(k_per_layer) or len(q_per_layer) == 0:
            raise ValueError("q_per_layer and k_per_layer must be non-empty, same len")
        alphas: list[torch.Tensor] = []
        c_prev: int | None = None
        for q, k in zip(q_per_layer, k_per_layer):
            a = compute_layer_alpha(q, k, head_dim)
            if c_prev is not None and a.shape[0] != c_prev:
                raise ValueError(f"context length varies: {c_prev} vs {a.shape[0]}")
            c_prev = a.shape[0]
            alphas.append(a)
        return fuse_and_select(
            alphas, self.config.recomp_ratio, self.config.min_tokens
        )

    def gaps(
        self,
        q_per_layer: Sequence[torch.Tensor],
        k_per_layer: Sequence[torch.Tensor],
        head_dim: int,
        total_len: int,
    ) -> list[tuple[int, int]]:
        if len(q_per_layer) != len(k_per_layer) or len(q_per_layer) == 0:
            raise ValueError("q_per_layer and k_per_layer must be non-empty, same len")
        alphas: list[torch.Tensor] = []
        c_prev: int | None = None
        for q, k in zip(q_per_layer, k_per_layer):
            a = compute_layer_alpha(q, k, head_dim)
            if c_prev is not None and a.shape[0] != c_prev:
                raise ValueError(f"context length varies: {c_prev} vs {a.shape[0]}")
            c_prev = a.shape[0]
            alphas.append(a)
        alpha_bar = torch.stack(alphas, dim=0).mean(dim=0)
        return scores_to_gap_intervals(
            alpha_bar,
            total_len=total_len,
            recomp_ratio=self.config.recomp_ratio,
            min_tokens=self.config.min_tokens,
            gap_threshold=self.config.gap_threshold,
            max_num_spans=self.config.max_num_spans,
        )


# --- GapPolicy plug-in ----------------------------------------------------


class ProphetKVGapPolicy(GapPolicy):
    """Query-driven top-p selection over a precomputed score tensor.

    Reads alpha_bar from `request.kv_transfer_params[score_key]`, populated
    by the model runner's lightweight Stage-I pass before scheduling. With
    the score absent this falls back to NoGapPolicy semantics (full reuse),
    which is the safe default for requests the runner did not annotate.
    """

    def __init__(
        self,
        recomp_ratio: float = 0.2,
        block_size: int = 1,
        min_tokens: int = 0,
        gap_threshold: int = 0,
        max_num_spans: int = 0,
        score_key: str = "prophet_kv_scores",
        score_provider: Callable[["Request"], torch.Tensor | None] | None = None,
    ) -> None:
        if not 0.0 <= recomp_ratio <= 1.0:
            raise ValueError(f"recomp_ratio must be in [0,1]; got {recomp_ratio}")
        self.recomp_ratio = recomp_ratio
        self.block_size = max(1, int(block_size))
        self.min_tokens = max(0, int(min_tokens))
        self.gap_threshold = max(0, int(gap_threshold))
        self.max_num_spans = max(0, int(max_num_spans))
        self.score_key = score_key
        self.score_provider = score_provider
        logger.info(
            "ProphetKVGapPolicy: ratio=%.3f min_tokens=%d gap_threshold=%d "
            "max_num_spans=%d block_size=%d key=%s",
            self.recomp_ratio,
            self.min_tokens,
            self.gap_threshold,
            self.max_num_spans,
            self.block_size,
            self.score_key,
        )

    def get_gaps(
        self,
        request: "Request",
        num_computed_tokens: int,
        num_external_tokens: int,
    ) -> list[tuple[int, int]]:
        total = int(num_computed_tokens) + int(num_external_tokens)
        if total <= 0:
            return []
        scores = self._fetch_scores(request)
        if scores is None:
            return []
        return scores_to_gap_intervals(
            scores,
            total_len=total,
            recomp_ratio=self.recomp_ratio,
            min_tokens=self.min_tokens,
            gap_threshold=self.gap_threshold,
            max_num_spans=self.max_num_spans,
        )

    def _fetch_scores(self, request: "Request") -> torch.Tensor | None:
        if self.score_provider is not None:
            return self.score_provider(request)
        params = getattr(request, "kv_transfer_params", None)
        if not params:
            return None
        val = params.get(self.score_key)
        if val is None:
            return None
        if not isinstance(val, torch.Tensor):
            val = torch.as_tensor(val)
        return val


def prophet_kv_policy_from_env(
    policy_name: str | None = None,
    policy_config: dict[str, Any] | None = None,
    default_block_size: int | None = None,
) -> GapPolicy | None:
    """Build a gap policy honoring CLI/config first, env var second.

    When ``VLLM_V1_SPANS_PROPHET_KV_ENABLED=True`` and no explicit
    `policy_name` was set, returns a ``ProphetKVGapPolicy`` constructed
    from the ``VLLM_V1_SPANS_PROPHET_KV_*`` env vars; otherwise defers
    to ``GapPolicyFactory.create_policy``. Intended call site: the
    scheduler's __init__, replacing direct GapPolicyFactory.create_policy.
    """
    if policy_name is None and envs.VLLM_V1_SPANS_PROPHET_KV_ENABLED:
        policy_name = "prophet_kv"
        policy_config = {
            "recomp_ratio": envs.VLLM_V1_SPANS_PROPHET_KV_RATIO,
            "min_tokens": envs.VLLM_V1_SPANS_PROPHET_KV_MIN_TOKENS,
            "gap_threshold": envs.VLLM_V1_SPANS_PROPHET_KV_GAP_THRESHOLD,
            "max_num_spans": envs.VLLM_V1_SPANS_PROPHET_KV_MAX_NUM_SPANS,
            "block_size": max(1, envs.VLLM_V1_SPANS_PROPHET_KV_BLOCK_SIZE),
            "score_key": envs.VLLM_V1_SPANS_PROPHET_KV_SCORE_KEY,
        }
    return GapPolicyFactory.create_policy(policy_name, policy_config)


# Auto-register so GapPolicyFactory.create_policy("prophet_kv", ...) works
# without callers having to remember an explicit register step.
GapPolicyFactory.register_policy("prophet_kv", ProphetKVGapPolicy)
