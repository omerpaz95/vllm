# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

import vllm.envs as envs
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.gap_policy import (
    GapPolicyFactory,
    NoGapPolicy,
    SpanAwareGapPolicy,
)
from vllm.v1.request import Request

pytestmark = pytest.mark.spans


def make_span_request(
    prompt_len: int,
    span_starts: list[int] | None = None,
    cross_span_starts: list[int] | None = None,
) -> Request:
    extra_args = {}
    if span_starts is not None:
        extra_args["span_starts"] = span_starts
    if cross_span_starts is not None:
        extra_args["cross_span_starts"] = cross_span_starts

    sampling_params = SamplingParams(
        max_tokens=17,
        extra_args=extra_args if extra_args else None,
    )
    sampling_params.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id="gap_test",
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=sampling_params,
        pooling_params=None,
    )


class TestSpanAwareGapPolicy:
    def setup_method(self):
        self._original = envs.VLLM_V1_SPANS_ENABLED
        envs.VLLM_V1_SPANS_ENABLED = True

    def teardown_method(self):
        envs.VLLM_V1_SPANS_ENABLED = self._original

    def test_gaps_at_span_starts(self):
        policy = SpanAwareGapPolicy(gap_length=32)
        req = make_span_request(256, span_starts=[64, 128])
        gaps = policy.get_gaps(req, num_computed_tokens=256, num_external_tokens=0)
        assert gaps == [(64, 96), (128, 160)]

    def test_no_gaps_when_no_span_starts(self):
        policy = SpanAwareGapPolicy(gap_length=32)
        req = make_span_request(256, span_starts=None)
        gaps = policy.get_gaps(req, num_computed_tokens=256, num_external_tokens=0)
        assert gaps == []

    def test_gap_clamped_to_next_span(self):
        policy = SpanAwareGapPolicy(gap_length=100)
        req = make_span_request(256, span_starts=[64, 128])
        gaps = policy.get_gaps(req, num_computed_tokens=256, num_external_tokens=0)
        assert gaps == [(64, 128), (128, 228)]

    def test_gap_clamped_to_computed_tokens(self):
        policy = SpanAwareGapPolicy(gap_length=100)
        req = make_span_request(256, span_starts=[200])
        gaps = policy.get_gaps(req, num_computed_tokens=256, num_external_tokens=0)
        assert gaps == [(200, 256)]

    def test_zero_gap_length_disables(self):
        policy = SpanAwareGapPolicy(gap_length=0)
        req = make_span_request(256, span_starts=[64])
        gaps = policy.get_gaps(req, num_computed_tokens=256, num_external_tokens=0)
        assert gaps == []

    def test_span_starts_beyond_computed_tokens_ignored(self):
        policy = SpanAwareGapPolicy(gap_length=32)
        req = make_span_request(256, span_starts=[64, 200])
        gaps = policy.get_gaps(req, num_computed_tokens=100, num_external_tokens=0)
        assert gaps == [(64, 96)]

    def test_no_gap_policy_returns_empty(self):
        policy = NoGapPolicy()
        req = make_span_request(256, span_starts=[64])
        gaps = policy.get_gaps(req, num_computed_tokens=256, num_external_tokens=0)
        assert gaps == []

    def test_factory_creates_span_aware(self):
        policy = GapPolicyFactory.create_policy("span_aware", {"gap_length": 64})
        assert isinstance(policy, SpanAwareGapPolicy)
        assert policy.gap_length == 64
