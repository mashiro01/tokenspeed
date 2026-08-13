# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import argparse

import pytest
from tokenspeed_kernel.benchmark.cutlass_mxfp4_tactic_sweep import (
    _best_valid,
    _parse_buckets,
    _select_robust_bucket,
)


def test_parse_buckets_sorts_and_deduplicates() -> None:
    assert _parse_buckets("16,1,4,16") == (1, 4, 16)


@pytest.mark.parametrize("raw", ["", "0,1", "-1,2"])
def test_parse_buckets_rejects_nonpositive_values(raw: str) -> None:
    with pytest.raises((argparse.ArgumentTypeError, ValueError)):
        _parse_buckets(raw)


def test_best_valid_ignores_failed_measurements() -> None:
    measurements = [
        {"status": "error", "profile_ids": [0, -1]},
        {"status": "ok", "profile_ids": [1, -1], "latency_us": 8.0},
        {"status": "ok", "profile_ids": [2, -1], "latency_us": 7.0},
    ]
    assert _best_valid(measurements)["profile_ids"] == [2, -1]


def test_best_valid_rejects_empty_valid_set() -> None:
    with pytest.raises(RuntimeError, match="no valid tactic"):
        _best_valid([{"status": "error"}])


def test_select_robust_bucket_minimizes_worst_profile_regret() -> None:
    def result(native, pairs):
        return {
            "num_tokens": 1,
            "native": {"latency_us": native},
            "pair_candidates": [
                {"status": "ok", "profile_ids": pair, "latency_us": latency}
                for pair, latency in pairs
            ],
        }

    selected = _select_robust_bucket(
        {
            "spread": result(20.0, [([1, 20], 10.0), ([2, 21], 11.0)]),
            "concentrated": result(20.0, [([1, 20], 13.0), ([2, 21], 10.0)]),
        }
    )
    assert selected["selected_profile_ids"] == [2, 21]
    assert selected["max_regret_vs_profile_optimum"] == 1.1
