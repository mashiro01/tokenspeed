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

"""Confidence-scheduled DSpark verification primitives.

The draft confidence head predicts conditional acceptance probability for each
candidate. This module turns those predictions into a device-resident number of
draft tokens to verify per request. It deliberately has no model, cache, or
scheduler dependency so its numerical contract can be tested on CPU before the
ragged target-forward plumbing consumes it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class DSparkScheduleProfile:
    """Calibrated confidence and target-throughput data for DSpark.

    The profile is intentionally data-only. Keeping it separate from model
    configuration means a deployment can collect a new calibration on its
    exact hardware and enable it without changing draft weights or runtime
    code.
    """

    candidate_count: int
    sts_temperatures: tuple[float, ...]
    token_points: tuple[int, ...]
    steps_per_second: tuple[float, ...]

    def materialize(
        self,
        *,
        max_tokens: int,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return device-resident STS temperatures and SPS lookup table."""

        temperatures = torch.tensor(
            self.sts_temperatures, dtype=torch.float32, device=device
        )
        throughput = build_sps_table(
            self.token_points,
            self.steps_per_second,
            max_tokens=max_tokens,
        ).to(device=device)
        return temperatures, throughput


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def load_dspark_schedule_profile(
    path: str | Path,
    *,
    candidate_count: int,
) -> DSparkScheduleProfile:
    """Load a strict, versioned DSpark scheduling profile from JSON.

    Expected schema::

        {
          "version": 1,
          "candidate_count": 7,
          "sts_temperatures": [1.0, ...],
          "throughput": {
            "token_points": [1, 8, 16],
            "steps_per_second": [62.1, 59.4, 53.0]
          }
        }

    Rejecting a profile from another verify width is deliberate: using its
    temperatures against a differently shaped draft block would make the
    scheduling decision uncalibrated while appearing enabled.
    """

    if candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    profile_path = Path(path)
    try:
        raw = json.loads(profile_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(
            f"cannot read DSpark schedule profile {profile_path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"DSpark schedule profile {profile_path} is not valid JSON"
        ) from exc
    if not isinstance(raw, dict):
        raise ValueError("DSpark schedule profile must be a JSON object")
    if raw.get("version") != 1:
        raise ValueError("DSpark schedule profile version must be 1")
    if raw.get("candidate_count") != candidate_count:
        raise ValueError(
            "DSpark schedule profile candidate_count does not match the "
            f"runtime draft block: expected {candidate_count}, got "
            f"{raw.get('candidate_count')!r}"
        )

    raw_temperatures = raw.get("sts_temperatures")
    if (
        not isinstance(raw_temperatures, list)
        or len(raw_temperatures) != candidate_count
    ):
        raise ValueError(
            "DSpark schedule profile sts_temperatures must have one value per "
            "draft candidate"
        )
    temperatures = tuple(
        _finite_float(value, field="sts_temperatures")
        for value in raw_temperatures
    )
    if any(value <= 0.0 for value in temperatures):
        raise ValueError("DSpark schedule profile temperatures must be positive")

    throughput = raw.get("throughput")
    if not isinstance(throughput, dict):
        raise ValueError("DSpark schedule profile throughput must be an object")
    raw_points = throughput.get("token_points")
    raw_rates = throughput.get("steps_per_second")
    if not isinstance(raw_points, list) or not isinstance(raw_rates, list):
        raise ValueError(
            "DSpark schedule profile throughput must contain token_points and "
            "steps_per_second lists"
        )
    if not raw_points or len(raw_points) != len(raw_rates):
        raise ValueError(
            "DSpark schedule profile throughput lists must be non-empty and "
            "equal length"
        )
    if any(
        isinstance(point, bool) or not isinstance(point, int)
        for point in raw_points
    ):
        raise ValueError("DSpark schedule profile token_points must be integers")
    points = tuple(int(point) for point in raw_points)
    if any(point < 0 for point in points) or any(
        left >= right for left, right in zip(points, points[1:])
    ):
        raise ValueError(
            "DSpark schedule profile token_points must be strictly increasing "
            "non-negative integers"
        )
    rates = tuple(
        _finite_float(value, field="throughput.steps_per_second")
        for value in raw_rates
    )
    if any(rate <= 0.0 for rate in rates):
        raise ValueError(
            "DSpark schedule profile throughput rates must be positive"
        )
    return DSparkScheduleProfile(
        candidate_count=candidate_count,
        sts_temperatures=temperatures,
        token_points=points,
        steps_per_second=rates,
    )


def calibrate_confidence_logits(
    confidence_logits: torch.Tensor,
    sts_temperatures: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return STS-calibrated conditional acceptance probabilities.

    Sequential temperature scaling is applied to each logit, not to an already
    squashed probability. A missing temperature vector is exactly the
    uncalibrated sigmoid path and is suitable for shadow collection.
    """

    if confidence_logits.ndim != 2:
        raise ValueError(
            "confidence_logits must have shape [requests, candidates], got "
            f"{tuple(confidence_logits.shape)}"
        )
    logits = confidence_logits.float()
    if sts_temperatures is None:
        return torch.sigmoid(logits)
    temperatures = sts_temperatures.to(device=logits.device, dtype=logits.dtype)
    if temperatures.ndim != 1 or temperatures.numel() != logits.shape[1]:
        raise ValueError(
            "sts_temperatures must have shape [candidates], got "
            f"{tuple(temperatures.shape)} for {logits.shape[1]} candidates"
        )
    if torch.any(temperatures <= 0):
        raise ValueError("sts_temperatures must be strictly positive")
    return torch.sigmoid(logits / temperatures)


def survival_probabilities(confidence: torch.Tensor) -> torch.Tensor:
    """Return prefix-survival probabilities for conditional confidence values."""

    if confidence.ndim != 2:
        raise ValueError(
            "confidence must have shape [requests, candidates], got "
            f"{tuple(confidence.shape)}"
        )
    return confidence.float().clamp_(0.0, 1.0).cumprod(dim=1)


def build_sps_table(
    token_points: Sequence[int],
    steps_per_second: Sequence[float],
    max_tokens: int,
) -> torch.Tensor:
    """Linearly interpolate a target-forward throughput profile."""

    if len(token_points) != len(steps_per_second):
        raise ValueError("token_points and steps_per_second must have equal length")
    if not token_points:
        raise ValueError("at least one throughput sample is required")
    if max_tokens < 0:
        raise ValueError("max_tokens must be non-negative")
    points = sorted(zip(token_points, steps_per_second))
    if any(point < 0 for point, _ in points):
        raise ValueError("throughput token points must be non-negative")
    if any(rate <= 0 for _, rate in points):
        raise ValueError("throughput samples must be positive")

    table = torch.empty(max_tokens + 1, dtype=torch.float32)
    index = 0
    for tokens in range(max_tokens + 1):
        if tokens <= points[0][0]:
            table[tokens] = points[0][1]
        elif tokens >= points[-1][0]:
            table[tokens] = points[-1][1]
        else:
            while index + 1 < len(points) and tokens > points[index + 1][0]:
                index += 1
            left_tokens, left_rate = points[index]
            right_tokens, right_rate = points[index + 1]
            fraction = (tokens - left_tokens) / (right_tokens - left_tokens)
            table[tokens] = left_rate + fraction * (right_rate - left_rate)
    return table


def schedule_prefix_lengths(
    confidence_logits: torch.Tensor,
    steps_per_second: torch.Tensor,
    *,
    sts_temperatures: torch.Tensor | None = None,
    early_stop: bool = True,
) -> torch.Tensor:
    """Choose a lossless per-request DSpark candidate prefix length.

    ``lengths[r]`` is the number of draft candidates to include after request
    ``r``'s anchor. The target verifier then receives ``1 + lengths[r]`` rows.
    Truncation changes only speculative work; exact target verification stays
    unchanged.
    """

    if steps_per_second.ndim != 1 or steps_per_second.numel() == 0:
        raise ValueError("steps_per_second must be a non-empty rank-1 tensor")
    requests, candidates = confidence_logits.shape
    if requests == 0 or candidates == 0:
        return torch.zeros(
            requests, dtype=torch.int64, device=confidence_logits.device
        )

    conditional = calibrate_confidence_logits(confidence_logits, sts_temperatures)
    survival = survival_probabilities(conditional)
    flat_survival = survival.reshape(-1)
    total_candidates = flat_survival.numel()
    request_ids = (
        torch.arange(requests, device=confidence_logits.device)
        .view(requests, 1)
        .expand(requests, candidates)
        .reshape(-1)
    )
    # Stable ordering preserves intra-request prefix order under equal scores.
    sort_indices = torch.argsort(flat_survival, descending=True, stable=True)
    sorted_survival = flat_survival[sort_indices]
    sorted_requests = request_ids[sort_indices]

    profile = steps_per_second.to(
        device=confidence_logits.device, dtype=torch.float32
    )
    admitted_count = torch.arange(
        total_candidates + 1, device=confidence_logits.device
    )
    expected_tokens = requests + torch.cat(
        [sorted_survival.new_zeros(1), sorted_survival.cumsum(dim=0)]
    )
    profile_indices = (requests + admitted_count).clamp(max=profile.numel() - 1)
    objective = expected_tokens * profile[profile_indices]

    if early_stop:
        non_increase = objective[1:] <= objective[:-1]
        has_non_increase = non_increase.any()
        first_non_increase = non_increase.to(torch.int64).argmax()
        all_candidates = torch.full_like(first_non_increase, total_candidates)
        selected_count = torch.where(
            has_non_increase, first_non_increase, all_candidates
        )
    else:
        selected_count = objective.argmax()

    selected = torch.arange(total_candidates, device=confidence_logits.device)
    selected = selected < selected_count
    lengths = torch.zeros(
        requests, dtype=torch.int64, device=confidence_logits.device
    )
    lengths.scatter_add_(0, sorted_requests, selected.to(torch.int64))
    return lengths
