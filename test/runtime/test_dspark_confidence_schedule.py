"""CPU coverage for the DSpark confidence scheduler contract."""

from __future__ import annotations

import torch

from tokenspeed.runtime.execution.drafter.dspark_schedule import (
    build_sps_table,
    calibrate_confidence_logits,
    schedule_prefix_lengths,
    survival_probabilities,
)


def test_calibration_applies_temperature_to_logits() -> None:
    logits = torch.tensor([[0.0, 1.3862944]])
    calibrated = calibrate_confidence_logits(logits, torch.tensor([1.0, 2.0]))

    torch.testing.assert_close(calibrated, torch.tensor([[0.5, 2.0 / 3.0]]))


def test_survival_probabilities_are_prefix_products() -> None:
    confidence = torch.tensor([[0.8, 0.5, 0.25]])

    torch.testing.assert_close(
        survival_probabilities(confidence), torch.tensor([[0.8, 0.4, 0.1]])
    )


def test_scheduler_keeps_full_block_when_throughput_is_flat() -> None:
    logits = torch.full((2, 3), 10.0)
    profile = torch.ones(16)

    lengths = schedule_prefix_lengths(logits, profile)

    assert lengths.tolist() == [3, 3]


def test_scheduler_stops_at_first_objective_drop() -> None:
    logits = torch.tensor([[2.1972246, 1.3862944, -2.1972246]])
    # B=1: 1.0, B=2: 0.7, B=3: 0.4. The first candidate improves
    # expected throughput, while the second decreases it.
    profile = torch.tensor([1.0, 1.0, 0.7, 0.4, 0.3])

    lengths = schedule_prefix_lengths(logits, profile)

    assert lengths.tolist() == [1]


def test_scheduler_respects_prefixes_when_scores_tie() -> None:
    logits = torch.zeros((1, 3))
    profile = torch.ones(8)

    lengths = schedule_prefix_lengths(logits, profile)

    assert lengths.tolist() == [3]


def test_scheduler_admits_across_requests_by_survival_probability() -> None:
    logits = torch.logit(torch.tensor([[0.5, 0.5], [0.9, 0.1]]))
    # With two anchors, the first two candidate admissions maximize throughput.
    profile = torch.tensor([1.0, 1.0, 1.0, 0.9, 0.8, 0.5, 0.4])

    lengths = schedule_prefix_lengths(logits, profile)

    assert lengths.tolist() == [1, 1]


def test_build_sps_table_interpolates_between_profiled_points() -> None:
    table = build_sps_table([2, 6], [100.0, 60.0], max_tokens=8)

    torch.testing.assert_close(
        table,
        torch.tensor(
            [100.0, 100.0, 100.0, 90.0, 80.0, 70.0, 60.0, 60.0, 60.0]
        ),
    )
