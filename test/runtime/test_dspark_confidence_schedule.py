"""CPU coverage for the DSpark confidence scheduler contract."""

from __future__ import annotations

import json

import pytest
import torch

from tokenspeed.runtime.execution.drafter.dspark_schedule import (
    build_sps_table,
    calibrate_confidence_logits,
    load_dspark_schedule_profile,
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
        torch.tensor([100.0, 100.0, 100.0, 90.0, 80.0, 70.0, 60.0, 60.0, 60.0]),
    )


def test_schedule_profile_materializes_calibrated_device_tensors(tmp_path) -> None:
    profile_path = tmp_path / "dspark-profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "version": 1,
                "candidate_count": 3,
                "sts_temperatures": [1.0, 1.5, 2.0],
                "throughput": {
                    "token_points": [2, 6],
                    "steps_per_second": [100.0, 60.0],
                },
            }
        )
    )

    profile = load_dspark_schedule_profile(profile_path, candidate_count=3)
    temperatures, throughput = profile.materialize(max_tokens=8, device="cpu")

    assert temperatures.tolist() == [1.0, 1.5, 2.0]
    torch.testing.assert_close(
        throughput,
        torch.tensor([100.0, 100.0, 100.0, 90.0, 80.0, 70.0, 60.0, 60.0, 60.0]),
    )


def test_schedule_profile_materializes_exact_batch_sps_tables(tmp_path) -> None:
    profile_path = tmp_path / "batch-profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "version": 1,
                "candidate_count": 2,
                "sts_temperatures": [1.0, 1.0],
                "throughput": {
                    "token_points": [1, 4],
                    "steps_per_second": [100.0, 80.0],
                    "by_batch_size": {
                        "1": {
                            "token_points": [1, 2],
                            "steps_per_second": [100.0, 40.0],
                        }
                    },
                },
            }
        )
    )

    profile = load_dspark_schedule_profile(profile_path, candidate_count=2)
    batch_tables = profile.materialize_batch_throughput(max_tokens=4, device="cpu")

    torch.testing.assert_close(
        batch_tables[1], torch.tensor([100.0, 100.0, 40.0, 40.0, 40.0])
    )


def test_schedule_profile_rejects_another_verify_width(tmp_path) -> None:
    profile_path = tmp_path / "wrong-width.json"
    profile_path.write_text(
        json.dumps(
            {
                "version": 1,
                "candidate_count": 2,
                "sts_temperatures": [1.0, 1.0],
                "throughput": {
                    "token_points": [1],
                    "steps_per_second": [1.0],
                },
            }
        )
    )

    with pytest.raises(ValueError, match="candidate_count"):
        load_dspark_schedule_profile(profile_path, candidate_count=3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_confidence_scheduler_runs_on_cuda() -> None:
    logits = torch.full((2, 3), 10.0, device="cuda")
    profile = torch.ones(16, device="cuda")

    widths = schedule_prefix_lengths(logits, profile).to(torch.int32).add_(1)
    torch.cuda.synchronize()

    assert widths.cpu().tolist() == [4, 4]
