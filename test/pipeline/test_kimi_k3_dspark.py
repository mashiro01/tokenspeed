"""Kimi-K3 DSpark PP placement and activation ABI contracts."""

from __future__ import annotations

import pytest

from tokenspeed.runtime.pipeline.adapters.kimi_k3 import (
    build_balanced_kimi_k3_pipeline_plan,
)
from tokenspeed.runtime.pipeline.kimi_k3_dspark import (
    resolve_kimi_k3_dspark_placement,
)


def _k3_pp8_plan(*, dspark_context_hidden_size: int | None = None):
    return build_balanced_kimi_k3_pipeline_plan(
        num_layers=93,
        hidden_size=7168,
        attn_res_block_size=12,
        stage_count=8,
        dspark_context_hidden_size=dspark_context_hidden_size,
    )


def test_k3_dspark_uses_one_projected_context_stream_per_pp_boundary() -> None:
    baseline = _k3_pp8_plan()
    dspark = _k3_pp8_plan(dspark_context_hidden_size=7168)

    assert baseline.digest != dspark.digest
    assert [field.field_id for field in baseline.stages[0].output_schema.fields] == [
        "prefix_sum",
        "block_residual.0",
    ]
    for stage in dspark.stages[:-1]:
        assert stage.output_schema.fields[-1].field_id == "dspark_context"
        assert stage.output_schema.fields[-1].trailing_shape == (7168,)
        assert all(
            not field.field_id.startswith("aux_capture.")
            for field in stage.output_schema.fields
        )


def test_k3_dspark_real_taps_are_projected_on_their_owner_stages() -> None:
    placement = resolve_kimi_k3_dspark_placement(
        _k3_pp8_plan(dspark_context_hidden_size=7168),
        target_layer_ids=[2, 23, 47, 71, 89],
        target_hidden_size=7168,
        context_hidden_size=7168,
    )

    assert placement.draft_owner_stage_id == 0
    assert placement.verify_stage_id == 7
    assert [
        (
            slice_.target_layer_id,
            slice_.stage_id,
            slice_.input_start,
            slice_.input_end,
        )
        for slice_ in placement.projection_slices
    ] == [
        (2, 0, 0, 7168),
        (23, 1, 7168, 14336),
        (47, 3, 14336, 21504),
        (71, 5, 21504, 28672),
        (89, 7, 28672, 35840),
    ]
    assert [
        tuple(
            slice_.target_layer_id
            for slice_ in placement.projection_slices_for_stage(stage_id)
        )
        for stage_id in range(8)
    ] == [(2,), (23,), (), (47,), (), (71,), (), (89,)]


@pytest.mark.parametrize(
    ("target_layer_ids", "message"),
    [
        ([23, 2], "sorted"),
        ([2, 2], "unique"),
        ([2, 93], "outside"),
    ],
)
def test_k3_dspark_rejects_invalid_tap_ownership(target_layer_ids, message) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_kimi_k3_dspark_placement(
            _k3_pp8_plan(),
            target_layer_ids=target_layer_ids,
            target_hidden_size=7168,
            context_hidden_size=7168,
        )


def test_k3_dspark_rejects_a_draft_owner_without_target_embeddings() -> None:
    with pytest.raises(ValueError, match="stage 0"):
        resolve_kimi_k3_dspark_placement(
            _k3_pp8_plan(),
            target_layer_ids=[2, 23, 47, 71, 89],
            target_hidden_size=7168,
            context_hidden_size=7168,
            draft_owner_stage_id=7,
        )


def test_k3_dspark_context_width_must_be_positive() -> None:
    with pytest.raises(ValueError, match="dspark_context_hidden_size"):
        _k3_pp8_plan(dspark_context_hidden_size=0)
