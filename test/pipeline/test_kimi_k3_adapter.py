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

import pytest

from tokenspeed.runtime.pipeline.adapters.kimi_k3 import (
    balanced_kimi_k3_stage_layer_counts,
    build_balanced_kimi_k3_pipeline_plan,
    build_kimi_k3_pipeline_plan,
    kimi_k3_stage_checkpoint_weight_filter,
)


def _k3_pp4_plan():
    return build_kimi_k3_pipeline_plan(
        num_layers=93,
        hidden_size=7168,
        attn_res_block_size=12,
        stage_layer_counts=(24, 24, 24, 21),
    )


def test_k3_pp4_plan_has_expected_layers_ownership_and_payloads():
    plan = _k3_pp4_plan()

    assert [(stage.first_layer, stage.end_layer) for stage in plan.stages] == [
        (0, 24),
        (24, 48),
        (48, 72),
        (72, 93),
    ]
    assert [stage.owns_embedding for stage in plan.stages] == [
        True,
        False,
        False,
        False,
    ]
    assert [stage.owns_head for stage in plan.stages] == [False, False, False, True]
    assert [len(stage.output_schema.fields) for stage in plan.stages[:-1]] == [3, 5, 7]

    for stage in plan.stages[:-1]:
        schema = stage.output_schema
        assert schema.fields[0].field_id == "prefix_sum"
        assert all(field.dtype == "bfloat16" for field in schema.fields)
        assert all(field.trailing_shape == (7168,) for field in schema.fields)
        assert plan.stages[stage.stage_id + 1].input_schema.digest == schema.digest


def test_k3_plan_digest_is_stable_and_changes_with_dtype():
    first = _k3_pp4_plan()
    second = _k3_pp4_plan()
    fp16 = build_kimi_k3_pipeline_plan(
        num_layers=93,
        hidden_size=7168,
        attn_res_block_size=12,
        stage_layer_counts=(24, 24, 24, 21),
        activation_dtype="float16",
    )

    assert first.digest == second.digest
    assert first.digest != fp16.digest


def test_k3_boundary_digest_includes_attn_res_semantics():
    block_12 = build_kimi_k3_pipeline_plan(
        num_layers=48,
        hidden_size=128,
        attn_res_block_size=12,
        stage_layer_counts=(24, 24),
    )
    block_8 = build_kimi_k3_pipeline_plan(
        num_layers=32,
        hidden_size=128,
        attn_res_block_size=8,
        stage_layer_counts=(16, 16),
    )

    assert len(block_12.stages[0].output_schema.fields) == 3
    assert len(block_8.stages[0].output_schema.fields) == 3
    assert (
        block_12.stages[0].output_schema.digest
        != block_8.stages[0].output_schema.digest
    )


def test_k3_single_stage_uses_the_generic_empty_boundary_contract():
    plan = build_kimi_k3_pipeline_plan(
        num_layers=93,
        hidden_size=7168,
        attn_res_block_size=12,
        stage_layer_counts=(93,),
    )

    assert len(plan.stages) == 1
    assert plan.stages[0].input_schema is None
    assert plan.stages[0].output_schema is None


@pytest.mark.parametrize(
    ("stage_count", "expected"),
    [
        (1, (93,)),
        (2, (48, 45)),
        (4, (24, 24, 24, 21)),
        (8, (12, 12, 12, 12, 12, 12, 12, 9)),
    ],
)
def test_balanced_k3_partitions_use_nearest_attn_res_boundaries(stage_count, expected):
    assert (
        balanced_kimi_k3_stage_layer_counts(
            num_layers=93,
            attn_res_block_size=12,
            stage_count=stage_count,
        )
        == expected
    )

    plan = build_balanced_kimi_k3_pipeline_plan(
        num_layers=93,
        hidden_size=7168,
        attn_res_block_size=12,
        stage_count=stage_count,
    )
    assert (
        tuple(stage.end_layer - stage.first_layer for stage in plan.stages) == expected
    )


def test_balanced_k3_partition_rejects_more_stages_than_legal_boundaries():
    with pytest.raises(ValueError, match="cannot place"):
        balanced_kimi_k3_stage_layer_counts(
            num_layers=93,
            attn_res_block_size=12,
            stage_count=9,
        )


def test_checkpoint_filter_selects_only_stage_owned_k3_weights():
    plan = build_balanced_kimi_k3_pipeline_plan(
        num_layers=93,
        hidden_size=7168,
        attn_res_block_size=12,
        stage_count=8,
    )
    first = plan.stages[0]
    middle = plan.stages[3]
    last = plan.stages[-1]

    assert kimi_k3_stage_checkpoint_weight_filter(
        "vision_tower.patch_embed.weight",
        stage_plan=first,
        include_vision=True,
    )
    assert kimi_k3_stage_checkpoint_weight_filter(
        "language_model.model.embed_tokens.weight",
        stage_plan=first,
        include_vision=True,
    )
    assert not kimi_k3_stage_checkpoint_weight_filter(
        "language_model.model.layers.12.self_attn.q_proj.weight",
        stage_plan=first,
        include_vision=True,
    )
    assert kimi_k3_stage_checkpoint_weight_filter(
        "language_model.model.layers.36.self_attn.q_proj.weight",
        stage_plan=middle,
        include_vision=False,
    )
    assert not kimi_k3_stage_checkpoint_weight_filter(
        "language_model.model.embed_tokens.weight",
        stage_plan=middle,
        include_vision=False,
    )
    assert kimi_k3_stage_checkpoint_weight_filter(
        "language_model.model.norm.weight",
        stage_plan=last,
        include_vision=False,
    )
    assert kimi_k3_stage_checkpoint_weight_filter(
        "language_model.lm_head.weight",
        stage_plan=last,
        include_vision=False,
    )


@pytest.mark.parametrize(
    ("counts", "message"),
    [
        ((24, 24, 20, 25), "does not align"),
        ((24, 24, 24, 20), "sum to"),
        ((24, 0, 48, 21), "positive integer"),
        ((), "must not be empty"),
    ],
)
def test_k3_plan_rejects_invalid_layer_partitions(counts, message):
    with pytest.raises(ValueError, match=message):
        build_kimi_k3_pipeline_plan(
            num_layers=93,
            hidden_size=7168,
            attn_res_block_size=12,
            stage_layer_counts=counts,
        )
