"""Kimi-K3 DSpark PP placement and activation ABI contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import tokenspeed.runtime.models.kimi_k3 as kimi_k3_module
from tokenspeed.runtime.layers.vocab_parallel_embedding import ParallelLMHead
from tokenspeed.runtime.models.kimi_k3 import KimiK3ForConditionalGeneration
from tokenspeed.runtime.pipeline.adapters.kimi_k3 import (
    build_balanced_kimi_k3_pipeline_plan,
)
from tokenspeed.runtime.pipeline.kimi_k3_dspark import (
    KimiK3DSparkStageProjector,
    resolve_kimi_k3_dspark_placement,
    split_kimi_k3_dspark_context_projection,
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
        assert stage.output_schema.fields[-1].dtype == "float32"
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


def test_k3_dspark_stage_projections_match_the_full_context_linear() -> None:
    placement = resolve_kimi_k3_dspark_placement(
        _k3_pp8_plan(dspark_context_hidden_size=5),
        target_layer_ids=[2, 23, 47, 71, 89],
        target_hidden_size=3,
        context_hidden_size=5,
    )
    full_weight = torch.arange(5 * 15, dtype=torch.float32).reshape(5, 15)
    target_hidden = {
        layer_id: torch.full((2, 3), float(index + 1))
        for index, layer_id in enumerate(placement.target_layer_ids)
    }
    context = torch.zeros((2, 5), dtype=torch.float32)
    weights_by_stage = split_kimi_k3_dspark_context_projection(full_weight, placement)

    for stage_id in range(8):
        projector = KimiK3DSparkStageProjector(
            placement,
            stage_id=stage_id,
            projection_weights=weights_by_stage.get(stage_id, {}),
        )
        for slice_ in placement.projection_slices_for_stage(stage_id):
            projector.accumulate(
                context,
                target_layer_id=slice_.target_layer_id,
                target_hidden=target_hidden[slice_.target_layer_id],
            )

    expected = torch.mm(
        torch.cat(
            [target_hidden[layer_id] for layer_id in placement.target_layer_ids],
            dim=1,
        ),
        full_weight.t(),
    )
    torch.testing.assert_close(context, expected)


def test_k3_dspark_stage_projector_rejects_non_local_taps() -> None:
    placement = resolve_kimi_k3_dspark_placement(
        _k3_pp8_plan(dspark_context_hidden_size=4),
        target_layer_ids=[2, 23, 47, 71, 89],
        target_hidden_size=2,
        context_hidden_size=4,
    )
    full_weight = torch.zeros((4, 10), dtype=torch.float32)
    projector = KimiK3DSparkStageProjector(
        placement,
        stage_id=0,
        projection_weights=split_kimi_k3_dspark_context_projection(
            full_weight, placement
        )[0],
    )
    with pytest.raises(ValueError, match="does not own"):
        projector.accumulate(
            torch.zeros((1, 4), dtype=torch.float32),
            target_layer_id=23,
            target_hidden=torch.zeros((1, 2), dtype=torch.float32),
        )


def _k3_final_stage_with_head(head: ParallelLMHead):
    model = KimiK3ForConditionalGeneration.__new__(
        KimiK3ForConditionalGeneration
    )
    torch.nn.Module.__init__(model)
    model.mapping = SimpleNamespace(
        pipeline=SimpleNamespace(is_last_stage=True)
    )
    model.language_model = SimpleNamespace(lm_head=head)
    return model


def test_k3_dspark_exports_bf16_head_under_global_quantization() -> None:
    head = ParallelLMHead(
        64,
        16,
        params_dtype=torch.bfloat16,
        quant_config=object(),
        tp_rank=0,
        tp_size=1,
        tp_group=(0,),
    )

    weight = _k3_final_stage_with_head(
        head
    ).get_pipeline_dspark_source_head_weight(expected_dtype=torch.bfloat16)

    assert weight.data_ptr() == head.weight.data_ptr()
    assert weight.shape == head.weight.shape
    assert weight.dtype == torch.bfloat16
    assert not weight.requires_grad


def test_k3_dspark_rejects_an_actually_quantized_head() -> None:
    head = ParallelLMHead(
        64,
        16,
        params_dtype=torch.bfloat16,
        quant_config=object(),
        tp_rank=0,
        tp_size=1,
        tp_group=(0,),
    )
    head.linear_method = object()

    with pytest.raises(ValueError, match="does not support a quantized LM head"):
        _k3_final_stage_with_head(head).get_pipeline_dspark_source_head_weight(
            expected_dtype=torch.bfloat16
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_k3_dspark_installs_head_on_the_broadcast_weight_device(monkeypatch) -> None:
    model = KimiK3ForConditionalGeneration.__new__(
        KimiK3ForConditionalGeneration
    )
    torch.nn.Module.__init__(model)
    model.mapping = SimpleNamespace(
        pipeline=SimpleNamespace(is_first_stage=True),
        attn=SimpleNamespace(
            has_dp=False,
            tp_rank=0,
            tp_size=1,
            tp_group=(0,),
        ),
    )
    model.language_model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=16, vocab_size=64)
    )
    model._pipeline_dspark_lm_head = None
    model._pipeline_dspark_logits_processor = None
    monkeypatch.setattr(
        kimi_k3_module,
        "LogitsProcessor",
        lambda *args, **kwargs: object(),
    )
    source_weight = torch.arange(
        64 * 16,
        device="cuda",
        dtype=torch.bfloat16,
    ).reshape(64, 16)

    model.install_pipeline_dspark_draft_head(source_weight)

    installed_weight = model._pipeline_dspark_lm_head.weight
    assert installed_weight.device == source_weight.device
    torch.testing.assert_close(installed_weight, source_weight)
    hidden_states = torch.ones((2, 16), device="cuda", dtype=torch.bfloat16)
    logits = torch.matmul(hidden_states, installed_weight.T)
    assert logits.shape == (2, 64)
    assert logits.device == source_weight.device
