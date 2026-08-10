from __future__ import annotations

import os
import sys

_TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(_TEST_DIR))

from test.runtime.conftest import TP8_PAGE_SET_BYTES

import pytest
import torch

from tokenspeed.runtime.configs.kimi_k3_config import KimiLinearConfig
from tokenspeed.runtime.layers.attention.kv_cache.recipes.kimi_k3 import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    _kimi_k3_global_cache_field_dtypes,
    build_kimi_k3_logical_cache_fields,
    kimi_k3_layer_group_ids,
    kimi_k3_lcm_blocks_needed,
    kimi_k3_token_capacity_for_cache_pool,
    solve_kimi_k3_cache_layout,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.stage_layout import (
    CacheStagePlacement,
    pipeline_cache_abi_digest,
)
from tokenspeed.runtime.pipeline.adapters.kimi_k3 import (
    build_balanced_kimi_k3_pipeline_plan,
)


def _plan(num_lcm_blocks: int, *, tp_size: int = 8):
    layout = solve_kimi_k3_cache_layout(
        KimiLinearConfig(),
        tp_size=tp_size,
        mla_cache_dtype=torch.float8_e4m3fn,
        mla_quant_method=None,
    )
    return layout.with_num_lcm_blocks(num_lcm_blocks)


def test_lcm_reference_geometry_is_exact() -> None:
    plan = _plan(7)

    assert plan.logical_block_tokens == 128
    assert plan.lcm_block_bytes == TP8_PAGE_SET_BYTES
    assert len(plan.planes) == 24
    assert {
        group.group_id: group.cache_blocks_per_lcm_block for group in plan.groups
    } == {
        "full_attention": 12,
        "linear_attention_0": 1,
        "linear_attention_1": 1,
        "linear_attention_2": 1,
    }
    fields_by_group = {
        group_id: [field for field in plan.fields if field.group_id == group_id]
        for group_id in (
            "full_attention",
            "linear_attention_0",
            "linear_attention_1",
            "linear_attention_2",
        )
    }
    assert len(fields_by_group["full_attention"]) == 24
    assert all(
        len(fields_by_group[group_id]) == 46
        for group_id in fields_by_group
        if group_id != "full_attention"
    )
    for group_id in (
        "linear_attention_0",
        "linear_attention_1",
        "linear_attention_2",
    ):
        assert {field.plane_id for field in fields_by_group[group_id]} == {
            f"slot.{slot}" for slot in range(23)
        }


def test_pipeline_cache_dtype_abi_is_global_across_stages() -> None:
    text_config = KimiLinearConfig()
    group_ids = kimi_k3_layer_group_ids(text_config)
    global_layer_types = tuple(
        FULL_ATTENTION if group_id == FULL_ATTENTION else LINEAR_ATTENTION
        for group_id in group_ids
    )
    logical_fields = build_kimi_k3_logical_cache_fields(
        text_config,
        tp_size=8,
        mla_cache_dtype=torch.float8_e4m3fn,
        mla_quant_method=None,
    )
    global_layout = solve_kimi_k3_cache_layout(
        text_config,
        tp_size=8,
        mla_cache_dtype=torch.float8_e4m3fn,
        mla_quant_method=None,
    )
    conv_dtype = torch.bfloat16
    recurrent_dtype = torch.float32
    field_dtypes = _kimi_k3_global_cache_field_dtypes(
        logical_fields,
        global_layer_types,
        mla_cache_dtype=torch.float8_e4m3fn,
        conv_dtype=conv_dtype,
        recurrent_dtype=recurrent_dtype,
    )
    plan = build_balanced_kimi_k3_pipeline_plan(
        num_layers=text_config.num_hidden_layers,
        hidden_size=text_config.hidden_size,
        attn_res_block_size=text_config.attn_res_block_size,
        stage_count=8,
        activation_dtype="bfloat16",
    )
    placements = (
        CacheStagePlacement.from_pipeline_plan(rank=0, world_size=64, plan=plan),
        CacheStagePlacement.from_pipeline_plan(rank=8, world_size=64, plan=plan),
        CacheStagePlacement.from_pipeline_plan(rank=56, world_size=64, plan=plan),
    )

    digests = {
        pipeline_cache_abi_digest(
            logical_fields,
            placement,
            global_layout,
            field_dtypes=field_dtypes,
        )
        for placement in placements
    }

    assert len(digests) == 1
    assert len(field_dtypes) == 162
    assert sum(field_id.endswith(".latent_kv") for field_id in field_dtypes) == 24
    assert sum(field_id.endswith(".conv_state") for field_id in field_dtypes) == 69
    assert sum(field_id.endswith(".recurrent_state") for field_id in field_dtypes) == 69
    assert field_dtypes["layer.0.conv_state"] == conv_dtype
    assert field_dtypes["layer.0.recurrent_state"] == recurrent_dtype
    assert field_dtypes["layer.3.latent_kv"] == torch.float8_e4m3fn

    bf16_mla_dtypes = _kimi_k3_global_cache_field_dtypes(
        logical_fields,
        global_layer_types,
        mla_cache_dtype=torch.bfloat16,
        conv_dtype=conv_dtype,
        recurrent_dtype=recurrent_dtype,
    )
    assert bf16_mla_dtypes["layer.3.latent_kv"] == torch.bfloat16
    assert bf16_mla_dtypes["layer.0.recurrent_state"] == torch.float32

    with pytest.raises(ValueError, match="out-of-range layer id"):
        _kimi_k3_global_cache_field_dtypes(
            logical_fields,
            global_layer_types[:-1],
            mla_cache_dtype=torch.float8_e4m3fn,
            conv_dtype=conv_dtype,
            recurrent_dtype=recurrent_dtype,
        )
    with pytest.raises(ValueError, match="global cache fields for layer"):
        _kimi_k3_global_cache_field_dtypes(
            logical_fields[:-1],
            global_layer_types,
            mla_cache_dtype=torch.float8_e4m3fn,
            conv_dtype=conv_dtype,
            recurrent_dtype=recurrent_dtype,
        )


def test_lcm_geometry_packs_two_kda_pages_at_tp16() -> None:
    """KDA state halves at TP16; two pages pack per MLA-sized plane."""
    plan = _plan(7, tp_size=16)

    assert {
        group.group_id: group.cache_blocks_per_lcm_block for group in plan.groups
    } == {
        "full_attention": 12,
        "linear_attention_0": 2,
        "linear_attention_1": 2,
        "linear_attention_2": 2,
    }
    # MLA planes still dominate, so the LCM block geometry matches TP8.
    assert plan.lcm_block_bytes == TP8_PAGE_SET_BYTES
    assert len(plan.planes) == 24


def test_lcm_parent_demand_uses_per_group_packing() -> None:
    plan = _plan(300)
    sizing = dict(
        max_scheduled_tokens=8_192,
        max_live_requests=1,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )

    assert kimi_k3_lcm_blocks_needed(plan, token_capacity=131_072, **sizing) == 284
    assert (
        kimi_k3_token_capacity_for_cache_pool(
            plan,
            num_lcm_blocks=284,
            upper_bound_tokens=131_072,
            **sizing,
        )
        == 131_072
    )
    assert (
        kimi_k3_token_capacity_for_cache_pool(
            plan,
            num_lcm_blocks=283,
            upper_bound_tokens=131_072,
            **sizing,
        )
        < 131_072
    )


def test_k3_merged_solve_with_draft_shares_page_ids():
    """One big model: a draft MLA layer joins the K3 solve as continuation
    layer 93 in the full_attention group — same packing, same page-id
    space, one plan, one arena."""
    import torch

    from tokenspeed.runtime.configs.kimi_k3_config import KimiLinearConfig
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.ordinary import (
        mla_cache_fields,
    )

    draft_fields = mla_cache_fields(
        layer_group_ids=("full_attention",),
        logical_block_tokens=128,
        latent_width=576,
        element_size=1,
    )
    merged = solve_kimi_k3_cache_layout(
        KimiLinearConfig(),
        tp_size=8,
        mla_cache_dtype=torch.float8_e4m3fn,
        mla_quant_method=None,
        draft_fields=draft_fields,
    )
    # 24 target MLA planes + 1 draft continuation plane.
    assert len(merged.plane_bytes) == 25
    assert dict(merged.group_packing)["full_attention"] == 12
    plan = merged.with_num_lcm_blocks(7)
    draft_field = plan.field("layer.93.latent_kv")
    target_field = plan.field("layer.3.latent_kv")
    assert draft_field.group_id == target_field.group_id == "full_attention"
    assert draft_field.page_stride_bytes == target_field.page_stride_bytes
    # One group -> one page-id space: same page_count by identity.
    assert plan.group("full_attention").page_count == 1 + 7 * 12


def test_k3_binding_utilization_baseline_and_draft_widening():
    """Binding-hole metric on real K3 geometry: full bindings use
    the whole parent; state bindings use 88.2%, dropping to ~84.7% when a
    1-layer draft plane widens the parent (naive join)."""
    import torch

    from tokenspeed.runtime.configs.kimi_k3_config import KimiLinearConfig
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.ordinary import (
        mla_cache_fields,
    )

    base = solve_kimi_k3_cache_layout(
        KimiLinearConfig(),
        tp_size=8,
        mla_cache_dtype=torch.float8_e4m3fn,
        mla_quant_method=None,
    ).with_num_lcm_blocks(10)
    report = base.capacity_report()
    assert abs(report["full_attention"]["binding_utilization"] - 1.0) < 1e-3
    for k in range(3):
        assert (
            abs(report[f"linear_attention_{k}"]["binding_utilization"] - 0.882) < 1e-3
        )

    draft_fields = mla_cache_fields(
        layer_group_ids=("full_attention",),
        logical_block_tokens=128,
        latent_width=576,
        element_size=1,
    )
    merged = solve_kimi_k3_cache_layout(
        KimiLinearConfig(),
        tp_size=8,
        mla_cache_dtype=torch.float8_e4m3fn,
        mla_quant_method=None,
        draft_fields=draft_fields,
    ).with_num_lcm_blocks(10)
    widened = merged.capacity_report()
    assert abs(widened["full_attention"]["binding_utilization"] - 1.0) < 1e-3
    assert abs(widened["linear_attention_0"]["binding_utilization"] - 0.847) < 1e-3
