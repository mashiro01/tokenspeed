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

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.layers.moe.weights.loaders import load_model_weight
from tokenspeed.runtime.models.kimi_k3_weight_contract import (
    KimiK3WeightContractError,
    LoadSlot,
    assert_loader_target,
    consumed_load_slot,
    expected_rank_local_load_slots,
    parse_moe_checkpoint_slot,
    source_tensor_contract,
    validate_source_tensor,
)


def _config():
    return SimpleNamespace(
        hidden_size=7168,
        intermediate_size=33792,
        moe_intermediate_size=3072,
        routed_expert_hidden_size=3584,
        num_shared_experts=2,
        num_experts=896,
        num_attention_heads=96,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mla_use_output_gate=True,
        linear_attn_config={"num_heads": 96, "head_dim": 128},
        quantization_config={
            "config_groups": {
                "group_0": {"weights": {"group_size": 32}},
            }
        },
    )


def _mapping(ep_size: int = 1, ep_rank: int = 0):
    return SimpleNamespace(moe=SimpleNamespace(ep_size=ep_size, ep_rank=ep_rank))


def _tiny_cpu_tensor(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    return torch.empty(1, dtype=dtype).as_strided(shape, (0,) * len(shape))


def test_load_slot_is_immutable_and_sortable() -> None:
    direct = LoadSlot("model.norm.weight", "direct", -1, "direct")
    kda = LoadSlot(
        "model.layers.0.self_attn.qkvgb_proj.weight",
        "kda",
        0,
        "q",
    )

    assert sorted([kda, direct]) == [kda, direct]
    with pytest.raises(FrozenInstanceError):
        direct.component = "other"  # type: ignore[misc]


def test_expected_slots_distinguish_all_kda_mla_and_gate_up_components() -> None:
    config = _config()
    mapping = _mapping()
    kda_target = "model.layers.0.self_attn.qkvgb_proj.weight"
    mla_target = "model.layers.92.self_attn.fused_qkv_a_proj_with_mqa.weight"
    dense_target = "model.layers.0.mlp.gate_up_proj.weight"
    shared_target = "model.layers.1.block_sparse_moe.shared_experts.gate_up_proj.weight"

    assert {
        slot.component
        for slot in expected_rank_local_load_slots(
            kda_target, config=config, mapping=mapping
        )
    } == {"q", "k", "v", "g", "f_a", "b"}
    assert {
        slot.component
        for slot in expected_rank_local_load_slots(
            mla_target, config=config, mapping=mapping
        )
    } == {"q_a", "kv_a", "g"}
    for target in (dense_target, shared_target):
        slots = expected_rank_local_load_slots(target, config=config, mapping=mapping)
        assert {slot.component for slot in slots} == {"gate", "up"}
        assert all(slot.target_name == target for slot in slots)


def test_mla_expected_slots_follow_output_gate_config() -> None:
    config = _config()
    config.mla_use_output_gate = False
    slots = expected_rank_local_load_slots(
        "model.layers.3.self_attn.fused_qkv_a_proj_with_mqa.weight",
        config=config,
        mapping=_mapping(),
    )

    assert {slot.component for slot in slots} == {"q_a", "kv_a"}


def test_contract_accepts_top_level_kimi_k3_config_shape() -> None:
    config = SimpleNamespace(text_config=_config())
    target = "model.layers.0.self_attn.qkvgb_proj.weight"
    slot = next(
        slot
        for slot in expected_rank_local_load_slots(
            target,
            config=config,
            mapping=_mapping(),
        )
        if slot.component == "b"
    )

    assert source_tensor_contract(slot, config=config).shape == (96, 7168)


def test_expected_moe_slots_cover_ep1_and_ep8_rank_ranges() -> None:
    config = _config()
    w13_target = "model.layers.1.block_sparse_moe.experts.w13_weight"
    w2_scale_target = "model.layers.1.block_sparse_moe.experts.w2_weight_scale"

    ep1 = expected_rank_local_load_slots(
        w13_target, config=config, mapping=_mapping(ep_size=1, ep_rank=0)
    )
    assert len(ep1) == 1792
    assert {slot.expert_id for slot in ep1} == set(range(896))
    assert {slot.component for slot in ep1} == {"w1", "w3"}
    assert {slot.tensor_kind for slot in ep1} == {"weight_packed"}

    ep8 = expected_rank_local_load_slots(
        w13_target, config=config, mapping=_mapping(ep_size=8, ep_rank=3)
    )
    assert len(ep8) == 224
    assert {slot.expert_id for slot in ep8} == set(range(336, 448))

    ep8_w2 = expected_rank_local_load_slots(
        w2_scale_target, config=config, mapping=_mapping(ep_size=8, ep_rank=7)
    )
    assert len(ep8_w2) == 112
    assert {slot.expert_id for slot in ep8_w2} == set(range(784, 896))
    assert {slot.component for slot in ep8_w2} == {"w2"}
    assert {slot.tensor_kind for slot in ep8_w2} == {"weight_scale"}


def test_expected_moe_slots_reject_invalid_ep_partition() -> None:
    config = _config()
    config.num_experts = 10

    with pytest.raises(KimiK3WeightContractError, match="not divisible"):
        expected_rank_local_load_slots(
            "model.layers.1.block_sparse_moe.experts.w13_weight",
            config=config,
            mapping=_mapping(ep_size=8, ep_rank=0),
        )


def test_mxfp4_contract_rejects_nonstandard_group_size() -> None:
    config = _config()
    config.quantization_config["config_groups"]["group_0"]["weights"]["group_size"] = 16
    slot = parse_moe_checkpoint_slot(
        "model.layers.1.block_sparse_moe.experts.0.w1.weight_scale"
    )

    with pytest.raises(KimiK3WeightContractError, match="requires group_size=32"):
        source_tensor_contract(slot, config=config)


@pytest.mark.parametrize(
    ("checkpoint_name", "target_name", "component", "shard_id"),
    [
        (
            "language_model.model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.qkvgb_proj.weight",
            "q",
            "q",
        ),
        (
            "model.layers.92.self_attn.kv_a_proj_with_mqa.weight",
            "model.layers.92.self_attn.fused_qkv_a_proj_with_mqa.weight",
            "kv_a",
            None,
        ),
        (
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.gate_up_proj.weight",
            "gate",
            0,
        ),
        (
            "model.layers.1.block_sparse_moe.shared_experts.up_proj.weight",
            "model.layers.1.block_sparse_moe.shared_experts.gate_up_proj.weight",
            "up",
            1,
        ),
    ],
)
def test_consumed_slots_validate_checkpoint_component_and_shard(
    checkpoint_name: str,
    target_name: str,
    component: str,
    shard_id: str | int | None,
) -> None:
    slot = consumed_load_slot(
        checkpoint_name,
        target_name,
        component=component,
        shard_id=shard_id,
    )

    assert slot.target_name == target_name
    assert slot.component == component


def test_consumed_slot_rejects_wrong_component_or_layer() -> None:
    target = "model.layers.0.self_attn.qkvgb_proj.weight"
    with pytest.raises(KimiK3WeightContractError, match="caller supplied"):
        consumed_load_slot(
            "model.layers.0.self_attn.q_proj.weight",
            target,
            shard_id="k",
        )
    with pytest.raises(KimiK3WeightContractError, match="not an official source"):
        consumed_load_slot(
            "model.layers.1.self_attn.q_proj.weight",
            target,
            shard_id="q",
        )


@pytest.mark.parametrize(
    ("checkpoint_suffix", "target_suffix", "component", "tensor_kind"),
    [
        ("w1.weight_packed", "w13_weight", "w1", "weight_packed"),
        ("w3.weight_packed", "w13_weight", "w3", "weight_packed"),
        ("w1.weight_scale", "w13_weight_scale", "w1", "weight_scale"),
        ("w3.weight_scale", "w13_weight_scale", "w3", "weight_scale"),
        ("w2.weight_packed", "w2_weight", "w2", "weight_packed"),
        ("w2.weight_scale", "w2_weight_scale", "w2", "weight_scale"),
    ],
)
def test_moe_source_predicts_target_before_loader_and_checks_returned_target(
    checkpoint_suffix: str,
    target_suffix: str,
    component: str,
    tensor_kind: str,
) -> None:
    checkpoint_name = (
        "language_model.model.layers.1.block_sparse_moe.experts.17."
        f"{checkpoint_suffix}"
    )
    expected_target = "model.layers.1.block_sparse_moe.experts." f"{target_suffix}"

    slot = parse_moe_checkpoint_slot(checkpoint_name, shard_id=component)
    assert slot.target_name == expected_target
    assert slot.expert_id == 17
    assert slot.component == component
    assert slot.tensor_kind == tensor_kind
    validate_source_tensor(
        slot,
        _tiny_cpu_tensor(
            source_tensor_contract(slot, config=_config()).shape, torch.uint8
        ),
        config=_config(),
    )
    assert_loader_target(slot, expected_target)
    assert (
        consumed_load_slot(
            checkpoint_name,
            expected_target,
            shard_id=component,
        )
        == slot
    )


def test_moe_source_rejects_nonofficial_suffix_and_loader_target_mismatch() -> None:
    invalid = "model.layers.1.block_sparse_moe.experts.0.w1.weight"
    with pytest.raises(KimiK3WeightContractError, match="weight_packed"):
        parse_moe_checkpoint_slot(invalid, shard_id="w1")

    slot = parse_moe_checkpoint_slot(
        "model.layers.1.block_sparse_moe.experts.0.w1.weight_packed",
        shard_id="w1",
    )
    with pytest.raises(KimiK3WeightContractError, match="Loader target mismatch"):
        assert_loader_target(
            slot,
            "model.layers.1.block_sparse_moe.experts.w2_weight",
        )


def _validated_fusion_cases():
    cases = []
    kda_target = "model.layers.0.self_attn.qkvgb_proj.weight"
    kda_shapes = {
        "q": (12288, 7168),
        "k": (12288, 7168),
        "v": (12288, 7168),
        "g": (12288, 7168),
        "f_a": (128, 7168),
        "b": (96, 7168),
    }
    for component, source in {
        "q": "q_proj",
        "k": "k_proj",
        "v": "v_proj",
        "g": "g_proj",
        "f_a": "f_a_proj",
        "b": "b_proj",
    }.items():
        cases.append(
            (
                consumed_load_slot(
                    f"model.layers.0.self_attn.{source}.weight",
                    kda_target,
                    shard_id=component,
                ),
                kda_shapes[component],
                torch.bfloat16,
            )
        )

    mla_target = "model.layers.92.self_attn.fused_qkv_a_proj_with_mqa.weight"
    for component, source, shape in (
        ("q_a", "q_a_proj", (1536, 7168)),
        ("kv_a", "kv_a_proj_with_mqa", (576, 7168)),
        ("g", "g_proj", (12288, 7168)),
    ):
        cases.append(
            (
                consumed_load_slot(
                    f"model.layers.92.self_attn.{source}.weight",
                    mla_target,
                    component=component,
                ),
                shape,
                torch.bfloat16,
            )
        )

    for target, rows in (
        ("model.layers.0.mlp.gate_up_proj.weight", 33792),
        (
            "model.layers.1.block_sparse_moe.shared_experts.gate_up_proj.weight",
            6144,
        ),
    ):
        for component, shard_id in (("gate", 0), ("up", 1)):
            cases.append(
                (
                    consumed_load_slot(
                        target.replace("gate_up_proj", f"{component}_proj"),
                        target,
                        shard_id=shard_id,
                    ),
                    (rows, 7168),
                    torch.bfloat16,
                )
            )

    for source, shape in (
        ("w1.weight_packed", (3072, 1792)),
        ("w1.weight_scale", (3072, 112)),
        ("w3.weight_packed", (3072, 1792)),
        ("w3.weight_scale", (3072, 112)),
        ("w2.weight_packed", (3584, 1536)),
        ("w2.weight_scale", (3584, 96)),
    ):
        cases.append(
            (
                parse_moe_checkpoint_slot(
                    "model.layers.1.block_sparse_moe.experts.0." + source
                ),
                shape,
                torch.uint8,
            )
        )
    return cases


@pytest.mark.parametrize(("slot", "shape", "dtype"), _validated_fusion_cases())
def test_fused_source_contract_accepts_only_exact_shape_and_dtype(
    slot: LoadSlot,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    config = _config()
    contract = source_tensor_contract(slot, config=config)
    assert contract is not None
    assert contract.shape == shape
    assert contract.dtype == dtype
    validate_source_tensor(slot, _tiny_cpu_tensor(shape, dtype), config=config)

    off_by_one = (shape[0] - 1, *shape[1:])
    with pytest.raises(KimiK3WeightContractError, match="shape mismatch"):
        validate_source_tensor(
            slot,
            _tiny_cpu_tensor(off_by_one, dtype),
            config=config,
        )

    wrong_dtype = torch.float32 if dtype != torch.float32 else torch.float16
    with pytest.raises(KimiK3WeightContractError, match="dtype mismatch"):
        validate_source_tensor(
            slot,
            _tiny_cpu_tensor(shape, wrong_dtype),
            config=config,
        )


def test_direct_parameter_uses_singleton_slot_and_defers_tensor_validation() -> None:
    target = "model.layers.0.input_layernorm.weight"
    expected = expected_rank_local_load_slots(
        target,
        config=SimpleNamespace(text_config=_config()),
        mapping=_mapping(),
    )
    consumed = consumed_load_slot(
        "language_model.model.layers.0.input_layernorm.weight",
        target,
    )

    assert expected == (LoadSlot(target, "direct", 0, "direct"),)
    assert consumed == expected[0]
    assert source_tensor_contract(consumed, config=_config()) is None
    validate_source_tensor(
        consumed, torch.ones(3, dtype=torch.float64), config=_config()
    )


def test_validated_mxfp4_sources_fill_the_expected_tp_shards() -> None:
    config = _config()
    config.moe_intermediate_size = 64
    config.routed_expert_hidden_size = 64
    tp_rank = 1
    tp_size = 2

    w13 = torch.zeros((1, 64, 32), dtype=torch.uint8)
    w13_scale = torch.zeros((1, 64, 2), dtype=torch.uint8)
    w2 = torch.zeros((1, 64, 16), dtype=torch.uint8)
    w2_scale = torch.zeros((1, 64, 1), dtype=torch.uint8)

    sources = {
        "w1.weight_packed": torch.cat(
            (
                torch.full((32, 32), 11, dtype=torch.uint8),
                torch.full((32, 32), 12, dtype=torch.uint8),
            )
        ),
        "w3.weight_packed": torch.cat(
            (
                torch.full((32, 32), 21, dtype=torch.uint8),
                torch.full((32, 32), 22, dtype=torch.uint8),
            )
        ),
        "w1.weight_scale": torch.cat(
            (
                torch.full((32, 2), 31, dtype=torch.uint8),
                torch.full((32, 2), 32, dtype=torch.uint8),
            )
        ),
        "w3.weight_scale": torch.cat(
            (
                torch.full((32, 2), 41, dtype=torch.uint8),
                torch.full((32, 2), 42, dtype=torch.uint8),
            )
        ),
        "w2.weight_packed": torch.cat(
            (
                torch.full((64, 16), 51, dtype=torch.uint8),
                torch.full((64, 16), 52, dtype=torch.uint8),
            ),
            dim=1,
        ),
        "w2.weight_scale": torch.cat(
            (
                torch.full((64, 1), 61, dtype=torch.uint8),
                torch.full((64, 1), 62, dtype=torch.uint8),
            ),
            dim=1,
        ),
    }
    targets = {
        "w1.weight_packed": (w13, "w1"),
        "w3.weight_packed": (w13, "w3"),
        "w1.weight_scale": (w13_scale, "w1"),
        "w3.weight_scale": (w13_scale, "w3"),
        "w2.weight_packed": (w2, "w2"),
        "w2.weight_scale": (w2_scale, "w2"),
    }
    prefix = "model.layers.1.block_sparse_moe.experts.0."
    for suffix, source in sources.items():
        slot = parse_moe_checkpoint_slot(prefix + suffix)
        validate_source_tensor(slot, source, config=config)
        target, shard_id = targets[suffix]
        load_model_weight(
            target,
            source,
            shard_id,
            local_expert_id=0,
            tp_rank=tp_rank,
            is_bias=False,
            use_presharded_weights=False,
            do_transpose=False,
            tp_size=tp_size,
        )

    assert torch.all(w13[0, :32] == 12)
    assert torch.all(w13[0, 32:] == 22)
    assert torch.all(w13_scale[0, :32] == 32)
    assert torch.all(w13_scale[0, 32:] == 42)
    assert torch.all(w2 == 52)
    assert torch.all(w2_scale == 62)
