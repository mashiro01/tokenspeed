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

import copy

import pytest

from tokenspeed.tools.checkpoint_ledger import TensorInfo, build_ledger, parse_plan
from tokenspeed.tools.kimi_k3_checkpoint_plan import (
    KimiK3PlanError,
    build_kimi_k3_checkpoint_plan,
)


def _config() -> dict:
    return {
        "architectures": ["KimiK3ForConditionalGeneration"],
        "model_type": "kimi_k3",
        "text_config": {
            "attn_res_block_size": 12,
            "hidden_size": 7168,
            "intermediate_size": 33792,
            "moe_intermediate_size": 3072,
            "num_attention_heads": 96,
            "num_hidden_layers": 93,
            "vocab_size": 163840,
        },
        "vision_config": {
            "qkv_hidden_size": 1536,
            "vt_num_attention_heads": 12,
        },
    }


def _tensor(name: str, dtype: str, shape: tuple[int, ...]) -> TensorInfo:
    bits = {"BF16": 16, "F32": 32, "U8": 8}[dtype]
    elements = 1
    for size in shape:
        elements *= size
    return TensorInfo(
        name=name,
        file="model.safetensors",
        dtype=dtype,
        shape=shape,
        nbytes=elements * bits // 8,
        dtype_bits=bits,
        proof_error=None,
    )


def _representative_tensors() -> list[TensorInfo]:
    bf16 = [
        ("language_model.model.embed_tokens.weight", (163840, 7168)),
        ("language_model.lm_head.weight", (163840, 7168)),
        ("language_model.model.layers.0.input_layernorm.weight", (7168,)),
        ("language_model.model.layers.0.mlp.gate_proj.weight", (33792, 7168)),
        ("language_model.model.layers.0.mlp.up_proj.weight", (33792, 7168)),
        ("language_model.model.layers.0.mlp.down_proj.weight", (7168, 33792)),
        ("language_model.model.layers.0.mlp_res_norm.weight", (7168,)),
        ("language_model.model.layers.0.mlp_res_proj.weight", (1, 7168)),
        ("language_model.model.layers.0.post_attention_layernorm.weight", (7168,)),
        (
            "language_model.model.layers.0.self_attention_res_norm.weight",
            (7168,),
        ),
        (
            "language_model.model.layers.0.self_attention_res_proj.weight",
            (1, 7168),
        ),
        ("language_model.model.layers.0.self_attn.b_proj.weight", (96, 7168)),
        ("language_model.model.layers.0.self_attn.f_a_proj.weight", (128, 7168)),
        ("language_model.model.layers.0.self_attn.f_b_proj.weight", (12288, 128)),
        ("language_model.model.layers.0.self_attn.g_proj.weight", (12288, 7168)),
        ("language_model.model.layers.0.self_attn.k_proj.weight", (12288, 7168)),
        ("language_model.model.layers.0.self_attn.o_proj.weight", (7168, 12288)),
        ("language_model.model.layers.0.self_attn.q_proj.weight", (12288, 7168)),
        ("language_model.model.layers.0.self_attn.v_proj.weight", (12288, 7168)),
        (
            "language_model.model.layers.1.block_sparse_moe.gate.weight",
            (896, 7168),
        ),
        (
            "language_model.model.layers.1.block_sparse_moe."
            "routed_expert_down_proj.weight",
            (3584, 7168),
        ),
        (
            "language_model.model.layers.1.block_sparse_moe."
            "routed_expert_up_proj.weight",
            (7168, 3584),
        ),
        (
            "language_model.model.layers.1.block_sparse_moe."
            "routed_expert_norm.weight",
            (3584,),
        ),
        (
            "language_model.model.layers.1.block_sparse_moe."
            "shared_experts.gate_proj.weight",
            (6144, 7168),
        ),
        (
            "language_model.model.layers.1.block_sparse_moe."
            "shared_experts.up_proj.weight",
            (6144, 7168),
        ),
        (
            "language_model.model.layers.1.block_sparse_moe."
            "shared_experts.down_proj.weight",
            (7168, 6144),
        ),
        (
            "language_model.model.layers.3.self_attn.kv_a_layernorm.weight",
            (512,),
        ),
        (
            "language_model.model.layers.3.self_attn." "kv_a_proj_with_mqa.weight",
            (576, 7168),
        ),
        (
            "language_model.model.layers.3.self_attn.kv_b_proj.weight",
            (24576, 512),
        ),
        (
            "language_model.model.layers.3.self_attn.q_a_layernorm.weight",
            (1536,),
        ),
        (
            "language_model.model.layers.3.self_attn.q_a_proj.weight",
            (1536, 7168),
        ),
        (
            "language_model.model.layers.3.self_attn.q_b_proj.weight",
            (18432, 1536),
        ),
        ("language_model.model.norm.weight", (7168,)),
        ("language_model.model.output_attn_res_norm.weight", (7168,)),
        ("language_model.model.output_attn_res_proj.weight", (1, 7168)),
        ("vision_tower.encoder.blocks.0.wqkv.weight", (4608, 1024)),
        ("vision_tower.encoder.blocks.0.wo.weight", (1024, 1536)),
        ("vision_tower.encoder.blocks.0.mlp.fc0.weight", (4096, 1024)),
        ("vision_tower.encoder.blocks.0.mlp.fc1.weight", (1024, 4096)),
        ("vision_tower.encoder.blocks.0.norm0.weight", (1024,)),
        ("vision_tower.encoder.blocks.0.norm1.weight", (1024,)),
        ("vision_tower.encoder.final_layernorm.weight", (1024,)),
        ("vision_tower.patch_embed.pos_emb.weight", (64, 64, 1024)),
        ("vision_tower.patch_embed.proj.weight", (1024, 3, 14, 14)),
        ("mm_projector.post_norm.weight", (7168,)),
        ("mm_projector.proj.0.weight", (4096, 4096)),
        ("mm_projector.proj.2.weight", (7168, 4096)),
    ]
    f32 = [
        ("language_model.model.layers.0.self_attn.A_log", (128,)),
        ("language_model.model.layers.0.self_attn.dt_bias", (12288,)),
        ("language_model.model.layers.0.self_attn.k_conv1d.weight", (12288, 1, 4)),
        ("language_model.model.layers.0.self_attn.o_norm.weight", (128,)),
        ("language_model.model.layers.0.self_attn.q_conv1d.weight", (12288, 1, 4)),
        ("language_model.model.layers.0.self_attn.v_conv1d.weight", (12288, 1, 4)),
        (
            "language_model.model.layers.1.block_sparse_moe.gate."
            "e_score_correction_bias",
            (896,),
        ),
    ]
    u8 = [
        (
            "language_model.model.layers.1.block_sparse_moe.experts.0."
            "w1.weight_packed",
            (3072, 1792),
        ),
        (
            "language_model.model.layers.1.block_sparse_moe.experts.0."
            "w1.weight_scale",
            (3072, 112),
        ),
        (
            "language_model.model.layers.1.block_sparse_moe.experts.0."
            "w2.weight_packed",
            (3584, 1536),
        ),
        (
            "language_model.model.layers.1.block_sparse_moe.experts.0."
            "w2.weight_scale",
            (3584, 96),
        ),
        (
            "language_model.model.layers.1.block_sparse_moe.experts.0."
            "w3.weight_packed",
            (3072, 1792),
        ),
        (
            "language_model.model.layers.1.block_sparse_moe.experts.0."
            "w3.weight_scale",
            (3072, 112),
        ),
    ]
    return [
        *(_tensor(name, "BF16", shape) for name, shape in bf16),
        *(_tensor(name, "F32", shape) for name, shape in f32),
        *(_tensor(name, "U8", shape) for name, shape in u8),
    ]


def test_k3_pp8_tp8_plan_routes_representative_checkpoint_exactly() -> None:
    plan_data = build_kimi_k3_checkpoint_plan(
        _config(), pipeline_parallel_size=8, tensor_parallel_size=8
    )
    plan = parse_plan(plan_data)
    ledger = build_ledger(_representative_tensors(), plan)

    assert [len(stage["ranks"]) for stage in plan_data["stages"]] == [8] * 8
    assert [
        sum(
            1
            for rule in plan_data["ownership"]
            if rule["stage"] == f"stage-{stage_id}" and ".layers\\." in rule["pattern"]
        )
        for stage_id in range(8)
    ] == [12, 12, 12, 12, 12, 12, 12, 9]
    assert ledger["totals"]["unknown_source_bytes"] == 0
    assert ledger["totals"]["unknown_bytes"] == 0

    alog = [
        entry
        for entry in ledger["entries"]
        if entry["tensor"].endswith("self_attn.A_log")
    ]
    assert [entry["bytes"] for entry in alog] == [48] * 8
    assert alog[0]["slices"] == [{"axis": 0, "start": 0, "stop": 12}]
    assert alog[-1]["slices"] == [{"axis": 0, "start": 84, "stop": 96}]

    vision = [
        entry
        for entry in ledger["entries"]
        if entry["tensor"] == "vision_tower.encoder.blocks.0.wqkv.weight"
    ]
    assert [entry["rank"] for entry in vision] == list(range(8))
    assert {entry["classification"] for entry in vision} == {"replicated"}

    final_mix = [
        entry
        for entry in ledger["entries"]
        if entry["tensor"] == "language_model.model.output_attn_res_proj.weight"
    ]
    assert [entry["rank"] for entry in final_mix] == list(range(56, 64))


def test_k3_plan_rejects_unqualified_topologies() -> None:
    with pytest.raises(KimiK3PlanError, match="mm_encoder_tp_mode='data'"):
        build_kimi_k3_checkpoint_plan(
            _config(),
            pipeline_parallel_size=8,
            tensor_parallel_size=8,
            mm_encoder_tp_mode="weights",
        )

    invalid = copy.deepcopy(_config())
    invalid["text_config"]["num_attention_heads"] = 95
    with pytest.raises(KimiK3PlanError, match="not divisible"):
        build_kimi_k3_checkpoint_plan(
            invalid, pipeline_parallel_size=8, tensor_parallel_size=8
        )

    invalid = copy.deepcopy(_config())
    invalid["architectures"] = ["OtherModel"]
    with pytest.raises(KimiK3PlanError, match="architectures"):
        build_kimi_k3_checkpoint_plan(
            invalid, pipeline_parallel_size=8, tensor_parallel_size=8
        )
