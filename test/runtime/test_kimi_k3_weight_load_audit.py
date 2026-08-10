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

import json
import logging
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tokenspeed.runtime.models.kimi_k3 import KimiLinearForCausalLM
from tokenspeed.runtime.pipeline.adapters.kimi_k3 import (
    build_kimi_k3_pipeline_plan,
)

_FINAL_NORM = "model.output_attn_res_norm.weight"
_FINAL_PROJ = "model.output_attn_res_proj.weight"


class _LoaderHarness:
    def __init__(self, *, stage_plan, parameter_names: tuple[str, ...]) -> None:
        self.config = SimpleNamespace(
            hidden_size=2,
            intermediate_size=2,
            moe_intermediate_size=2,
            routed_expert_hidden_size=2,
            num_shared_experts=1,
            num_experts=1,
            num_hidden_layers=2,
            num_nextn_predict_layers=1,
            num_attention_heads=1,
            q_lora_rank=None,
            kv_lora_rank=1,
            qk_rope_head_dim=1,
            v_head_dim=1,
            mla_use_output_gate=False,
            linear_attn_config={"num_heads": 1, "head_dim": 1},
            tie_word_embeddings=False,
        )
        self.mapping = SimpleNamespace(
            rank=stage_plan.stage_id,
            attn=SimpleNamespace(tp_rank=0, tp_size=1),
            moe=SimpleNamespace(ep_rank=0, ep_size=1),
        )
        self.model = SimpleNamespace(stage_plan=stage_plan)
        self.params = {
            name: nn.Parameter(torch.zeros(2, dtype=torch.float32))
            for name in parameter_names
        }
        self.post_load_calls = 0

    def named_parameters(self):
        return self.params.items()

    def post_load_weights(self) -> None:
        self.post_load_calls += 1


def _pp2_stage(stage_id: int):
    return build_kimi_k3_pipeline_plan(
        num_layers=2,
        hidden_size=2,
        attn_res_block_size=1,
        stage_layer_counts=(1, 1),
    ).stages[stage_id]


def _single_stage():
    return build_kimi_k3_pipeline_plan(
        num_layers=2,
        hidden_size=2,
        attn_res_block_size=1,
        stage_layer_counts=(2,),
    ).stages[0]


def _weight(name: str, value: float = 1.0) -> tuple[str, torch.Tensor]:
    return name, torch.full((2,), value, dtype=torch.float32)


def _matrix_weight(name: str, value: float = 1.0) -> tuple[str, torch.Tensor]:
    return name, torch.full((2, 2), value, dtype=torch.bfloat16)


def _load(harness: _LoaderHarness, weights) -> None:
    KimiLinearForCausalLM.load_weights(harness, weights)


def _last_audit_event(caplog) -> dict:
    events = []
    for record in caplog.records:
        try:
            payload = json.loads(record.getMessage())
        except json.JSONDecodeError:
            continue
        if payload.get("event") == "kimi_k3_weight_load_audit":
            events.append(payload)
    assert events
    return events[-1]


def test_final_stage_audit_separates_unowned_and_auxiliary_weights(caplog) -> None:
    caplog.set_level(logging.INFO)
    harness = _LoaderHarness(
        stage_plan=_pp2_stage(1),
        parameter_names=(_FINAL_NORM, _FINAL_PROJ),
    )

    _load(
        harness,
        [
            _weight("language_model.model.layers.0.input_layernorm.weight"),
            _weight("language_model.model.layers.2.mtp.weight"),
            _weight(f"language_model.{_FINAL_NORM}", 2.0),
            _weight(f"language_model.{_FINAL_PROJ}", 3.0),
        ],
    )

    event = _last_audit_event(caplog)
    assert event["status"] == "success"
    assert event["stage_unowned_skipped"] == 1
    assert event["auxiliary_skipped"] == 1
    assert event["owned_missing"] == 0
    assert event["owned_unexpected"] == 0
    assert event["output_attn_res_consumption"] == {
        _FINAL_NORM: 1,
        _FINAL_PROJ: 1,
    }
    assert set(event["output_attn_res_target_sha256"]) == {
        _FINAL_NORM,
        _FINAL_PROJ,
    }
    assert all(
        len(digest) == 64 for digest in event["output_attn_res_target_sha256"].values()
    )
    assert harness.post_load_calls == 1


def test_owned_checkpoint_weight_without_destination_fails_closed(caplog) -> None:
    harness = _LoaderHarness(
        stage_plan=_pp2_stage(1),
        parameter_names=(_FINAL_NORM, _FINAL_PROJ),
    )

    with pytest.raises(RuntimeError, match="owned_unexpected=1"):
        _load(
            harness,
            [
                _weight("model.layers.1.not_a_parameter.weight"),
                _weight(_FINAL_NORM),
                _weight(_FINAL_PROJ),
            ],
        )

    event = _last_audit_event(caplog)
    assert event["status"] == "failure"
    assert event["owned_missing"] == 0
    assert event["owned_unexpected"] == 1
    assert event["owned_unexpected_sample"] == ["model.layers.1.not_a_parameter.weight"]
    assert harness.post_load_calls == 0


def test_owned_runtime_parameter_missing_from_checkpoint_fails_closed(caplog) -> None:
    harness = _LoaderHarness(
        stage_plan=_pp2_stage(1),
        parameter_names=(_FINAL_NORM, _FINAL_PROJ),
    )

    with pytest.raises(RuntimeError, match="owned_missing=1"):
        _load(harness, [_weight(_FINAL_NORM)])

    event = _last_audit_event(caplog)
    assert event["owned_missing"] == 1
    assert event["owned_missing_sample"] == [f"text.{_FINAL_PROJ}"]
    assert event["output_attn_res_errors"] == [_FINAL_PROJ]
    assert harness.post_load_calls == 0


def test_final_attn_res_weight_must_be_consumed_exactly_once(caplog) -> None:
    harness = _LoaderHarness(
        stage_plan=_pp2_stage(1),
        parameter_names=(_FINAL_NORM, _FINAL_PROJ),
    )

    with pytest.raises(RuntimeError, match="owned_duplicate=1"):
        _load(
            harness,
            [
                _weight(_FINAL_NORM),
                _weight(_FINAL_NORM),
                _weight(_FINAL_PROJ),
            ],
        )

    event = _last_audit_event(caplog)
    assert event["owned_missing"] == 0
    assert event["owned_duplicate"] == 1
    assert event["output_attn_res_consumption"][_FINAL_NORM] == 2
    assert event["output_attn_res_errors"] == [_FINAL_NORM]
    assert harness.post_load_calls == 0


def test_nonfinal_stage_does_not_require_final_attn_res_weights(caplog) -> None:
    caplog.set_level(logging.INFO)
    local_param = "model.layers.0.input_layernorm.weight"
    harness = _LoaderHarness(
        stage_plan=_pp2_stage(0),
        parameter_names=(local_param,),
    )

    _load(
        harness,
        [
            _weight(local_param),
            _weight("model.layers.1.input_layernorm.weight"),
            _weight(_FINAL_NORM),
        ],
    )

    event = _last_audit_event(caplog)
    assert event["status"] == "success"
    assert event["stage_unowned_skipped"] == 2
    assert event["auxiliary_skipped"] == 0
    assert event["output_attn_res_consumption"] == {
        _FINAL_NORM: 0,
        _FINAL_PROJ: 0,
    }
    assert event["output_attn_res_errors"] == []


def test_single_stage_k3_uses_the_same_strict_audit(caplog) -> None:
    caplog.set_level(logging.INFO)
    embedding = "model.embed_tokens.weight"
    harness = _LoaderHarness(
        stage_plan=_single_stage(),
        parameter_names=(embedding, _FINAL_NORM, _FINAL_PROJ),
    )

    _load(
        harness,
        [
            _weight(f"language_model.{embedding}"),
            _weight(f"language_model.{_FINAL_NORM}"),
            _weight(f"language_model.{_FINAL_PROJ}"),
        ],
    )

    event = _last_audit_event(caplog)
    assert event["status"] == "success"
    assert event["stage_count"] == 1
    assert event["owned_checkpoint_tensors_consumed"] == 3
    assert event["owned_missing"] == 0
    assert event["owned_unexpected"] == 0


@pytest.mark.parametrize(
    "checkpoint_name",
    [
        "model.layers.0.rotary_emb.inv_freq",
        "model.layers.0.self_attn.k_scale",
    ],
)
def test_undeclared_auxiliary_checkpoint_weights_fail_closed(
    caplog,
    checkpoint_name: str,
) -> None:
    local_param = "model.layers.0.input_layernorm.weight"
    harness = _LoaderHarness(
        stage_plan=_pp2_stage(0),
        parameter_names=(local_param,),
    )

    with pytest.raises(RuntimeError, match="owned_unexpected=1"):
        _load(harness, [_weight(local_param), _weight(checkpoint_name)])

    event = _last_audit_event(caplog)
    assert event["owned_unexpected"] == 1
    assert event["owned_unexpected_sample"] == [checkpoint_name]


def test_redundant_lm_head_without_runtime_target_fails_closed(caplog) -> None:
    embedding = "model.embed_tokens.weight"
    harness = _LoaderHarness(
        stage_plan=_single_stage(),
        parameter_names=(embedding, _FINAL_NORM, _FINAL_PROJ),
    )

    with pytest.raises(RuntimeError, match="owned_unexpected=1"):
        _load(
            harness,
            [
                _weight(embedding),
                _weight(_FINAL_NORM),
                _weight(_FINAL_PROJ),
                _weight("lm_head.weight"),
            ],
        )

    event = _last_audit_event(caplog)
    assert event["owned_unexpected"] == 1
    assert event["owned_unexpected_sample"] == ["lm_head.weight"]


def test_partial_stacked_parameter_fails_closed(caplog) -> None:
    target = "model.layers.0.mlp.gate_up_proj.weight"
    harness = _LoaderHarness(
        stage_plan=_pp2_stage(0),
        parameter_names=(target,),
    )
    harness.params[target].weight_loader = lambda _param, _weight, _shard: None

    with pytest.raises(RuntimeError, match="owned_target_load_mismatch=1"):
        _load(harness, [_matrix_weight("model.layers.0.mlp.gate_proj.weight")])

    event = _last_audit_event(caplog)
    assert event["owned_missing"] == 0
    assert event["owned_target_load_mismatch"] == 1
    assert event["owned_target_load_mismatch_sample"] == {
        f"text.{target}": {"actual": 1, "expected": 2}
    }
    assert harness.post_load_calls == 0


def test_complete_stacked_parameter_satisfies_exact_load_count(caplog) -> None:
    caplog.set_level(logging.INFO)
    target = "model.layers.0.mlp.gate_up_proj.weight"
    harness = _LoaderHarness(
        stage_plan=_pp2_stage(0),
        parameter_names=(target,),
    )
    harness.params[target].weight_loader = lambda _param, _weight, _shard: None

    _load(
        harness,
        [
            _matrix_weight("model.layers.0.mlp.gate_proj.weight"),
            _matrix_weight("model.layers.0.mlp.up_proj.weight"),
        ],
    )

    event = _last_audit_event(caplog)
    assert event["status"] == "success"
    assert event["owned_target_load_mismatch"] == 0
    assert harness.post_load_calls == 1


def test_malformed_fused_source_is_rejected_before_loader_write(caplog) -> None:
    target = "model.layers.0.mlp.gate_up_proj.weight"
    harness = _LoaderHarness(
        stage_plan=_pp2_stage(0),
        parameter_names=(target,),
    )
    loader_calls = []
    harness.params[target].weight_loader = lambda *_args: loader_calls.append(True)

    with pytest.raises(RuntimeError, match="contract_violations=1"):
        _load(
            harness,
            [
                (
                    "model.layers.0.mlp.gate_proj.weight",
                    torch.ones((1, 2), dtype=torch.bfloat16),
                )
            ],
        )

    event = _last_audit_event(caplog)
    assert event["contract_violations"] == 1
    assert event["load_slots_missing"] == 2
    assert loader_calls == []
