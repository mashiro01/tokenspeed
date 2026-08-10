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

"""Strict, rank-local checkpoint contracts for Kimi-K3 fused weights."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch


class KimiK3WeightContractError(ValueError):
    """Raised when a checkpoint source violates the Kimi-K3 weight contract."""


@dataclass(frozen=True, order=True)
class LoadSlot:
    """One immutable source obligation for a rank-local runtime parameter."""

    target_name: str
    family: str
    layer_id: int
    component: str
    expert_id: int = -1
    tensor_kind: str = ""

    def __post_init__(self) -> None:
        if not self.target_name or self.target_name.startswith("language_model."):
            raise KimiK3WeightContractError(
                f"LoadSlot requires a canonical runtime target, got {self.target_name!r}"
            )
        if self.family not in {"direct", "gate_up", "kda", "mla", "moe"}:
            raise KimiK3WeightContractError(
                f"Unknown Kimi-K3 load family: {self.family!r}"
            )
        if self.layer_id < -1 or not self.component:
            raise KimiK3WeightContractError("LoadSlot has invalid layer/component")
        target_layer_id = _layer_id(self.target_name)
        if target_layer_id != self.layer_id:
            raise KimiK3WeightContractError(
                f"LoadSlot layer mismatch: target has {target_layer_id}, "
                f"slot has {self.layer_id}"
            )
        expected_target_suffix = {
            "kda": ".self_attn.qkvgb_proj.weight",
            "mla": ".self_attn.fused_qkv_a_proj_with_mqa.weight",
            "gate_up": ".gate_up_proj.weight",
        }.get(self.family)
        if expected_target_suffix is not None and not self.target_name.endswith(
            expected_target_suffix
        ):
            raise KimiK3WeightContractError(
                f"{self.family} slot has incompatible target {self.target_name!r}"
            )
        if self.family == "moe":
            if self.expert_id < 0:
                raise KimiK3WeightContractError("MoE LoadSlot requires an expert_id")
            if self.component not in {"w1", "w2", "w3"}:
                raise KimiK3WeightContractError(
                    f"Unknown MoE projection: {self.component!r}"
                )
            if self.tensor_kind not in {"weight_packed", "weight_scale"}:
                raise KimiK3WeightContractError(
                    f"Unknown MoE tensor kind: {self.tensor_kind!r}"
                )
            target_match = _MOE_TARGET_RE.fullmatch(self.target_name)
            if target_match is None:
                raise KimiK3WeightContractError(
                    f"MoE slot has incompatible target {self.target_name!r}"
                )
            target_kind = target_match.group("kind")
            expected_projections = (
                {"w1", "w3"} if target_kind.startswith("w13") else {"w2"}
            )
            expected_tensor_kind = (
                "weight_scale" if target_kind.endswith("_scale") else "weight_packed"
            )
            if (
                self.component not in expected_projections
                or self.tensor_kind != expected_tensor_kind
            ):
                raise KimiK3WeightContractError(
                    f"MoE slot does not match target {self.target_name!r}"
                )
        elif self.expert_id != -1 or self.tensor_kind:
            raise KimiK3WeightContractError(
                "Only MoE LoadSlots may carry expert_id/tensor_kind"
            )


@dataclass(frozen=True)
class SourceTensorContract:
    """Exact on-disk shape and dtype required before a loader writes a tensor."""

    shape: tuple[int, ...]
    dtype: torch.dtype


_KDA_COMPONENT_SOURCES = {
    "q": "q_proj",
    "k": "k_proj",
    "v": "v_proj",
    "g": "g_proj",
    "f_a": "f_a_proj",
    "b": "b_proj",
}
_MLA_COMPONENT_SOURCES = {
    "q_a": "q_a_proj",
    "kv_a": "kv_a_proj_with_mqa",
    "g": "g_proj",
}
_GATE_COMPONENT_SOURCES = {"gate": "gate_proj", "up": "up_proj"}
_GATE_SHARD_COMPONENTS = {0: "gate", 1: "up", "0": "gate", "1": "up"}
_MXFP4_GROUP_SIZE = 32
_LAYER_RE = re.compile(r"^model\.layers\.(?P<layer>\d+)\.")
_MOE_SOURCE_RE = re.compile(
    r"^(?P<prefix>model\.layers\.(?P<layer>\d+)\.block_sparse_moe\.)"
    r"experts\.(?P<expert>\d+)\.(?P<projection>w[123])\."
    r"(?P<kind>weight_packed|weight_scale)$"
)
_MOE_TARGET_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.block_sparse_moe\.experts\."
    r"(?P<kind>w13_weight|w13_weight_scale|w2_weight|w2_weight_scale)$"
)
_MISSING = object()


def _field(value: Any, name: str, default: Any = _MISSING) -> Any:
    if isinstance(value, Mapping):
        if name in value:
            return value[name]
    elif hasattr(value, name):
        return getattr(value, name)
    if default is not _MISSING:
        return default
    raise KimiK3WeightContractError(f"Kimi-K3 config is missing {name!r}")


def _text_config(config: Any) -> Any:
    return _field(config, "text_config", config)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise KimiK3WeightContractError(
            f"{name} must be a positive integer, got {value!r}"
        )
    return value


def _runtime_target_name(name: str) -> str:
    if not isinstance(name, str) or not name or name.startswith("language_model."):
        raise KimiK3WeightContractError(
            f"Expected a canonical runtime target name, got {name!r}"
        )
    return name


def _checkpoint_name(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise KimiK3WeightContractError(
            f"Expected a non-empty checkpoint name, got {name!r}"
        )
    return name.removeprefix("language_model.")


def _layer_id(name: str) -> int:
    match = _LAYER_RE.match(name)
    return -1 if match is None else int(match.group("layer"))


def _require_layer_id(name: str) -> int:
    layer_id = _layer_id(name)
    if layer_id < 0:
        raise KimiK3WeightContractError(
            f"Fused Kimi-K3 target has no decoder layer: {name!r}"
        )
    return layer_id


def _provided_component(
    inferred: str,
    *,
    component: str | None,
    shard_id: str | int | None,
    gate_up: bool = False,
) -> None:
    provided: list[str] = []
    if component is not None:
        provided.append(component)
    if shard_id is not None:
        normalized = (
            _GATE_SHARD_COMPONENTS.get(shard_id, shard_id) if gate_up else shard_id
        )
        if not isinstance(normalized, str):
            raise KimiK3WeightContractError(f"Invalid shard_id: {shard_id!r}")
        provided.append(normalized)
    if any(value != inferred for value in provided):
        raise KimiK3WeightContractError(
            f"Checkpoint component is {inferred!r}, caller supplied {provided!r}"
        )


def _moe_runtime_target(prefix: str, projection: str, tensor_kind: str) -> str:
    if projection in {"w1", "w3"}:
        suffix = "w13_weight" if tensor_kind == "weight_packed" else "w13_weight_scale"
    else:
        suffix = "w2_weight" if tensor_kind == "weight_packed" else "w2_weight_scale"
    return f"{prefix}experts.{suffix}"


def parse_moe_checkpoint_slot(
    checkpoint_name: str,
    *,
    component: str | None = None,
    shard_id: str | None = None,
) -> LoadSlot:
    """Parse an official MXFP4 source before calling ``moe_loader.load``.

    The returned slot contains the runtime target that the generic loader must
    subsequently return. Non-official suffixes such as ``.weight`` are rejected.
    """

    canonical_name = _checkpoint_name(checkpoint_name)
    match = _MOE_SOURCE_RE.fullmatch(canonical_name)
    if match is None:
        raise KimiK3WeightContractError(
            "Kimi-K3 MXFP4 sources must end in exactly "
            f"'.weight_packed' or '.weight_scale', got {checkpoint_name!r}"
        )
    projection = match.group("projection")
    _provided_component(
        projection,
        component=component,
        shard_id=shard_id,
    )
    tensor_kind = match.group("kind")
    return LoadSlot(
        target_name=_moe_runtime_target(match.group("prefix"), projection, tensor_kind),
        family="moe",
        layer_id=int(match.group("layer")),
        component=projection,
        expert_id=int(match.group("expert")),
        tensor_kind=tensor_kind,
    )


def assert_loader_target(slot: LoadSlot, loader_target_name: str) -> None:
    """Assert that a loader returned the target predicted before the write."""

    actual = _runtime_target_name(loader_target_name)
    if actual != slot.target_name:
        raise KimiK3WeightContractError(
            f"Loader target mismatch: expected {slot.target_name!r}, got {actual!r}"
        )


def expected_rank_local_load_slots(
    target_name: str,
    *,
    config: Any,
    mapping: Any,
) -> tuple[LoadSlot, ...]:
    """Generate the exact source obligations for one rank-local runtime target."""

    target_name = _runtime_target_name(target_name)
    layer_id = _layer_id(target_name)

    if target_name.endswith(".self_attn.qkvgb_proj.weight"):
        layer_id = _require_layer_id(target_name)
        return tuple(
            sorted(
                LoadSlot(target_name, "kda", layer_id, component)
                for component in _KDA_COMPONENT_SOURCES
            )
        )

    if target_name.endswith(".self_attn.fused_qkv_a_proj_with_mqa.weight"):
        layer_id = _require_layer_id(target_name)
        text_config = _text_config(config)
        components = ["q_a", "kv_a"]
        if bool(_field(text_config, "mla_use_output_gate", False)):
            components.append("g")
        return tuple(
            sorted(
                LoadSlot(target_name, "mla", layer_id, component)
                for component in components
            )
        )

    if target_name.endswith(".gate_up_proj.weight"):
        layer_id = _require_layer_id(target_name)
        return tuple(
            sorted(
                LoadSlot(target_name, "gate_up", layer_id, component)
                for component in _GATE_COMPONENT_SOURCES
            )
        )

    moe_match = _MOE_TARGET_RE.fullmatch(target_name)
    if moe_match is not None:
        text_config = _text_config(config)
        num_experts = _positive_int(_field(text_config, "num_experts"), "num_experts")
        moe_mapping = _field(mapping, "moe")
        ep_size = _positive_int(_field(moe_mapping, "ep_size"), "moe.ep_size")
        ep_rank = _field(moe_mapping, "ep_rank")
        if (
            isinstance(ep_rank, bool)
            or not isinstance(ep_rank, int)
            or not 0 <= ep_rank < ep_size
        ):
            raise KimiK3WeightContractError(
                f"moe.ep_rank must be in [0, {ep_size}), got {ep_rank!r}"
            )
        if num_experts % ep_size:
            raise KimiK3WeightContractError(
                f"num_experts={num_experts} is not divisible by ep_size={ep_size}"
            )
        local_experts = num_experts // ep_size
        first_expert = ep_rank * local_experts
        target_kind = moe_match.group("kind")
        if target_kind.startswith("w13"):
            projections = ("w1", "w3")
        else:
            projections = ("w2",)
        tensor_kind = (
            "weight_scale" if target_kind.endswith("_scale") else "weight_packed"
        )
        return tuple(
            sorted(
                LoadSlot(
                    target_name=target_name,
                    family="moe",
                    layer_id=int(moe_match.group("layer")),
                    component=projection,
                    expert_id=expert_id,
                    tensor_kind=tensor_kind,
                )
                for expert_id in range(first_expert, first_expert + local_experts)
                for projection in projections
            )
        )

    return (LoadSlot(target_name, "direct", layer_id, "direct"),)


def consumed_load_slot(
    checkpoint_name: str,
    target_name: str,
    *,
    component: str | None = None,
    shard_id: str | int | None = None,
) -> LoadSlot:
    """Create the exact slot consumed by one successful source load."""

    target_name = _runtime_target_name(target_name)
    canonical_name = _checkpoint_name(checkpoint_name)
    layer_id = _layer_id(target_name)

    if _MOE_TARGET_RE.fullmatch(target_name) is not None:
        if shard_id is not None and not isinstance(shard_id, str):
            raise KimiK3WeightContractError(f"Invalid MoE shard_id: {shard_id!r}")
        slot = parse_moe_checkpoint_slot(
            checkpoint_name,
            component=component,
            shard_id=shard_id,
        )
        assert_loader_target(slot, target_name)
        return slot

    if target_name.endswith(".self_attn.qkvgb_proj.weight"):
        prefix = target_name[: -len("qkvgb_proj.weight")]
        sources = {
            f"{prefix}{source}.weight": name
            for name, source in _KDA_COMPONENT_SOURCES.items()
        }
        inferred = sources.get(canonical_name)
        family = "kda"
    elif target_name.endswith(".self_attn.fused_qkv_a_proj_with_mqa.weight"):
        prefix = target_name[: -len("fused_qkv_a_proj_with_mqa.weight")]
        sources = {
            f"{prefix}{source}.weight": name
            for name, source in _MLA_COMPONENT_SOURCES.items()
        }
        inferred = sources.get(canonical_name)
        family = "mla"
    elif target_name.endswith(".gate_up_proj.weight"):
        prefix = target_name[: -len("gate_up_proj.weight")]
        sources = {
            f"{prefix}{source}.weight": name
            for name, source in _GATE_COMPONENT_SOURCES.items()
        }
        inferred = sources.get(canonical_name)
        family = "gate_up"
    else:
        if component is not None or shard_id is not None:
            raise KimiK3WeightContractError(
                "Direct Kimi-K3 loads do not accept component/shard_id"
            )
        return LoadSlot(target_name, "direct", layer_id, "direct")

    if inferred is None:
        raise KimiK3WeightContractError(
            f"Checkpoint {checkpoint_name!r} is not an official source for {target_name!r}"
        )
    _provided_component(
        inferred,
        component=component,
        shard_id=shard_id,
        gate_up=family == "gate_up",
    )
    return LoadSlot(
        target_name=target_name,
        family=family,
        layer_id=_require_layer_id(target_name),
        component=inferred,
    )


def _mxfp4_group_size(text_config: Any) -> int:
    quantization_config = _field(text_config, "quantization_config", None)
    if quantization_config is None:
        return 32
    groups = _field(quantization_config, "config_groups", None)
    if not isinstance(groups, Mapping) or not groups:
        return _MXFP4_GROUP_SIZE
    first_group = groups[sorted(groups)[0]]
    weights = _field(first_group, "weights")
    group_size = _positive_int(_field(weights, "group_size"), "MXFP4 group_size")
    if group_size != _MXFP4_GROUP_SIZE:
        raise KimiK3WeightContractError(
            f"Kimi-K3 MXFP4 requires group_size={_MXFP4_GROUP_SIZE}, "
            f"got {group_size}"
        )
    return group_size


def source_tensor_contract(
    slot: LoadSlot,
    *,
    config: Any,
) -> SourceTensorContract | None:
    """Return the exact source tensor contract, or ``None`` for direct loads."""

    if slot.family == "direct":
        return None

    text_config = _text_config(config)
    hidden_size = _positive_int(_field(text_config, "hidden_size"), "hidden_size")

    if slot.family == "kda":
        linear_config = _field(text_config, "linear_attn_config")
        num_heads = _positive_int(
            _field(linear_config, "num_heads"), "linear_attn_config.num_heads"
        )
        head_dim = _positive_int(
            _field(linear_config, "head_dim"), "linear_attn_config.head_dim"
        )
        rows = {
            "q": num_heads * head_dim,
            "k": num_heads * head_dim,
            "v": num_heads * head_dim,
            "g": num_heads * head_dim,
            "f_a": head_dim,
            "b": num_heads,
        }.get(slot.component)
        if rows is None:
            raise KimiK3WeightContractError(
                f"Unknown KDA component: {slot.component!r}"
            )
        return SourceTensorContract((rows, hidden_size), torch.bfloat16)

    if slot.family == "mla":
        if slot.component == "q_a":
            rows = _positive_int(_field(text_config, "q_lora_rank"), "q_lora_rank")
        elif slot.component == "kv_a":
            kv_lora_rank = _positive_int(
                _field(text_config, "kv_lora_rank"), "kv_lora_rank"
            )
            rope_head_dim = _positive_int(
                _field(text_config, "qk_rope_head_dim"), "qk_rope_head_dim"
            )
            rows = kv_lora_rank + rope_head_dim
        elif slot.component == "g":
            num_heads = _positive_int(
                _field(text_config, "num_attention_heads"), "num_attention_heads"
            )
            value_head_dim = _positive_int(
                _field(text_config, "v_head_dim"), "v_head_dim"
            )
            rows = num_heads * value_head_dim
        else:
            raise KimiK3WeightContractError(
                f"Unknown MLA component: {slot.component!r}"
            )
        return SourceTensorContract((rows, hidden_size), torch.bfloat16)

    if slot.family == "gate_up":
        if ".shared_experts.gate_up_proj.weight" in slot.target_name:
            rows = _positive_int(
                _field(text_config, "moe_intermediate_size"),
                "moe_intermediate_size",
            ) * _positive_int(
                _field(text_config, "num_shared_experts"), "num_shared_experts"
            )
        elif ".mlp.gate_up_proj.weight" in slot.target_name:
            rows = _positive_int(
                _field(text_config, "intermediate_size"), "intermediate_size"
            )
        else:
            raise KimiK3WeightContractError(
                f"Unknown gate/up target: {slot.target_name!r}"
            )
        return SourceTensorContract((rows, hidden_size), torch.bfloat16)

    if slot.family == "moe":
        moe_intermediate = _positive_int(
            _field(text_config, "moe_intermediate_size"), "moe_intermediate_size"
        )
        routed_hidden = _positive_int(
            _field(text_config, "routed_expert_hidden_size"),
            "routed_expert_hidden_size",
        )
        group_size = _mxfp4_group_size(text_config)
        divisor = 2 if slot.tensor_kind == "weight_packed" else group_size
        input_size = (
            routed_hidden if slot.component in {"w1", "w3"} else moe_intermediate
        )
        output_size = (
            moe_intermediate if slot.component in {"w1", "w3"} else routed_hidden
        )
        if input_size % divisor:
            raise KimiK3WeightContractError(
                f"MXFP4 input size {input_size} is not divisible by {divisor}"
            )
        return SourceTensorContract(
            (output_size, input_size // divisor),
            torch.uint8,
        )

    raise KimiK3WeightContractError(f"Unknown load family: {slot.family!r}")


def validate_source_tensor(
    slot: LoadSlot,
    tensor: torch.Tensor,
    *,
    config: Any,
) -> None:
    """Reject malformed fused/MXFP4 tensors before any loader writes them."""

    if not isinstance(tensor, torch.Tensor):
        raise KimiK3WeightContractError(
            f"Expected torch.Tensor for {slot}, got {type(tensor).__name__}"
        )
    contract = source_tensor_contract(slot, config=config)
    if contract is None:
        return
    actual_shape = tuple(tensor.shape)
    if actual_shape != contract.shape:
        raise KimiK3WeightContractError(
            f"Source shape mismatch for {slot}: expected {contract.shape}, "
            f"got {actual_shape}"
        )
    if tensor.dtype != contract.dtype:
        raise KimiK3WeightContractError(
            f"Source dtype mismatch for {slot}: expected {contract.dtype}, "
            f"got {tensor.dtype}"
        )


__all__ = [
    "KimiK3WeightContractError",
    "LoadSlot",
    "SourceTensorContract",
    "assert_loader_target",
    "consumed_load_slot",
    "expected_rank_local_load_slots",
    "parse_moe_checkpoint_slot",
    "source_tensor_contract",
    "validate_source_tensor",
]
