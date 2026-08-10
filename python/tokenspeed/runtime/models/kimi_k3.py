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

"""Kimi-K3 model.

Kimi-K3 = MoonViT3d vision tower (reused from Kimi-K2.5) + ``KimiLinear`` text
backbone (hybrid KDA linear-attention / NoPE-MLA full-attention decoder with a
DeepSeek-V3 style latent MoE and block-level attention residuals).

Implemented (full text path):

* ``KimiLinearMLAAttention`` — NoPE MLA + sigmoid output gate.
* ``KimiLinearKDA`` — per-head gated delta-rule linear attention; routes the
  conv + gated-delta scan + conv/recurrent state cache through the hybrid
  ``MambaAttnBackend`` KDA branch.
* ``KimiLinearMLP`` — dense / shared-expert MLP with the SiTU activation.
* ``KimiLinearMoE`` — sigmoid/noaux_tc router + Latent MoE + flashinfer's
  TRT-LLM fused SiTU + shared experts.
* ``KimiLinearDecoderLayer`` + ``KimiLinearModel`` — the AttnRes block-residual data
  flow.
* ``KimiLinearForCausalLM.load_weights`` — stacked / fused-qkv-a / expert
  mappings; post-load absorbed MLA ``w_kc``/``w_vc`` prep.
* ``KimiK3ForConditionalGeneration`` registration.

The multimodal path uses the shared MoonViT3d implementation with K3's
wide-QKV/RMSNorm configuration. At TP8 it runs as item-DP8 (vision TP1) via
``--mm-encoder-tp-mode data`` and gathers exact-size encoder outputs before
splicing them into the text embeddings.

Module hierarchy matches the checkpoint::

    KimiK3ForConditionalGeneration
      language_model (KimiLinearForCausalLM)
        model (KimiLinearModel): embed_tokens, layers[.self_attn/.mlp/...], norm
        lm_head
      vision (KimiK3Vision): vision_tower.*, mm_projector.*
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from collections.abc import Iterable
from collections.abc import Mapping as MappingABC
from functools import partial
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.activation.triton import (
    add3,
    attnres_combine,
    attnres_partial,
    attnres_partial_dual,
    rmsnorm_gated_sigmoid,
    sigmoid_mul,
)
from tokenspeed_kernel.ops.attention import mla_normalize_project_query
from tokenspeed_kernel.ops.attn_res import attn_res_fwd
from tokenspeed_kernel.ops.communication import allreduce_fusion_lane
from tokenspeed_kernel.ops.gemm import (
    kimi3_latent_projection_add3,
    kimi3_mla_qkv_gate_projection,
    kimi3_qkvfab_projection,
    kimi3_router_projection,
    kimi3_shared_down_projection,
    kimi3_shared_situ_projection,
)
from tokenspeed_kernel.ops.gemm.triton_gemv import (
    decode_gemv,
)
from tokenspeed_kernel.ops.moe import (
    latent_moe_decode_pipeline_available,
    latent_moe_expert_shared,
    latent_moe_input_projections,
)
from tokenspeed_kernel.ops.moe.flashinfer.trtllm_mxfp4 import (
    situ_moe_unavailable_reason,
)
from tokenspeed_kernel.ops.tuning import load_packaged_flashinfer_tuning_cache
from torch import nn

from tokenspeed.runtime.configs.kimi_k3_config import KimiK3Config, KimiLinearConfig
from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.distributed.comm_ops import (
    all_reduce,
    all_reduce_two,
    prepare_all_reduce_fusion,
    prepare_all_reduce_lane,
)
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.cuda_graph_wrapper import get_is_capture_mode
from tokenspeed.runtime.layers.activation import SituAndMul
from tokenspeed.runtime.layers.layernorm import (
    RMSNorm,
    _get_process_group,
)
from tokenspeed.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.logits_processor import LogitsMetadata
from tokenspeed.runtime.layers.moe.expert import MoELayer
from tokenspeed.runtime.layers.moe.latent import (
    Kimi3LatentProjection,
    Kimi3MoEExecutionPlan,
    LatentMoELayer,
    kimi3_join_reduce_moe,
)
from tokenspeed.runtime.layers.moe.loader import build_moe_checkpoint_loader
from tokenspeed.runtime.layers.moe.schema import ExpertCheckpointSchema
from tokenspeed.runtime.layers.moe.topk import TopK, TopKOutput, TopKOutputFormat
from tokenspeed.runtime.layers.moe.utils import RoutingMethodType, get_moe_backend
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.vocab_parallel_embedding import VocabParallelEmbedding
from tokenspeed.runtime.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from tokenspeed.runtime.models.base.causal_lm import BaseCausalLM
from tokenspeed.runtime.models.deepseek_v3 import (
    DeepseekV3AttentionMLA,
    DeepseekV3FusedQkvAProjWithMqa,
    _prepare_mla_kv_b_proj_weights,
)
from tokenspeed.runtime.models.kimi_k3_weight_contract import (
    KimiK3WeightContractError,
    LoadSlot,
    assert_loader_target,
    consumed_load_slot,
    expected_rank_local_load_slots,
    parse_moe_checkpoint_slot,
    validate_source_tensor,
)
from tokenspeed.runtime.models.moonvit import MoonViTVisionPath
from tokenspeed.runtime.multimodal.embedder import (
    EncoderSpec,
    VisionEmbedder,
    pad_input_tokens,
)
from tokenspeed.runtime.multimodal.inputs import (
    Modality,
    MultimodalInputs,
)
from tokenspeed.runtime.pipeline.adapters.kimi_k3 import (
    build_balanced_kimi_k3_pipeline_plan,
    kimi_k3_stage_checkpoint_weight_filter,
)
from tokenspeed.runtime.pipeline.contracts import (
    PipelineProtocolError,
    StageActivation,
    StageOutput,
)
from tokenspeed.runtime.utils import add_prefix, ceil_div
from tokenspeed.runtime.utils.cuda_stream import StreamFork
from tokenspeed.runtime.utils.env import global_server_args_dict
from tokenspeed.runtime.utils.pdl import pdl_enabled

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.multimodal.encoder_cudagraph import (
        EncoderCudaGraphWrapper,
    )

logger = logging.getLogger(__name__)

_KIMI_K3_WEIGHT_AUDIT_EVENT = "kimi_k3_weight_load_audit"
_KIMI_K3_FINAL_ATTN_RES_WEIGHTS = (
    "model.output_attn_res_norm.weight",
    "model.output_attn_res_proj.weight",
)
_KIMI_K3_OPTIONAL_CHECKPOINT_SUFFIXES = (".k_scale", ".v_scale")
_KIMI_K3_OPTIONAL_RUNTIME_PARAM_SUFFIXES = (
    "._dummy_norm.weight",
    *_KIMI_K3_OPTIONAL_CHECKPOINT_SUFFIXES,
)
_KIMI_K3_TEXT_WEIGHT_PREFIXES = (
    "model.layers.",
    "model.embed_tokens.",
    "model.norm.",
    "model.output_attn_res_norm.",
    "model.output_attn_res_proj.",
    "lm_head.",
)


def _strip_kimi_k3_language_prefix(name: str) -> str:
    return name.removeprefix("language_model.")


def _kimi_k3_checkpoint_layer_id(name: str) -> int | None:
    name = _strip_kimi_k3_language_prefix(name)
    if not name.startswith("model.layers."):
        return None
    parts = name.split(".", 3)
    if len(parts) < 4 or not parts[2].isdigit():
        return None
    return int(parts[2])


def _is_kimi_k3_checkpoint_auxiliary(
    name: str,
    *,
    num_hidden_layers: int,
    num_nextn_predict_layers: int,
) -> bool:
    """Return whether a checkpoint tensor is intentionally not target state."""

    canonical_name = _strip_kimi_k3_language_prefix(name)
    layer_id = _kimi_k3_checkpoint_layer_id(canonical_name)
    # Target checkpoints may carry appended MTP/NextN layers. They are loaded
    # by the draft worker, not by the target stage.
    return (
        layer_id is not None
        and num_hidden_layers <= layer_id < num_hidden_layers + num_nextn_predict_layers
    )


def _classify_kimi_k3_checkpoint_weight(
    name: str,
    *,
    stage_plan,
    include_vision: bool,
    num_hidden_layers: int,
    num_nextn_predict_layers: int = 0,
    language_enabled: bool = True,
) -> str:
    """Classify a checkpoint key before a stage-local loader consumes it."""

    if not isinstance(name, str) or not name:
        return "unexpected"
    if _is_kimi_k3_checkpoint_auxiliary(
        name,
        num_hidden_layers=num_hidden_layers,
        num_nextn_predict_layers=num_nextn_predict_layers,
    ):
        return "auxiliary"
    if name.startswith(("vision_tower.", "mm_projector.")):
        return "owned" if include_vision else "stage_unowned"

    canonical_name = _strip_kimi_k3_language_prefix(name)
    if not language_enabled:
        if name.startswith("language_model.") or canonical_name.startswith(
            _KIMI_K3_TEXT_WEIGHT_PREFIXES
        ):
            return "stage_unowned"
        return "unexpected"
    if kimi_k3_stage_checkpoint_weight_filter(
        name,
        stage_plan=stage_plan,
        include_vision=include_vision,
    ):
        return "owned"

    layer_id = _kimi_k3_checkpoint_layer_id(canonical_name)
    if layer_id is not None and layer_id < num_hidden_layers:
        return "stage_unowned"
    if canonical_name.startswith(_KIMI_K3_TEXT_WEIGHT_PREFIXES[1:]):
        return "stage_unowned"
    return "unexpected"


def _is_optional_kimi_k3_runtime_param(name: str) -> bool:
    return name.endswith(_KIMI_K3_OPTIONAL_RUNTIME_PARAM_SUFFIXES)


def _kimi_k3_tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach()
    if value.device.type != "cpu":
        value = value.cpu()
    value = value.contiguous().view(torch.uint8)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


class _KimiK3WeightLoadAudit:
    """Rank-local, fail-closed audit for one Kimi-K3 pipeline stage."""

    _SAMPLE_LIMIT = 8

    def __init__(self, *, stage_plan, mapping, config) -> None:
        self.stage_plan = stage_plan
        self.mapping = mapping
        self.config = config
        self.required_targets: set[str] = set()
        self.expected_target_loads: dict[str, int] = {}
        self.consumed_targets: set[str] = set()
        self.target_consumption: Counter[str] = Counter()
        self.checkpoint_consumption: Counter[str] = Counter()
        self.expected_slots: set[tuple[str, LoadSlot]] = set()
        self.consumed_slots: Counter[tuple[str, LoadSlot]] = Counter()
        self._final_target_parameters: dict[str, torch.Tensor] = {}
        self.owned_seen = 0
        self.owned_unexpected = 0
        self.stage_unowned_skipped = 0
        self.rank_unowned_skipped = 0
        self.auxiliary_skipped = 0
        self.contract_violations = 0
        self._unexpected_samples: list[str] = []
        self._stage_unowned_samples: list[str] = []
        self._rank_unowned_samples: list[str] = []
        self._auxiliary_samples: list[str] = []
        self._contract_violation_samples: list[str] = []
        self._finished = False

    @staticmethod
    def target_name(namespace: str, name: str) -> str:
        return f"{namespace}.{name}"

    @staticmethod
    def _append_sample(samples: list[str], name: str) -> None:
        if len(samples) < _KimiK3WeightLoadAudit._SAMPLE_LIMIT:
            samples.append(name)

    def require_targets(
        self,
        namespace: str,
        parameters: MappingABC[str, torch.Tensor],
    ) -> None:
        for name, parameter in parameters.items():
            if _is_optional_kimi_k3_runtime_param(name):
                continue
            qualified_name = self.target_name(namespace, name)
            slots = (
                expected_rank_local_load_slots(
                    name,
                    config=self.config,
                    mapping=self.mapping,
                )
                if namespace == "text"
                else (LoadSlot(name, "direct", -1, "direct"),)
            )
            self.required_targets.add(qualified_name)
            self.expected_target_loads[qualified_name] = len(slots)
            self.expected_slots.update((namespace, slot) for slot in slots)
            if namespace == "text" and name in _KIMI_K3_FINAL_ATTN_RES_WEIGHTS:
                self._final_target_parameters[name] = parameter

    def prepare_slot(
        self,
        checkpoint_name: str,
        loaded_weight: torch.Tensor,
        *,
        namespace: str,
        target_name: str,
        component: str | None = None,
        shard_id: str | int | None = None,
    ) -> LoadSlot | None:
        try:
            slot = (
                consumed_load_slot(
                    checkpoint_name,
                    target_name,
                    component=component,
                    shard_id=shard_id,
                )
                if namespace == "text"
                else LoadSlot(target_name, "direct", -1, "direct")
            )
            validate_source_tensor(slot, loaded_weight, config=self.config)
            return slot
        except KimiK3WeightContractError as error:
            self.record_contract_violation(checkpoint_name, error)
            return None

    def prepare_moe_slot(
        self,
        checkpoint_name: str,
        loaded_weight: torch.Tensor,
    ) -> LoadSlot | None:
        try:
            slot = parse_moe_checkpoint_slot(checkpoint_name)
            validate_source_tensor(slot, loaded_weight, config=self.config)
            return slot
        except KimiK3WeightContractError as error:
            self.record_contract_violation(checkpoint_name, error)
            return None

    def confirm_loader_target(
        self,
        checkpoint_name: str,
        slot: LoadSlot,
        loader_target_name: str,
    ) -> bool:
        try:
            assert_loader_target(slot, loader_target_name)
            return True
        except KimiK3WeightContractError as error:
            self.record_contract_violation(checkpoint_name, error)
            return False

    def record_contract_violation(
        self,
        checkpoint_name: str,
        error: Exception,
    ) -> None:
        self.contract_violations += 1
        self._append_sample(
            self._contract_violation_samples,
            f"{checkpoint_name}: {error}",
        )

    def record_consumed(
        self,
        checkpoint_name: str,
        *,
        namespace: str,
        target_name: str,
        slot: LoadSlot,
    ) -> None:
        canonical_name = _strip_kimi_k3_language_prefix(checkpoint_name)
        qualified_target = self.target_name(namespace, target_name)
        self.owned_seen += 1
        self.checkpoint_consumption[canonical_name] += 1
        self.consumed_targets.add(qualified_target)
        self.target_consumption[qualified_target] += 1
        self.consumed_slots[(namespace, slot)] += 1

    def record_unexpected(self, checkpoint_name: str) -> None:
        self.owned_seen += 1
        self.owned_unexpected += 1
        self._append_sample(self._unexpected_samples, checkpoint_name)

    def record_stage_unowned(self, checkpoint_name: str) -> None:
        self.stage_unowned_skipped += 1
        self._append_sample(self._stage_unowned_samples, checkpoint_name)

    def record_rank_unowned(self, checkpoint_name: str) -> None:
        self.rank_unowned_skipped += 1
        self._append_sample(self._rank_unowned_samples, checkpoint_name)

    def record_auxiliary(self, checkpoint_name: str) -> None:
        self.auxiliary_skipped += 1
        self._append_sample(self._auxiliary_samples, checkpoint_name)

    def finish(self) -> None:
        if self._finished:
            raise RuntimeError("Kimi-K3 weight load audit was finalized twice")
        self._finished = True

        missing = sorted(self.required_targets - self.consumed_targets)
        missing_slots = sorted(self.expected_slots - self.consumed_slots.keys())
        unexpected_slots = sorted(self.consumed_slots.keys() - self.expected_slots)
        duplicate_slots = sorted(
            slot for slot, count in self.consumed_slots.items() if count != 1
        )
        target_load_errors = {
            name: {
                "actual": self.target_consumption[name],
                "expected": expected,
            }
            for name, expected in sorted(self.expected_target_loads.items())
            if self.target_consumption[name] != expected
        }
        duplicate_sources = sorted(
            name for name, count in self.checkpoint_consumption.items() if count != 1
        )
        output_attn_res_consumption = {
            name: self.checkpoint_consumption[name]
            for name in _KIMI_K3_FINAL_ATTN_RES_WEIGHTS
        }
        output_attn_res_errors = []
        output_attn_res_target_sha256 = {
            name: _kimi_k3_tensor_sha256(parameter)
            for name, parameter in sorted(self._final_target_parameters.items())
        }
        if self.stage_plan.owns_head:
            output_attn_res_errors = sorted(
                name
                for name, count in output_attn_res_consumption.items()
                if count != 1
            )

        failed = bool(
            missing
            or self.owned_unexpected
            or self.contract_violations
            or duplicate_sources
            or missing_slots
            or unexpected_slots
            or duplicate_slots
            or target_load_errors
            or output_attn_res_errors
        )
        payload = {
            "auxiliary_samples": self._auxiliary_samples,
            "auxiliary_skipped": self.auxiliary_skipped,
            "event": _KIMI_K3_WEIGHT_AUDIT_EVENT,
            "first_layer": self.stage_plan.first_layer,
            "global_rank": getattr(self.mapping, "rank", None),
            "last_layer_exclusive": self.stage_plan.end_layer,
            "output_attn_res_consumption": output_attn_res_consumption,
            "output_attn_res_errors": output_attn_res_errors,
            "output_attn_res_target_sha256": output_attn_res_target_sha256,
            "owned_checkpoint_tensors_consumed": sum(
                self.checkpoint_consumption.values()
            ),
            "owned_checkpoint_tensors_seen": self.owned_seen,
            "contract_violations": self.contract_violations,
            "contract_violation_samples": self._contract_violation_samples,
            "owned_duplicate": len(duplicate_sources),
            "owned_duplicate_sample": duplicate_sources[: self._SAMPLE_LIMIT],
            "owned_missing": len(missing),
            "owned_missing_sample": missing[: self._SAMPLE_LIMIT],
            "owned_target_load_mismatch": len(target_load_errors),
            "owned_target_load_mismatch_sample": dict(
                list(target_load_errors.items())[: self._SAMPLE_LIMIT]
            ),
            "load_slots_duplicate": len(duplicate_slots),
            "load_slots_duplicate_sample": [
                f"{namespace}:{slot}"
                for namespace, slot in duplicate_slots[: self._SAMPLE_LIMIT]
            ],
            "load_slots_missing": len(missing_slots),
            "load_slots_missing_sample": [
                f"{namespace}:{slot}"
                for namespace, slot in missing_slots[: self._SAMPLE_LIMIT]
            ],
            "load_slots_unexpected": len(unexpected_slots),
            "load_slots_unexpected_sample": [
                f"{namespace}:{slot}"
                for namespace, slot in unexpected_slots[: self._SAMPLE_LIMIT]
            ],
            "owned_unexpected": self.owned_unexpected,
            "owned_unexpected_sample": self._unexpected_samples,
            "rank_unowned_samples": self._rank_unowned_samples,
            "rank_unowned_skipped": self.rank_unowned_skipped,
            "schema": "tokenspeed.qualification",
            "schema_version": 1,
            "stage_count": self.stage_plan.stage_count,
            "stage_id": self.stage_plan.stage_id,
            "stage_unowned_samples": self._stage_unowned_samples,
            "stage_unowned_skipped": self.stage_unowned_skipped,
            "status": "failure" if failed else "success",
            "tp_rank": getattr(getattr(self.mapping, "attn", None), "tp_rank", None),
        }
        log = logger.error if failed else logger.info
        log(
            "%s",
            json.dumps(
                payload,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        )
        if failed:
            raise RuntimeError(
                "Kimi-K3 strict weight load audit failed: "
                f"stage={self.stage_plan.stage_id} "
                f"owned_missing={len(missing)} "
                f"owned_unexpected={self.owned_unexpected} "
                f"contract_violations={self.contract_violations} "
                f"owned_duplicate={len(duplicate_sources)} "
                f"load_slots_missing={len(missing_slots)} "
                f"load_slots_duplicate={len(duplicate_slots)} "
                f"load_slots_unexpected={len(unexpected_slots)} "
                f"owned_target_load_mismatch={len(target_load_errors)} "
                f"output_attn_res_errors={output_attn_res_errors}"
            )


# ===----------------------------------------------------------------------=== #
# Multimodal vision path
# ===----------------------------------------------------------------------=== #


class KimiK3Vision(MoonViTVisionPath):
    """K3 MoonViT3d tower and patchmergerv2 projector.

    The encoder decomposition intentionally matches Kimi-K2.5: patch embedding
    and patch merging stay eager while only the shape-stable transformer block
    loop is captured. Keeping this object separate from the text wrapper also
    lets the top-level ``image_encoder`` callable be replaced by ModelExecutor's
    CUDA-graph wrapper without changing checkpoint parameter names.
    """

    @staticmethod
    def checkpoint_parameter_name(name: str) -> str:
        name = name.replace("wqkv.", "attn.qkv_proj.")
        name = name.replace("wo.", "attn.proj.")
        name = name.replace("mm_projector.proj.0", "mm_projector.linear_1")
        return name.replace("mm_projector.proj.2", "mm_projector.linear_2")

    def load_weight(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        params_dict: dict[str, nn.Parameter],
    ) -> str:
        name = self.checkpoint_parameter_name(name)
        if name not in params_dict:
            raise ValueError(f"Weight {name} not found in Kimi-K3 vision model")
        param = params_dict[name]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)
        return name


# ===----------------------------------------------------------------------=== #
# Text sublayers: dense MLP (SiTU), NoPE-MLA attention, AttnRes helper
# ===----------------------------------------------------------------------=== #


class KimiLinearMLP(nn.Module):
    """Dense / shared-expert MLP with the SiTU (SituGLU) activation.

    Mirrors ``DeepseekV3MLP`` (gate_up_proj + down_proj) but swaps SiLU for the
    Kimi SiTU activation. ``down_proj`` normally reduces its partial sum in
    place because Kimi-K3 runs the AttnRes residual path outside
    ``CommManager`` (see decision D4). EP Kimi can defer the shared-expert
    reduction so one Iris launch reduces it together with the routed latent.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        reduce_results: bool = True,
        is_shared_expert: bool = False,
        activation_situ_beta: float = 1.0,
        activation_situ_linear_beta: float | None = None,
    ) -> None:
        super().__init__()
        self.mapping = mapping
        if is_shared_expert:
            tp_rank = mapping.moe.tp_ep_rank
            tp_size = mapping.moe.tp_ep_size
            tp_group = mapping.moe.tp_ep_group
        else:
            tp_rank = mapping.dense.tp_rank
            tp_size = mapping.dense.tp_size
            tp_group = mapping.dense.tp_group

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            tp_size=tp_size,
            tp_rank=tp_rank,
            tp_group=tp_group,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            reduce_results=reduce_results,
            tp_size=tp_size,
            tp_rank=tp_rank,
            tp_group=tp_group,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act_fn = SituAndMul(
            beta=activation_situ_beta, linear_beta=activation_situ_linear_beta
        )
        self.is_shared_expert = is_shared_expert

    def forward(
        self, x: torch.Tensor, down_out: torch.Tensor | None = None
    ) -> torch.Tensor:
        if x.size(0) == 0:
            return x
        if self.is_shared_expert:
            x = kimi3_shared_situ_projection(
                x,
                self.gate_up_proj.weight,
                beta=self.act_fn.beta,
                linear_beta=self.act_fn.linear_beta,
            )
            x = kimi3_shared_down_projection(
                x,
                self.down_proj.weight,
                out=down_out,
            )
            if self.down_proj.reduce_results and self.down_proj.tp_size > 1:
                x = all_reduce(x, self.down_proj.tp_group)
            return x
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        if down_out is not None:
            # Direct-write partial (unquantized bf16 shared experts only):
            # lands the TP partial straight into the fused-AR lane slice.
            torch.mm(x, self.down_proj.weight.t(), out=down_out)
            return down_out
        x, _ = self.down_proj(x)
        return x


class KimiLinearMLAAttention(DeepseekV3AttentionMLA):
    """Kimi-K3 full-attention layer: NoPE MLA + optional sigmoid output gate.

    Reuses ``DeepseekV3AttentionMLA`` wholesale (absorbed decode, chunked
    prefill, MLA kernels, latent KV pool) with two K3 deltas:

    * **NoPE** via the parent's ``skip_rope=True`` — no rotary embedding is
      built and every rope application in the parent is already guarded by
      ``self.rotary_emb is not None``.
    * **Output gate** (``mla_use_output_gate``): ``attn_out *= sigmoid(g_proj(x))``
      injected before the single ``o_proj`` call.

    ``reduce_attn_results=True`` (unlike DeepSeek's deferred RSAG reduce) so
    ``o_proj`` all-reduces here — the AttnRes path does not use
    ``CommManager`` to fold the attention comm into the residual.
    """

    def __init__(
        self,
        config: KimiLinearConfig,
        mapping: Mapping,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
        max_position_embeddings: int = 8192,
        quant_config: QuantizationConfig | None = None,
        layer_id=None,
        prefix: str = "",
        reduce_attn_results: bool = True,
        alt_stream: torch.cuda.Stream | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            config=config,
            mapping=mapping,
            hidden_size=hidden_size,
            num_heads=num_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            layer_id=layer_id,
            prefix=prefix,
            reduce_attn_results=reduce_attn_results,
            alt_stream=alt_stream,
            skip_rope=True,  # K3 MLA is NoPE (mla_use_nope=True)
        )
        self.use_output_gate = config.mla_use_output_gate
        if self.use_output_gate:
            assert q_lora_rank is not None, "gated MLA assumes the q-lora path"
            # The gate projection shares its input with the a-projections, so
            # its per-rank shard rides the same GEMV: one weight laid out as
            # [q_a | kv_a+rope | g_shard] x hidden. (The dsv3 min-latency
            # kernel is shape-locked to 2112 rows and measures below nvjet's
            # effective bandwidth here anyway.)
            self._qkv_a_width = q_lora_rank + kv_lora_rank + qk_rope_head_dim
            self._gate_width = num_heads * v_head_dim // mapping.attn.tp_size
            self.fused_qkv_a_proj_with_mqa = DeepseekV3FusedQkvAProjWithMqa(
                hidden_size,
                self._qkv_a_width + self._gate_width,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("fused_qkv_a_proj_with_mqa", prefix),
            )

    def _project_q_latent_gated(
        self,
        hidden_states: torch.Tensor,
        ctx: "ForwardContext",
        comm_manager: CommManager,
        block_scale: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Project MLA Q, latent KV, and the local output gate in one GEMM.

        Returns:
            query: Normalized and projected MLA query.
            latent_cache: Compressed latent KV and RoPE cache row.
            gate: Local output-gate shard.
            absorbed_query: Optional decode query projected into latent key space.
        """
        if block_scale is not None:
            qkv_gate = self.fused_qkv_a_proj_with_mqa(
                hidden_states, block_scale, torch.bfloat16
            )
            qkv_gate = comm_manager.pre_attn_comm(qkv_gate, ctx)
            q_a, latent_cache, gate = qkv_gate.split(
                [
                    self.q_lora_rank,
                    self.kv_lora_rank + self.qk_rope_head_dim,
                    self._gate_width,
                ],
                dim=-1,
            )
        else:
            projection = kimi3_mla_qkv_gate_projection(
                hidden_states,
                self.fused_qkv_a_proj_with_mqa.weight,
                self._qkv_a_width,
            )
            if projection.packed is not None:
                qkv_gate = comm_manager.pre_attn_comm(projection.packed, ctx)
                q_a, latent_cache, gate = qkv_gate.split(
                    [
                        self.q_lora_rank,
                        self.kv_lora_rank + self.qk_rope_head_dim,
                        self._gate_width,
                    ],
                    dim=-1,
                )
            else:
                qkv = comm_manager.pre_attn_comm(projection.qkv, ctx)
                q_a, latent_cache = qkv.split(
                    [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                    dim=-1,
                )
                gate = projection.gate
        kv_a = latent_cache[..., : self.kv_lora_rank]
        projection = mla_normalize_project_query(
            q_a,
            kv_a,
            self.fused_qk_layernorm.weight_q_a,
            self.fused_qk_layernorm.weight_kv_a,
            self.q_b_proj.weight,
            eps=self.q_a_layernorm.variance_epsilon,
            prepare_absorbed_query=ctx is not None and ctx.num_extends == 0,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
        )
        return projection.query, latent_cache, gate, projection.absorbed_query

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: "ForwardContext",
        out_cache_loc: torch.Tensor,
        comm_manager,
        block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states
        if self.use_output_gate:
            q, latent_cache, gate, absorbed_query = self._project_q_latent_gated(
                hidden_states, ctx, comm_manager, block_scale
            )
        else:
            q, latent_cache = self._project_q_latent(
                hidden_states, ctx, comm_manager, block_scale
            )
            gate = None
            absorbed_query = None
        fuse_value_gate = (
            gate is not None
            and ctx.num_extends == 0
            and ctx.attn_backend.supports_mla_projected_value_decode
        )
        attn_output = self._attn(
            positions,
            q,
            latent_cache,
            ctx,
            out_cache_loc,
            output_gate=gate if fuse_value_gate else None,
            absorbed_query=absorbed_query,
        )
        if gate is not None and not fuse_value_gate:
            # Fused in-place fp32 sigmoid+mul; the gate shard matches the
            # head-sharded attn_output.
            attn_output = sigmoid_mul(attn_output, gate)
        output, _ = self.o_proj(attn_output)
        return output


def _sliced_scratch(like: torch.Tensor, slot: int, n_tokens: int):
    """The (m, s, acc) scratch views for the first ``n_tokens`` rows."""
    m, s_, acc = _attnres_scratch(like, slot=slot)
    return m[:n_tokens], s_[:n_tokens], acc[:n_tokens]


def _apply_attn_res(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj: nn.Module,
    norm: RMSNorm,
    num_valid_blocks: int,
    out_norm: RMSNorm | None = None,
) -> torch.Tensor:
    """AttnRes mixing: a learned softmax attention over block-residual snapshots.

    Candidates are the ``num_valid_blocks`` historical snapshots
    ``block_residual[:num_valid_blocks]`` plus the current ``prefix_sum``, mixed
    by a learned per-block weight ``softmax(RMSNorm(v) @ (norm.weight *
    proj.weight))`` (mirrors the checkpoint's ``modeling_kimi.py``), replacing
    the plain residual add on the Kimi-K3 AttnRes path. The fused ``attn_res``
    kernel does the whole mix in fp32 (it sits on the global residual backbone,
    where bf16 rounding would drift the stream), with a torch fallback for
    unsupported shapes. Both paths are CUDA-graph capture-compatible.

    ``block_residual`` is block-major ``[num_blocks, T, hidden]``. When
    ``out_norm`` is given (same eps as ``norm``), the following RMSNorm is
    fused into the kernel epilogue and the normed mix is returned.
    """
    if num_valid_blocks <= 0:
        return prefix_sum if out_norm is None else out_norm(prefix_sum)
    return attn_res_fwd(
        prefix_sum,
        block_residual[:num_valid_blocks],
        proj.weight.reshape(-1),
        norm.weight,
        norm.variance_epsilon,
        out_norm_weight=None if out_norm is None else out_norm.weight,
    )


def _situ_betas(config: KimiLinearConfig) -> tuple[float, float | None]:
    """(situ_beta, situ_linear_beta); every K3 MLP runs SiTU, so fail loud
    on any other ``hidden_act`` instead of silently running SiTU anyway."""
    if config.hidden_act != "situ":
        raise ValueError(
            f"KimiLinear MLPs only implement the 'situ' activation, got "
            f"{config.hidden_act!r}"
        )
    return (config.activation_situ_beta, config.activation_situ_linear_beta)


# ===----------------------------------------------------------------------=== #
# Text decoder layers
# ===----------------------------------------------------------------------=== #


class KimiKDAMergedProj(nn.Module):
    """Merged bf16 KDA input projections: ``[q | k | v | g | f_a | b]``, one GEMM.

    All six consume the post-norm hidden states (``self_attn`` is on the
    quantization ignore list, so everything is plain bf16). q/k/v/g/b shard
    per-head over the attention TP group; ``f_a`` (low-rank decay-gate down
    projection) is replicated, so each rank carries a full copy. The ``q|k|v``
    slice reproduces the layout the hybrid backend's conv expects; ``g``,
    ``f_a`` and ``b`` (beta logits) ride along as strided slices, replacing two
    extra latency-bound GEMVs per layer. Rows are padded to a multiple of 16:
    off-multiple row counts fall off cublasLt's fast M=1 tactic (30us vs 13us
    at 6284 vs 6288 x 7168). Loader ``shard_id`` in {"q","k","v","g","f_a","b"}.
    """

    _ROW_ALIGN = 16

    def __init__(
        self,
        hidden_size: int,
        proj: int,
        num_heads: int,
        head_dim: int,
        tp_rank: int,
        tp_size: int,
    ) -> None:
        super().__init__()
        self.proj_local = proj // tp_size
        self.local_num_heads = num_heads // tp_size
        self.head_dim = head_dim
        self.tp_rank = tp_rank
        p = self.proj_local
        self._offsets = {
            "q": 0,
            "k": p,
            "v": 2 * p,
            "g": 3 * p,
            "f_a": 4 * p,
            "b": 4 * p + head_dim,
        }
        self._rows = {
            "q": p,
            "k": p,
            "v": p,
            "g": p,
            "f_a": head_dim,
            "b": self.local_num_heads,
        }
        used = 4 * p + head_dim + self.local_num_heads
        total = ceil_div(used, self._ROW_ALIGN) * self._ROW_ALIGN
        self.used_rows = used
        # Explicit bf16: default-dtype fp32 would cost +6 GiB/rank and starve the KV budget.
        self.weight = nn.Parameter(
            torch.empty(total, hidden_size, dtype=torch.bfloat16)
        )
        # Padding rows are never read back, but keep them finite.
        self.weight.data[used:].zero_()
        self.weight.weight_loader = self._load_weight

    def _load_weight(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: str
    ) -> None:
        rows = self._rows[shard_id]
        # f_a is replicated (full copy per rank); the rest are row-sharded.
        src = (
            loaded_weight
            if shard_id == "f_a"
            else loaded_weight.narrow(0, self.tp_rank * rows, rows)
        )
        start = self._offsets[shard_id]
        param.data[start : start + rows].copy_(src)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Registry-dispatched: rowcta at M=1, cublasLt otherwise.
        out = decode_gemv(x, self.weight)
        p = self.proj_local
        mixed_qkv = out[:, : 3 * p]
        # No-op at decode (single row); prefill pays a small copy for the conv.
        if not mixed_qkv.is_contiguous():
            mixed_qkv = mixed_qkv.contiguous()
        f_a_end = 4 * p + self.head_dim
        return (
            mixed_qkv,
            out[:, 3 * p : 4 * p],
            out[:, 4 * p : f_a_end],
            out[:, f_a_end : f_a_end + self.local_num_heads],
        )


class KimiLinearKDA(nn.Module):
    """KDA (Kimi Delta Attention) linear-attention sublayer.

    Gated delta-rule: ``q/k/v_proj`` + short causal conv (SiLU), a decay gate
    ``f_b(f_a(x))`` combined with a **per-head** ``A_log[num_heads]`` (stored in
    the checkpoint as a zero-padded ``[head_dim]`` buffer) / per-(head,
    channel) ``dt_bias``, a per-head ``beta``, then the gated-delta scan (``fla``
    ``chunk_kda`` for prefill), a gated RMSNorm with the ``g_proj`` sigmoid output
    gate, and ``o_proj``.

    The layer owns the projections + gates + output norm and routes the conv +
    gated-delta scan + conv/recurrent state cache through the hybrid attention
    backend (``ctx.attn_backend`` -> ``MambaAttnBackend`` KDA branch), mirroring
    ``Qwen3_5GatedDeltaNet``. The ``q/k/v_conv1d_weight`` parameters only hold the
    conv kernels; the convolution itself runs in the backend.
    """

    def __init__(
        self,
        config: KimiLinearConfig,
        mapping: Mapping,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.layer_id = layer_id

        la = config.linear_attn_config
        if not la.get("use_full_rank_gate", False):
            # The reference has a low-rank g_a_proj/g_b_proj output-gate variant;
            # only the full-rank g_proj (what the K3 checkpoint uses) is wired
            # here. Fail loudly instead of surfacing missing-weight errors.
            raise NotImplementedError(
                "KimiLinearKDA only implements the full-rank output gate "
                "(linear_attn_config.use_full_rank_gate=True)."
            )
        self.num_heads = la["num_heads"]
        self.head_dim = la["head_dim"]
        self.conv_size = la["short_conv_kernel_size"]
        self.gate_lower_bound = la.get("gate_lower_bound")
        proj = self.num_heads * self.head_dim
        hidden = config.hidden_size

        tp_rank = mapping.attn.tp_rank
        tp_size = mapping.attn.tp_size
        tp_group = mapping.attn.tp_group
        self.local_num_heads = self.num_heads // tp_size
        proj_local = proj // tp_size

        def _col(in_f, out_f, name):
            return ColumnParallelLinear(
                in_f,
                out_f,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix(name, prefix),
                tp_rank=tp_rank,
                tp_size=tp_size,
                tp_group=tp_group,
            )

        # One merged GEMM replaces four per-head-sharded projections + the qkv concat.
        self.qkvgb_proj = KimiKDAMergedProj(
            hidden_size=hidden,
            proj=proj,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        # Decay-gate up projection (f_a and beta ride in the merged GEMM).
        self.f_b_proj = _col(self.head_dim, proj, "f_b_proj")

        # q/k/v short-conv kernels [proj_local, 1, W]. The conv itself runs in the
        # backend; these params only hold the weights. Named ``*_conv1d_weight``
        # (plain parameters, no wrapper module) -- the checkpoint's
        # ``<name>_conv1d.weight`` key is remapped in load_weights.
        self.q_conv1d_weight = nn.Parameter(torch.zeros(proj_local, 1, self.conv_size))
        self.k_conv1d_weight = nn.Parameter(torch.zeros(proj_local, 1, self.conv_size))
        self.v_conv1d_weight = nn.Parameter(torch.zeros(proj_local, 1, self.conv_size))

        # A_log is per-head [num_heads] (one log-decay per head). The
        # checkpoint stores it in a [head_dim]-sized buffer with only the first
        # num_heads entries populated (the rest zero-padded), so load this rank's
        # heads [local*rank : local*(rank+1)] and drop the padded tail. dt_bias
        # and the q/k/v conv weights are per-(head, channel), so they shard along
        # dim 0 by the attention TP rank.
        self.A_log = nn.Parameter(
            torch.zeros(self.local_num_heads, dtype=torch.float32)
        )
        _alog_start = self.local_num_heads * tp_rank
        _alog_n = self.local_num_heads

        def _a_log_head_loader(param, loaded_weight):
            param.data.copy_(loaded_weight.narrow(0, _alog_start, _alog_n))

        self.A_log.weight_loader = _a_log_head_loader
        self.dt_bias = nn.Parameter(torch.zeros(proj_local, dtype=torch.float32))
        self.dt_bias.weight_loader = sharded_weight_loader(0, tp_rank)
        for w in (self.q_conv1d_weight, self.k_conv1d_weight, self.v_conv1d_weight):
            w.weight_loader = sharded_weight_loader(0, tp_rank)
        # Fused (q, k, v) conv kernel bank; built once in post_load_weights.
        self.conv_weights: torch.Tensor | None = None

        self.o_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.o_proj = RowParallelLinear(
            proj,
            hidden,
            bias=False,
            reduce_results=False,  # layer-level fused AR+residual owns the reduce
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

    def fuse_conv_weights(self) -> None:
        """Concatenate the loaded q/k/v conv kernels into ``self.conv_weights``."""
        self.conv_weights = torch.cat(
            (self.q_conv1d_weight, self.k_conv1d_weight, self.v_conv1d_weight), dim=0
        ).squeeze(1)

    def _project_qkvfab(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project every KDA hidden-state consumer in one decode GEMV."""
        proj_local = self.local_num_heads * self.head_dim
        output = kimi3_qkvfab_projection(
            hidden_states,
            self.qkvgb_proj.weight,
        )
        f_a_end = 4 * proj_local + self.head_dim
        return (
            output[:, : 3 * proj_local].contiguous(),
            output[:, 3 * proj_local : 4 * proj_local],
            output[:, 4 * proj_local : f_a_end],
            output[:, f_a_end : self.qkvgb_proj.used_rows],
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: "ForwardContext",
        out_cache_loc: torch.Tensor,
        comm_manager,
        block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states

        h = hidden_states
        num_tokens = h.shape[0]
        hn, hd = self.local_num_heads, self.head_dim
        # The hybrid backend re-splits key_dim/value_dim by attn_tp_size to
        # recover the per-rank head count, so pass the FULL (pre-TP) projection
        # width. The projected tensors (mixed_qkv, g_raw, beta) and the returned
        # core_out are already per-rank from their column-parallel layers, so the
        # output reshape below uses the per-rank head count ``hn``.
        proj = self.num_heads * hd

        # Raw (pre-conv) q/k/v projections concatenated; the hybrid backend runs
        # the short causal conv (+ SiLU) and manages the conv / recurrent state
        # cache (KDA branch of MambaAttnBackend). g_raw is the raw decay-gate
        # input, beta the per-head logits (sigmoid applied in-kernel).
        mixed_qkv, out_gate, f_a_out, beta = self._project_qkvfab(h)
        # f_b runs inside the backend: fused into the decode scan kernel, a
        # plain GEMV on the prefill path.
        # Fused [3*proj, k] conv kernel bank, built once in post_load_weights.
        conv_weights = self.conv_weights

        core_out = ctx.attn_backend.forward(
            q=None,
            k=None,
            v=None,
            layer=None,
            out_cache_loc=out_cache_loc,
            token_to_kv_pool=ctx.token_to_kv_pool,
            forward_mode=ctx.forward_mode,
            bs=ctx.bs,
            mixed_qkv=mixed_qkv,
            conv_weights=conv_weights,
            bias=None,
            activation="silu",
            key_dim=proj,
            value_dim=proj,
            attention_tp_size=self.mapping.attn.tp_size,
            head_k_dim=hd,
            head_v_dim=hd,
            f_a_out=f_a_out,
            f_b_weight=self.f_b_proj.weight,
            beta_raw=beta,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            lower_bound=self.gate_lower_bound,
            layer_id=self.layer_id,
            seq_len=num_tokens,
        )

        # Per-head gated RMSNorm + sigmoid output gate in one kernel.
        core_out = rmsnorm_gated_sigmoid(
            core_out.reshape(num_tokens, hn * hd).contiguous(),
            out_gate.contiguous(),
            self.o_norm.weight,
            self.o_norm.variance_epsilon,
            hn,
            hd,
        )
        output, _ = self.o_proj(core_out)
        return output


class KimiLinearMoEGate(nn.Module):
    """Router for Kimi-K3 MoE: linear scorer + noaux_tc correction bias.

    Matches the checkpoint's ``block_sparse_moe.gate.{weight,e_score_correction_bias}``.
    Built inline (rather than reusing ``DeepseekV3.MoEGate``) because Kimi-K3's
    config uses ``num_experts`` where DeepSeek expects ``n_routed_experts``.
    """

    def __init__(self, hidden_size: int, num_experts: int) -> None:
        super().__init__()
        # Keep the checkpoint's BF16 weights at rest. Both the specialized AMD
        # decode GEMV and the portable fallback accumulate router logits in
        # FP32 before exact top-k selection.
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))
        self.e_score_correction_bias = nn.Parameter(
            torch.empty(num_experts, dtype=torch.float32)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return kimi3_router_projection(
            hidden_states,
            self.weight,
            enable_pdl=pdl_enabled(),
        )


# Decode-shape ceiling for the split AttnRes fast paths (partial/combine and
# the scratch that links them). Decode steps carry at most --max-num-seqs
# tokens (1 token per sequence, no speculative decoding), so any batch above
# this is a prefill chunk and takes the unsplit path; the scratch buffers are
# allocated to exactly this many rows. Raise together with --max-num-seqs.
ATTNRES_FAST_PATH_MAX_TOKENS = 32

_ATTNRES_SCRATCH: list | None = None


def _attnres_scratch(
    like: torch.Tensor, slot: int = 0, cap: int = ATTNRES_FAST_PATH_MAX_TOKENS
):
    """Shared (m, s, acc) scratch for the split attn_res mixing (bs <= cap).

    slot 0 = the layer's mlp-side mix; slot 1 = the next layer's attn-side mix
    (their lifetimes overlap, so they get separate buffers).
    """
    global _ATTNRES_SCRATCH
    sc = _ATTNRES_SCRATCH
    if (
        sc is None
        or sc[0][2].shape[1] != like.shape[-1]
        or sc[0][2].device != like.device
    ):
        sc = [
            (
                torch.empty(cap, dtype=torch.float32, device=like.device),
                torch.empty(cap, dtype=torch.float32, device=like.device),
                torch.empty(
                    cap, like.shape[-1], dtype=torch.float32, device=like.device
                ),
            )
            for _ in range(2)
        ]
        _ATTNRES_SCRATCH = sc
    return sc[slot]


class KimiLinearMoE(nn.Module):
    """Kimi-K3 MoE block: sigmoid / noaux_tc router + Latent MoE + shared experts.

    Structure:

    * **Router** ``KimiLinearMoEGate`` + ``TopK`` (grouped, ``n_group=topk_group=1``,
      sigmoid scoring with ``e_score_correction_bias`` — DeepSeek-V3 noaux_tc).
    * **Latent MoE**: routed experts run at ``routed_expert_hidden_size`` (3584),
      so ``routed_expert_down_proj`` (7168->3584) feeds the experts and
      ``routed_expert_up_proj``/``routed_expert_norm`` project back (7168).
    * **Routed experts** (MXFP4): AMD uses the native ``MoELayer`` plan wrapped
      by ``LatentMoELayer`` so Triton/Gluon owns EP8 dispatch and SiTU. Non-AMD
      platforms use flashinfer's TRTLLM-Gen SiTU MoE.
    * **Shared experts**: a plain ``KimiLinearMLP`` (SiTU).
    """

    def __init__(
        self,
        config: KimiLinearConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        layer_index: int = -1,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        # Router (gate+topk) and shared experts run on this stream during
        # graph capture, overlapped with the main-stream routed chain
        # (down_proj -> fused SiTU MoE -> up_proj). Collectives stay on the default
        # stream (aux-stream collectives can deadlock across ranks).
        self.stream_fork = StreamFork(alt_stream)
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token
        self.routed_scaling_factor = config.routed_scaling_factor
        # Latent MoE: routed experts run at routed_expert_hidden_size.
        self.routed_hidden = (
            config.routed_expert_hidden_size
            if config.routed_expert_hidden_size is not None
            else config.hidden_size
        )
        situ_beta, situ_linear_beta = _situ_betas(config)

        moe_backend = get_moe_backend()
        self.execution_plan = Kimi3MoEExecutionPlan.build(
            mapping,
            moe_backend,
            alt_stream,
            enforce_eager=bool(global_server_args_dict["enforce_eager"]),
        )
        # AUTO intentionally requests the flashinfer-backed SiTU plan when it was
        # registered at import time; AUTO cannot override MoELayer per model.
        self.use_trtllm_situ_moe = self.execution_plan.use_trtllm
        self.use_marlin_situ_moe = self.execution_plan.use_marlin
        if not self.execution_plan.use_native and not self.use_marlin_situ_moe:
            if not self.use_trtllm_situ_moe:
                raise RuntimeError(
                    "Kimi-K3 MXFP4 SiTU MoE requires the native, FlashInfer "
                    "TRT-LLM, or Marlin (Hopper W4A16) backend; no portable SiTU "
                    f"Triton fallback exists (selected MoE backend: "
                    f"{moe_backend.value!r})."
                )
            # Fail here with the actual reason instead of letting MoELayer's
            # kernel selection miss the (never-registered) SiTU kernel.
            reason = situ_moe_unavailable_reason()
            if reason is not None:
                raise RuntimeError(
                    "Kimi-K3's fused SiTU MoE requires flashinfer > 0.6.15 "
                    f"with native SiTU, unavailable: {reason}. Upgrade "
                    "flashinfer; no portable SiTU Triton fallback exists."
                )
            # Out-of-box tactics: seed the autotuner from the in-tree
            # swept table for this GPU/flashinfer combo, if one ships
            # (flashinfer's own heuristic mispicks MoE tactics at prefill
            # batch sizes). A lookup miss leaves the startup autotune
            # window to tune these shapes.
            load_packaged_flashinfer_tuning_cache(
                "kimi-k3", mapping.moe.ep_size, mapping.moe.tp_size
            )
        if not self.execution_plan.use_native:
            # Both the TRT-LLM and Marlin SiTU paths currently require a
            # replicated-token all-reduce topology (attn TP == MoE TP*EP);
            # Attn-DP/MoE-EP RSAG is not wired for either.
            if mapping.moe.has_tp_ep and mapping.attn.tp_size != mapping.moe.tp_ep_size:
                raise ValueError(
                    "Kimi-K3's SiTU MoE currently requires a "
                    "replicated-token all-reduce topology: attn TP size must equal "
                    "MoE TP*EP size. Attn-DP/MoE-EP RSAG is not supported."
                )
        self.gate = KimiLinearMoEGate(config.hidden_size, config.num_experts)

        self.topk = TopK(
            top_k=self.top_k,
            renormalize=config.moe_renormalize,
            use_grouped_topk=config.use_grouped_topk,
            num_expert_group=config.num_expert_group,
            num_fused_shared_experts=0,
            topk_group=config.topk_group,
            correction_bias=self.gate.e_score_correction_bias,
            routed_scaling_factor=self.routed_scaling_factor,
            output_format=TopKOutputFormat.STANDARD,
            # bf16 weights out: makes the fused SiTU kernel's cast a no-op.
            topk_weights_dtype=(
                torch.bfloat16 if self.use_trtllm_situ_moe else torch.float32
            ),
        )

        # AMD native and flashinfer's TRT-LLM SiTU MoE both consume K3's
        # precomputed sigmoid/noaux_tc TopK.
        self.experts = MoELayer(
            top_k=self.top_k,
            num_experts=self.num_experts,
            hidden_size=self.routed_hidden,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            layer_index=layer_index,
            prefix=prefix,
            tp_rank=mapping.moe.tp_rank,
            tp_size=mapping.moe.tp_size,
            ep_rank=mapping.moe.ep_rank,
            ep_size=mapping.moe.ep_size,
            activation="situ",
            activation_situ_beta=situ_beta,
            activation_situ_linear_beta=situ_linear_beta,
            routing_config={
                "n_group": config.num_expert_group,
                "topk_group": config.topk_group,
                "routed_scaling_factor": self.routed_scaling_factor,
                "normalize_topk_weights": config.moe_renormalize,
                "correction_bias": self.gate.e_score_correction_bias,
                "routing_method_type": RoutingMethodType.DeepSeekV3,
                "activation_situ_beta": situ_beta,
                "activation_situ_linear_beta": situ_linear_beta,
            },
            routing_mode="precomputed_topk",
            # Native gfx950 and Hopper Marlin both run A16W4 (bf16 activations);
            # FlashInfer's SiTU cubins require MXFP4 weights with MXFP8 acts.
            internal_activation_dtype_override=(
                "input"
                if self.execution_plan.use_native or self.execution_plan.use_marlin
                else "fp8" if self.execution_plan.use_trtllm else None
            ),
        )
        if self.experts.support_routing:
            raise RuntimeError(
                "Kimi-K3 requires a precomputed-TopK SiTU MoE kernel; the "
                "selected backend unexpectedly performs internal routing"
            )

        self.routed_expert_down_proj = Kimi3LatentProjection(
            config.hidden_size,
            self.routed_hidden,
            prefix=add_prefix("routed_expert_down_proj", prefix),
        )
        self.routed_expert_up_proj = Kimi3LatentProjection(
            self.routed_hidden,
            config.hidden_size,
            prefix=add_prefix("routed_expert_up_proj", prefix),
        )
        self.routed_expert_norm = (
            RMSNorm(self.routed_hidden, eps=config.rms_norm_eps)
            if config.latent_moe_use_norm
            else None
        )
        self.execution_plan = self.execution_plan.prepare_latent_fusion(
            mapping,
            lane_width=self.routed_hidden + config.hidden_size,
            has_latent_norm=self.routed_expert_norm is not None,
            max_token_num=max(
                int(global_server_args_dict["comm_fusion_max_num_tokens"]),
                1,
            ),
        )

        self._topk_ready = torch.cuda.Event() if alt_stream is not None else None

        # Shared experts (SiTU dense MLP over the full hidden size).
        self.shared_experts = KimiLinearMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size * config.num_shared_experts,
            mapping=mapping,
            quant_config=quant_config,
            prefix=add_prefix("shared_experts", prefix),
            # Partial sums only: the reduce happens in forward on the default
            # stream after the aux-stream join, or in the joint Iris reduction.
            reduce_results=False,
            is_shared_expert=True,
            activation_situ_beta=situ_beta,
            activation_situ_linear_beta=situ_linear_beta,
        )
        self.packed_input_projection_weight: torch.Tensor | None = None
        self.native_latent_moe = (
            LatentMoELayer(
                router=self.gate,
                topk=self.topk,
                routed_down_proj=self.routed_expert_down_proj,
                experts=self.experts,
                routed_norm=self.routed_expert_norm,
                routed_up_proj=self.routed_expert_up_proj,
                shared_experts=self.shared_experts,
                shared_reduce=(
                    None
                    if self.execution_plan.joint_moe_reduce
                    else self._reduce_shared
                ),
                joint_reduce=(
                    partial(all_reduce_two, group=mapping.moe.ep_group)
                    if self.execution_plan.joint_moe_reduce
                    else None
                ),
                shared_expert_stream=(
                    alt_stream if self.execution_plan.overlap_shared_experts else None
                ),
                expert_parallel_group=mapping.moe.ep_group,
                input_projections=self._latent_input_projections,
            )
            if self.execution_plan.use_native
            else None
        )
        # Dry-run exact registry selection for the fused decode pipeline.
        self._use_fused_decode_pipeline = (
            self.execution_plan.joint_moe_reduce
            and latent_moe_decode_pipeline_available(
                self.gate.weight,
                self.routed_expert_down_proj.weight,
                self.shared_experts.gate_up_proj.weight,
                self.shared_experts.down_proj.weight,
                self.experts.w13_weight,
                self.experts.w13_weight_scale,
                self.experts.w2_weight,
                self.experts.w2_weight_scale,
                self.experts.plan,
                topk=self.top_k,
            )
        )

    def pack_input_projection_weights(self) -> None:
        """Back the router, routed-down, and shared gate/up weights with one tensor.

        The three projections consume the same activation and reduce over the
        same width, so one ``[experts + latent + 2 * shared, hidden]`` weight
        lets a single GEMM replace three. Each module keeps a contiguous row
        view of that tensor, so the rebinding adds no steady-state memory and
        leaves the separate composition working unchanged.
        """
        modules = (
            self.gate,
            self.routed_expert_down_proj,
            self.shared_experts.gate_up_proj,
        )
        # Held so the row views stay alive; deliberately not a registered
        # buffer, which would duplicate every projection in the state dict.
        # Any layout a packed GEMM cannot read is caught downstream by
        # ``packed_projection_weight_view``, leaving the composition in place.
        self.packed_input_projection_weight = torch.cat(
            [module.weight for module in modules], dim=0
        )
        offset = 0
        for module in modules:
            rows = module.weight.shape[0]
            module.weight.data = self.packed_input_projection_weight.narrow(
                0, offset, rows
            )
            offset += rows

    def _latent_input_projections(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Project the router, routed latent, and shared partial in one pass.

        Returns ``None`` before the projection weights are concatenated, which
        leaves the caller on the separate per-module projections.
        """
        if self.packed_input_projection_weight is None:
            return None
        router_logits, routed_input, shared_input = latent_moe_input_projections(
            hidden_states,
            self.gate.weight,
            self.routed_expert_down_proj.weight,
            self.shared_experts.gate_up_proj.weight,
            gate_clamp=self.shared_experts.act_fn.beta,
            up_clamp=self.shared_experts.act_fn.linear_beta,
        )
        # The shared experts hold reduce_results=False, so this partial is
        # reduced by the layer's shared or joint reducer, not here.
        shared_output = kimi3_shared_down_projection(
            shared_input,
            self.shared_experts.down_proj.weight,
        )
        return router_logits, routed_input, shared_output

    def _routed_experts(
        self,
        routed_in: torch.Tensor,
        topk_output: TopKOutput,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
        skip_reduce: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the precomputed-TopK SiTU MoE (TRT-LLM cubin or Hopper Marlin)."""
        if not self.use_trtllm_situ_moe and not self.use_marlin_situ_moe:
            raise RuntimeError(
                "Kimi-K3 has no portable SiTU Triton fallback; use the native, "
                "FlashInfer TRT-LLM, or Marlin SiTU MoE path."
            )
        out = self.experts(
            hidden_states=routed_in,
            topk_output=topk_output,
            num_global_tokens=num_global_tokens,
            max_num_tokens_per_gpu=max_num_tokens_per_gpu,
        )
        # Each TP rank owns an intermediate shard; each EP rank owns a
        # contiguous expert shard. The routed kernel returns that rank's
        # partial contribution, which must be combined before the replicated
        # norm/up projection (skip_reduce: caller's fused AR covers it).
        if self.mapping.moe.has_tp_ep and not skip_reduce:
            out = all_reduce(out, self.mapping.moe.tp_ep_group)
        return out

    def _forward_fused_decode_pipeline(
        self,
        hidden_states: torch.Tensor,
        prefix_sum: torch.Tensor,
    ) -> torch.Tensor:
        """Run the fused multi-launch K3 decode pipeline.

        The input-projection operation produces router logits, the routed
        latent, and the activated shared input from one packed weight. Routed
        W13 and SiTU run separately; the next launch computes routed W2 beside
        the shared down projection. One collective reduces both partials. The
        final routed up projection adds ``prefix_sum`` and the shared output in
        its epilogue when a specialized implementation is available. Top-k and
        the optional routed norm remain separate launches.
        """

        router_logits, routed_input, shared_input = latent_moe_input_projections(
            hidden_states,
            self.gate.weight,
            self.routed_expert_down_proj.weight,
            self.shared_experts.gate_up_proj.weight,
            gate_clamp=self.shared_experts.act_fn.beta,
            up_clamp=self.shared_experts.act_fn.linear_beta,
        )
        topk_output = self.topk(hidden_states, router_logits)
        routed_output = hidden_states.new_empty(
            (hidden_states.shape[0], self.routed_expert_down_proj.weight.shape[0])
        )
        shared_output = hidden_states.new_empty(
            (hidden_states.shape[0], self.shared_experts.down_proj.weight.shape[0])
        )
        routed_latent, shared_output = latent_moe_expert_shared(
            routed_input,
            self.experts.w13_weight,
            self.experts.w13_weight_scale,
            self.experts.w2_weight,
            self.experts.w2_weight_scale,
            topk_output.topk_weights,
            topk_output.topk_ids,
            shared_input,
            self.shared_experts.down_proj.weight,
            activation_clamp=float(self.experts.activation_situ_beta),
            linear_clamp=self.experts.activation_situ_linear_beta,
            expert_start=self.experts.ep_rank * self.experts.num_local_experts,
            w13_interleaved=self.experts.w13_input_layout == "interleaved",
            routed_out=routed_output,
            shared_out=shared_output,
        )
        shared_output, routed_latent = all_reduce_two(
            shared_output,
            routed_latent,
            group=self.mapping.moe.ep_group,
        )
        if self.routed_expert_norm is not None:
            routed_latent = self.routed_expert_norm(routed_latent)
        return self.native_latent_moe.finalize_output(
            routed_latent,
            prefix_sum,
            shared_output,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        prefix_sum: torch.Tensor,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
    ) -> torch.Tensor:
        """Routed + shared experts, accumulated onto ``prefix_sum``.

        Returns the new prefix (``prefix_sum + routed + shared``); at bs=1
        the up-projection's store performs the accumulate in-kernel.
        """
        if self.native_latent_moe is not None:
            if self._use_fused_decode_pipeline and hidden_states.shape[0] == 1:
                return self._forward_fused_decode_pipeline(hidden_states, prefix_sum)
            return self.native_latent_moe(
                hidden_states,
                num_global_tokens=num_global_tokens,
                max_num_tokens_per_gpu=max_num_tokens_per_gpu,
                prefix_sum=prefix_sum,
            )

        num_tokens, hidden_size = hidden_states.shape
        if num_tokens == 0:
            return prefix_sum

        # Router runs uncontended on main (3us; on aux it starves to 14us
        # under concurrent GEMMs). Topk is a single small CTA, so it overlaps
        # down_proj from the aux stream, followed by the shared chain.
        router_logits = self.gate(hidden_states)
        lane = allreduce_fusion_lane(
            hidden_states,
            self.routed_hidden + hidden_size,
            enabled=self.execution_plan.fused_moe_ar,
        )
        if lane is not None:
            self.experts._situ_output_buffer = lane[:, : self.routed_hidden]
        else:
            self.experts._situ_output_buffer = None
        with self.stream_fork.scope(enable=get_is_capture_mode()) as fork:
            with fork.branch():
                topk_output = self.topk(hidden_states, router_logits)
                if self._topk_ready is not None and fork._active:
                    self._topk_ready.record(torch.cuda.current_stream())
                shared_partial = self.shared_experts(
                    hidden_states,
                    down_out=(
                        lane[:, self.routed_hidden :] if lane is not None else None
                    ),
                )
            routed_in = decode_gemv(hidden_states, self.routed_expert_down_proj.weight)
            if self._topk_ready is not None and fork._active:
                self._topk_ready.wait(torch.cuda.current_stream())
            routed_out = self._routed_experts(
                routed_in,
                topk_output,
                num_global_tokens,
                max_num_tokens_per_gpu,
                skip_reduce=self.execution_plan.fused_moe_ar,
            )
            if not self.execution_plan.fused_moe_ar:
                if self.routed_expert_norm is not None:
                    routed_out = self.routed_expert_norm(routed_out)
                routed_out = self.routed_expert_up_proj(routed_out)[0]
        if self.execution_plan.fused_moe_ar:
            routed_out, shared_out = kimi3_join_reduce_moe(
                routed_out,
                shared_partial,
                lane=lane,
                routed_hidden=self.routed_hidden,
                routed_norm=self.routed_expert_norm,
                group=self.mapping.moe.tp_ep_group,
                enable_lane_norm=self.execution_plan.lane_latent_norm_ar,
                max_token_num=self.execution_plan.comm_fusion_max_num_tokens,
            )
            return kimi3_latent_projection_add3(
                routed_out,
                self.routed_expert_up_proj.weight,
                prefix_sum,
                shared_out,
            ).view(num_tokens, hidden_size)
        else:
            shared_out = self._reduce_shared(shared_partial)
        # routed_scaling_factor already applied in TopK; not re-applied here
        # (matches the reference).
        return add3(
            prefix_sum,
            routed_out.view(num_tokens, hidden_size),
            shared_out.view(num_tokens, hidden_size),
        )

    def _reduce_shared(self, shared_partial: torch.Tensor) -> torch.Tensor:
        """Reduce the shared experts' TP partial on the current (default) stream."""
        if self.mapping.moe.tp_ep_size > 1:
            return all_reduce(shared_partial, self.mapping.moe.tp_ep_group)
        return shared_partial


class KimiLinearDecoderLayer(nn.Module):
    """Kimi-K3 decoder layer: KDA/MLA dispatch + dense/MoE FFN + AttnRes.

    One class for both layer types — the AttnRes data flow is identical, only
    ``self_attn`` differs (dispatched by ``config.is_kda_layer(layer_id)``). The
    AttnRes path replaces the plain pre-norm residual with a
    learned block-residual mixing (``_apply_attn_res``) and runs *outside*
    ``CommManager`` fusion: attention/FFN output projections all-reduce in place
    (``reduce_results=True``) and the residual is threaded explicitly as the
    per-token ``block_residual`` buffer.
    """

    def __init__(
        self,
        config: KimiLinearConfig,
        mapping: Mapping,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.layer_id = layer_id

        # --- attention: KDA (linear) or NoPE-MLA (full); both under "self_attn" ---
        attn_prefix = add_prefix("self_attn", prefix)
        if config.is_kda_layer(layer_id):
            self.self_attn = KimiLinearKDA(
                config, mapping, layer_id, quant_config, attn_prefix
            )
        else:
            self.self_attn = KimiLinearMLAAttention(
                config=config,
                mapping=mapping,
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                max_position_embeddings=config.max_position_embeddings,
                quant_config=quant_config,
                layer_id=layer_id,
                prefix=attn_prefix,
                reduce_attn_results=False,  # layer fused AR+residual reduces
                alt_stream=alt_stream,
            )

        # --- FFN: dense MLP (first_k_dense_replace) or MoE block ---
        self.is_moe_layer = (
            config.num_experts is not None
            and layer_id >= config.first_k_dense_replace
            and layer_id % config.moe_layer_freq == 0
        )
        situ_beta, situ_linear_beta = _situ_betas(config)
        if self.is_moe_layer:
            # Named for the checkpoint index; not aliased as self.mlp (double
            # registration would duplicate every MoE param in state_dict).
            self.block_sparse_moe = KimiLinearMoE(
                config,
                mapping,
                quant_config,
                layer_id,
                add_prefix("block_sparse_moe", prefix),
                alt_stream=alt_stream,
            )
        else:
            self.mlp = KimiLinearMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                mapping=mapping,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                reduce_results=True,
                activation_situ_beta=situ_beta,
                activation_situ_linear_beta=situ_linear_beta,
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # --- AttnRes modules ---
        block = config.attn_res_block_size
        self.is_block_write_layer = layer_id % block == 0
        self.block_write_idx = layer_id // block
        self.prev_valid_blocks = ceil_div(layer_id, block)
        self.self_attention_res_norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp_res_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attention_res_proj = ReplicatedLinear(
            config.hidden_size,
            1,
            bias=False,
            prefix=add_prefix("self_attention_res_proj", prefix),
        )
        self.mlp_res_proj = ReplicatedLinear(
            config.hidden_size, 1, bias=False, prefix=add_prefix("mlp_res_proj", prefix)
        )

        # K3 AttnRes bypasses CommManager's fused residual, but the MLA attention
        # still uses it for the (no-op in AllReduce mode) pre_attn_comm.
        # Fused AR+residual for the attention reduce: ones-weight RMSNorm rides
        # the one-shot pattern; its norm output is discarded. Enabled when the
        # attention TP group's one-shot lane is armed.
        self._attn_ar_residual_fusion = (
            mapping.attn.tp_size > 1
            and prepare_all_reduce_lane(mapping.attn.tp_group, config.hidden_size)
            and prepare_all_reduce_fusion(
                mapping.attn.tp_group,
                config.hidden_size,
                max(
                    int(global_server_args_dict["comm_fusion_max_num_tokens"]),
                    1,
                ),
            )
        )
        self._dummy_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._dummy_norm.weight.data = torch.ones(
            config.hidden_size, dtype=torch.bfloat16
        )
        self._dummy_norm.weight.requires_grad_(False)

        self.attn_fork = StreamFork(alt_stream)
        # (proj_w_getter, norm, valid_blocks) for the NEXT layer's attn-side
        # mix; set by the backbone after all layers exist. The partial launches
        # from this (MoE) layer's aux branch, hidden under the routed experts.
        self._next_attn_mix = None
        # Precomputed rms_w * res_w products (filled in post_load_weights).
        self._attn_wp = None
        self._mlp_wp = None
        # True when the PREVIOUS layer precomputes our attn-side block partial.
        self._attn_split = False
        # True when the NEXT layer folds our routed+shared residual accumulate
        # into its attn-side combine (we return the parts unsummed).
        self.comm_manager = CommManager(
            mapping=mapping,
            layer_id=layer_id,
            is_moe=self.is_moe_layer,
            prev_is_moe=False,
            input_layernorm=self.input_layernorm,
            post_attn_layernorm=self.post_attention_layernorm,
        )

    def _reduce_attn_accumulate(
        self,
        attn_partial: torch.Tensor,
        prefix_sum: torch.Tensor | None,
        combine: tuple | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """All-reduce the attention partial and accumulate the residual.

        Small batches fold the residual add into the one-shot AR kernel;
        with ``combine = (scratch, res_w, rms_w, out_norm_w, eps)`` the
        mlp-side AttnRes prefix combine also rides its epilogue and the mixed
        hidden comes back as the second return (else None -- block-write
        layers, large batches and the plain-reduce fallback).
        """
        num_tokens = attn_partial.shape[0]
        if (
            prefix_sum is not None
            and self._attn_ar_residual_fusion
            and 0 < num_tokens
            and num_tokens <= global_server_args_dict["comm_fusion_max_num_tokens"]
        ):
            if combine is not None:
                from tokenspeed_kernel.ops.communication.trtllm import (
                    allreduce_residual_attnres_combine,
                )

                from tokenspeed.runtime.utils.pdl import pdl_enabled

                scratch, res_w, rms_w, out_norm_w, eps = combine
                h, residual_out = allreduce_residual_attnres_combine(
                    attn_partial,
                    prefix_sum,
                    res_w,
                    rms_w,
                    out_norm_w,
                    scratch=scratch,
                    rank=self.mapping.attn.tp_rank,
                    group=_get_process_group(self.mapping.attn.tp_group),
                    eps=eps,
                    max_token_num=global_server_args_dict["comm_fusion_max_num_tokens"],
                    launch_with_pdl=pdl_enabled(),
                )
                return residual_out, h
            _, residual_out, *_ = self._dummy_norm.forward_with_allreduce_fusion(
                self.mapping.attn.tp_rank,
                self.mapping.attn.tp_group,
                attn_partial,
                prefix_sum,
            )
            if residual_out is not None:
                return residual_out, None
        reduced = all_reduce(attn_partial, self.mapping.attn.tp_group)
        return (reduced if prefix_sum is None else prefix_sum + reduced), None

    def _mix_into_attention(
        self, hidden_states: torch.Tensor, block_residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """AttnRes entry: mix the residual candidates into the attention input.

        Returns ``(h, prefix_sum)`` -- with ``prefix_sum`` None at block-write
        layers (the snapshot consumed it).
        """
        prefix_sum = hidden_states
        n_tok = prefix_sum.shape[0]
        fast_mix = (
            self._attn_split
            and 0 < n_tok <= ATTNRES_FAST_PATH_MAX_TOKENS
            and prefix_sum.is_cuda
            and self.prev_valid_blocks > 0
        )
        if fast_mix:
            # The block partial was precomputed on the previous layer's aux
            # stream; only the prefix candidate is folded here.
            h = attnres_combine(
                prefix_sum,
                self._attn_wp,
                self.input_layernorm.weight,
                self.self_attention_res_norm.variance_epsilon,
                _sliced_scratch(prefix_sum, 1, n_tok),
                torch.empty_like(prefix_sum),
            )
        else:
            h = _apply_attn_res(
                prefix_sum,
                block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
                self.prev_valid_blocks,
                out_norm=self.input_layernorm,
            )
        if self.is_block_write_layer:
            block_residual[self.block_write_idx] = prefix_sum  # snapshot
            prefix_sum = None
        return h, prefix_sum

    @torch.no_grad()
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: "ForwardContext",
        out_cache_loc: torch.Tensor,
        block_residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, prefix_sum = self._mix_into_attention(hidden_states, block_residual)
        # The mlp-side mixing's block partial hides under attention on the aux
        # stream (blocks are final for this layer once the snapshot above ran);
        # the combine after the attention AR only touches the prefix candidate.
        mlp_valid_blocks = self.prev_valid_blocks + (
            1 if self.is_block_write_layer else 0
        )
        num_tokens = h.shape[0]
        split_mix = (
            0 < num_tokens <= ATTNRES_FAST_PATH_MAX_TOKENS
            and h.is_cuda
            and mlp_valid_blocks > 0
        )
        scratch = _sliced_scratch(h, 0, num_tokens) if split_mix else None
        # The next layer's attn-side partial reads the same (now final) block
        # set as our mlp-side partial, so both ride one sweep when armed.
        next_mix = self._next_attn_mix if split_mix else None
        sc1 = _sliced_scratch(h, 1, num_tokens) if next_mix is not None else None
        # The mlp-side combine (blocks partial + post-AR prefix) rides the
        # attention AR epilogue on the fused path.
        ar_combine = (
            (
                scratch,
                self.mlp_res_proj.weight.reshape(-1),
                self.mlp_res_norm.weight,
                self.post_attention_layernorm.weight,
                self.mlp_res_norm.variance_epsilon,
            )
            if split_mix
            else None
        )
        with self.attn_fork.scope(enable=get_is_capture_mode()) as fork:
            with fork.branch():
                if next_mix is not None:
                    nxt_layer, _ = next_mix
                    attnres_partial_dual(
                        block_residual[:mlp_valid_blocks],
                        self._mlp_wp,
                        nxt_layer._attn_wp,
                        self.mlp_res_norm.variance_epsilon,
                        scratch,
                        sc1,
                    )
                elif split_mix:
                    attnres_partial(
                        block_residual[:mlp_valid_blocks],
                        self._mlp_wp,
                        self.mlp_res_norm.variance_epsilon,
                        scratch,
                    )
            attn_out = self.self_attn(
                positions=positions,
                hidden_states=h,
                ctx=ctx,
                out_cache_loc=out_cache_loc,
                comm_manager=self.comm_manager,
            )
            prefix_sum, h_fused = self._reduce_attn_accumulate(
                attn_out, prefix_sum, combine=ar_combine
            )
        # --- mlp: AttnRes mixing -> norm -> FFN -> accumulate ---
        if h_fused is not None:
            h = h_fused
        elif split_mix:
            h = attnres_combine(
                prefix_sum,
                self._mlp_wp,
                self.post_attention_layernorm.weight,
                self.mlp_res_norm.variance_epsilon,
                scratch,
                torch.empty_like(prefix_sum),
            )
        else:
            h = _apply_attn_res(
                prefix_sum,
                block_residual,
                self.mlp_res_proj,
                self.mlp_res_norm,
                mlp_valid_blocks,
                out_norm=self.post_attention_layernorm,
            )
        if self.is_moe_layer:
            num_global_tokens, max_num_tokens_per_gpu = (
                self.comm_manager.get_num_tokens(ctx)
            )
            prefix_sum = self.block_sparse_moe(
                h,
                prefix_sum,
                num_global_tokens=num_global_tokens,
                max_num_tokens_per_gpu=max_num_tokens_per_gpu,
            )
        else:
            prefix_sum = prefix_sum + self.mlp(h)
        return prefix_sum, block_residual


# ===----------------------------------------------------------------------=== #
# Text backbone (KimiLinear)
# ===----------------------------------------------------------------------=== #


class KimiLinearModel(nn.Module):
    """Kimi-K3 text transformer: embedding + hybrid decoder layers + AttnRes.

    Runs the block-level attention-residual (AttnRes) data flow:
    a per-token ``block_residual`` buffer is threaded through the layers, and a
    final ``_apply_attn_res`` mixes the accumulated stream against the block
    snapshots before the output norm. The per-layer ``self_attn`` dispatch is
    ``KimiLinearDecoderLayer``'s job.
    """

    def __init__(
        self,
        config: KimiLinearConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.quant_config = quant_config
        self.pipeline_plan = build_balanced_kimi_k3_pipeline_plan(
            num_layers=config.num_hidden_layers,
            hidden_size=config.hidden_size,
            attn_res_block_size=config.attn_res_block_size,
            stage_count=mapping.pipeline.stage_count,
            activation_dtype=str(torch.get_default_dtype()).removeprefix("torch."),
        )
        self.stage_plan = self.pipeline_plan.stages[mapping.pipeline.stage_index]

        alt_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

        self.embed_tokens = (
            VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                tp_rank=mapping.attn.tp_rank,
                tp_size=mapping.attn.tp_size,
                tp_group=mapping.attn.tp_group,
            )
            if self.stage_plan.owns_embedding
            else None
        )

        def get_layer(idx: int, prefix: str):
            if not self.stage_plan.first_layer <= idx < self.stage_plan.end_layer:
                return nn.Identity()
            return KimiLinearDecoderLayer(
                config=config,
                mapping=mapping,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            )

        layers_prefix = add_prefix("layers", prefix)
        self.layers = nn.ModuleList(
            get_layer(idx, add_prefix(idx, layers_prefix))
            for idx in range(config.num_hidden_layers)
        )
        # Cross-layer attn-side mix precompute: a layer's aux stream computes
        # the NEXT layer's block partial alongside its own mlp-side partial
        # (one dual sweep under attention; blocks are final by then).
        for i in range(self.stage_plan.first_layer, self.stage_plan.end_layer - 1):
            cur, nxt = self.layers[i], self.layers[i + 1]
            assert isinstance(cur, KimiLinearDecoderLayer)
            assert isinstance(nxt, KimiLinearDecoderLayer)
            if nxt.prev_valid_blocks > 0:
                assert (
                    cur.mlp_res_norm.variance_epsilon
                    == nxt.self_attention_res_norm.variance_epsilon
                ), "dual partial assumes one shared RMS epsilon"
                cur._next_attn_mix = (nxt, nxt.prev_valid_blocks)
                nxt._attn_split = True

        self.norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if self.stage_plan.owns_head
            else None
        )

        # Model-level AttnRes output mixing.
        self.output_attn_res_norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if self.stage_plan.owns_head
            else None
        )
        self.output_attn_res_proj = (
            ReplicatedLinear(
                config.hidden_size,
                1,
                bias=False,
                prefix=add_prefix("output_attn_res_proj", prefix),
            )
            if self.stage_plan.owns_head
            else None
        )
        # One-based completed-layer ids; see set_eagle3_layers_to_capture.
        self.eagle3_layers_to_capture: tuple[int, ...] = ()

        # DFLASH/DSpark speculative decoding: layer indices whose *output*
        # stream is captured for the draft. Populated by
        # ``set_dflash_layers_to_capture``; empty means no capture.
        self.layers_to_capture: list[int] = []
        self._dflash_incremental_callback = None
        self._dflash_slot_bufs = None
        self._dflash_capture_idx_map: dict[int, int] = {}

    def get_input_embeddings(self) -> nn.Module:
        if self.embed_tokens is None:
            raise AttributeError(
                "only the first Kimi-K3 pipeline stage owns embeddings"
            )
        return self.embed_tokens

    def _dspark_capture_stream(
        self,
        layer_idx: int,
        prefix_sum: torch.Tensor,
        block_residual: torch.Tensor,
    ) -> torch.Tensor:
        """Capture vLLM's completed-layer DSpark target stream.

        vLLM captures ``prefix_sum + hidden_states`` immediately after the
        named layer. This implementation accumulates the layer output into
        ``prefix_sum`` before returning from the layer, so that sum is already
        represented by ``prefix_sum``. Clone because later layers may update
        the same storage in place.
        """
        del layer_idx, block_residual
        return prefix_sum.clone()

    @torch.no_grad()
    def forward_pipeline_stage(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: "ForwardContext",
        out_cache_loc: torch.Tensor,
        incoming: StageActivation | None,
        input_embeds: torch.Tensor | None = None,
    ) -> StageOutput:
        """Execute this rank's contiguous K3 layer range.

        The boundary payload is chain-full: the prefix stream and every
        completed AttnRes snapshot are transferred as independent tensors.
        """

        plan = self.stage_plan
        num_blocks = ceil_div(
            self.config.num_hidden_layers, self.config.attn_res_block_size
        )
        if plan.owns_embedding:
            if incoming is not None:
                raise PipelineProtocolError(
                    "the first Kimi-K3 stage cannot consume an activation"
                )
            if input_embeds is not None:
                prefix_sum = input_embeds
            else:
                assert self.embed_tokens is not None
                prefix_sum = self.embed_tokens(input_ids)
            block_residual = prefix_sum.new_empty(
                num_blocks, prefix_sum.size(0), prefix_sum.size(1)
            )
        else:
            if input_embeds is not None:
                raise PipelineProtocolError(
                    "only the first Kimi-K3 stage accepts input embeddings"
                )
            if incoming is None or plan.input_schema is None:
                raise PipelineProtocolError(
                    f"Kimi-K3 stage {plan.stage_id} is missing its activation"
                )
            plan.input_schema.validate(incoming)
            prefix_sum = incoming.values[0]
            block_residual = prefix_sum.new_empty(
                num_blocks, prefix_sum.size(0), prefix_sum.size(1)
            )
            completed_blocks = plan.first_layer // self.config.attn_res_block_size
            for block_id, snapshot in enumerate(
                incoming.values[1 : completed_blocks + 1]
            ):
                block_residual[block_id].copy_(snapshot)

        if plan.stage_count > 1:
            expected_tokens = int(ctx.input_num_tokens)
            observed_tokens = {
                "activation": int(prefix_sum.shape[0]),
                "input_ids": int(input_ids.shape[0]),
                "positions": int(positions.shape[-1]),
                "out_cache_loc": int(out_cache_loc.shape[0]),
            }
            mismatched = {
                name: count
                for name, count in observed_tokens.items()
                if count != expected_tokens
            }
            if mismatched:
                raise PipelineProtocolError(
                    f"Kimi-K3 stage {plan.stage_id} expected {expected_tokens} "
                    f"tokens, got {mismatched}"
                )

        capture_layers = self.layers_to_capture
        capture_dflash = bool(capture_layers)
        capture_eagle3 = bool(self.eagle3_layers_to_capture)
        aux_hidden_states: list[torch.Tensor] | None = (
            [] if capture_dflash or capture_eagle3 else None
        )

        for layer_idx in range(plan.first_layer, plan.end_layer):
            layer = self.layers[layer_idx]
            prefix_sum, block_residual = layer(
                positions, prefix_sum, ctx, out_cache_loc, block_residual
            )
            if capture_dflash and layer_idx in capture_layers:
                captured = self._dspark_capture_stream(
                    layer_idx, prefix_sum, block_residual
                )
                capture_idx = self._dflash_capture_idx_map.get(layer_idx)
                if self._dflash_slot_bufs is not None and capture_idx is not None:
                    num_tokens = captured.shape[0]
                    self._dflash_slot_bufs[capture_idx][:num_tokens].copy_(captured)
                    if self._dflash_incremental_callback is not None:
                        self._dflash_incremental_callback(capture_idx, num_tokens)
                assert aux_hidden_states is not None
                aux_hidden_states.append(captured)
            # Clone: the copy must survive the next layer's in-place residual writes.
            elif capture_eagle3 and layer_idx + 1 in self.eagle3_layers_to_capture:
                assert aux_hidden_states is not None
                aux_hidden_states.append(prefix_sum.clone())

        if not plan.owns_head:
            assert plan.output_schema is not None
            completed_blocks = plan.end_layer // self.config.attn_res_block_size
            activation = plan.output_schema.bind(
                (
                    prefix_sum,
                    *(block_residual[index] for index in range(completed_blocks)),
                )
            )
            return StageOutput(activation=activation)

        assert self.output_attn_res_proj is not None
        assert self.output_attn_res_norm is not None
        assert self.norm is not None
        hidden_states = _apply_attn_res(
            prefix_sum,
            block_residual,
            self.output_attn_res_proj,
            self.output_attn_res_norm,
            num_blocks,
            out_norm=self.norm,
        )
        return StageOutput(final_output=(hidden_states, aux_hidden_states))

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: "ForwardContext",
        out_cache_loc: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, list | None]:
        if self.stage_plan.stage_count != 1:
            raise RuntimeError(
                "multi-stage Kimi-K3 must run through the pipeline executor"
            )
        output = self.forward_pipeline_stage(
            input_ids,
            positions,
            ctx,
            out_cache_loc,
            incoming=None,
            input_embeds=input_embeds,
        )
        assert output.final_output is not None
        return output.final_output


class KimiLinearForCausalLM(BaseCausalLM):
    """Kimi-K3 text backbone: ``KimiLinearModel`` + lm head + logits processor.

    Inherits ``BaseCausalLM`` so the ``model.*`` / ``lm_head.*`` weight hierarchy
    matches the checkpoint (``language_model.model.*`` / ``language_model.lm_head.*``
    after the wrapper strips the ``language_model.`` prefix).
    """

    model_cls = KimiLinearModel
    _supports_kimi_k3_strict_weight_audit = True

    def resolve_lm_head(
        self,
        config: KimiLinearConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> nn.Module | None:
        if not self.mapping.pipeline.is_last_stage:
            return None
        if self.mapping.pipeline.stage_count > 1 and getattr(
            config, "tie_word_embeddings", False
        ):
            raise ValueError(
                "multi-stage Kimi-K3 does not support tied input/output embeddings"
            )
        return super().resolve_lm_head(config, quant_config, prefix)

    def resolve_logits_processor(self, config: KimiLinearConfig):
        if not self.mapping.pipeline.is_last_stage:
            return None
        return super().resolve_logits_processor(config)

    @torch.no_grad()
    def forward_pipeline_stage(
        self,
        ctx: "ForwardContext",
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        out_cache_loc: torch.Tensor,
        incoming: StageActivation | None,
        input_embeds: torch.Tensor | None = None,
    ) -> StageOutput:
        output = self.model.forward_pipeline_stage(
            input_ids,
            positions,
            ctx,
            out_cache_loc,
            incoming=incoming,
            input_embeds=input_embeds,
        )
        if output.activation is not None:
            return output

        assert output.final_output is not None
        hidden_states, aux_hidden_states = output.final_output
        assert self.logits_processor is not None
        assert self.lm_head is not None
        return StageOutput(
            final_output=self.logits_processor(
                input_ids,
                hidden_states,
                self.lm_head,
                LogitsMetadata.from_forward_context(ctx),
                aux_hidden_states,
            )
        )

    def set_eagle3_layers_to_capture(self, layer_ids: list[int] | None = None) -> None:
        """Take the draft config's one-based completed-layer ids unchanged."""
        if self.mapping.pipeline.stage_count > 1:
            raise ValueError(
                "K3 EAGLE3 capture is not supported with pipeline parallelism"
            )
        num_layers = len(self.model.layers)
        selected = (
            [2, num_layers // 2, num_layers - 3]
            if layer_ids is None
            else list(layer_ids)
        )
        if selected != sorted(selected) or len(set(selected)) != len(selected):
            raise ValueError(
                "K3 EAGLE3 layer ids must be unique and sorted ascending, "
                f"got {selected}."
            )
        invalid = [layer_id for layer_id in selected if not 1 <= layer_id <= num_layers]
        if invalid:
            raise ValueError(
                "K3 EAGLE3 layer ids must identify a completed K3 layer; "
                f"got invalid ids {invalid} for {num_layers} layers."
            )
        self.model.eagle3_layers_to_capture = tuple(selected)

    def get_embed_and_head(self):
        if self.mapping.pipeline.stage_count > 1:
            raise AttributeError(
                "Kimi-K3 pipeline stages do not colocate embeddings and LM head"
            )
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_dflash_layers_to_capture(
        self,
        layer_ids: list[int],
        incremental_callback=None,
        slot_bufs: list | None = None,
    ) -> None:
        """Capture the K3 residual stream after each named target layer.

        DFLASH/DSpark checkpoints name 0-indexed completed-layer outputs. The
        per-layer return has already accumulated the layer output into K3's
        prefix stream, matching vLLM's target-side capture contract.
        """
        if self.mapping.pipeline.stage_count > 1:
            raise ValueError(
                "DFLASH target capture is not supported with pipeline parallelism"
            )
        num_layers = len(self.model.layers)
        if len(set(layer_ids)) != len(layer_ids):
            raise ValueError("DFLASH target_layer_ids must be unique.")
        invalid = [val for val in layer_ids if val < 0 or val >= num_layers]
        if invalid:
            raise ValueError(
                "DFLASH target_layer_ids must map to capturable target layer "
                f"outputs. Got invalid ids {invalid}; valid range is "
                f"[0, {num_layers - 1}] for {num_layers} target layers."
            )
        self.capture_aux_hidden_states = True
        # Ascending: the draft concatenates the taps positionally, so the
        # capture order is part of the weight contract.
        self.model.layers_to_capture = sorted(layer_ids)
        self.model._dflash_capture_idx_map = {
            layer_idx: i for i, layer_idx in enumerate(self.model.layers_to_capture)
        }
        self.model._dflash_incremental_callback = incremental_callback
        self.model._dflash_slot_bufs = slot_bufs

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        *,
        _weight_audit: _KimiK3WeightLoadAudit | None = None,
    ) -> None:
        """Load the ``model.*`` / ``lm_head.*`` text weights.

        Reuses the DeepSeek machinery: ``gate_proj``/``up_proj`` stack into
        ``gate_up_proj``; ``q_a_proj``/``kv_a_proj_with_mqa`` fuse into
        ``fused_qkv_a_proj_with_mqa``; routed experts go through the MoE
        checkpoint loader (``w1``/``w3``/``w2`` -> ``w13``/``w2``, MXFP4).

        KDA layers' remaining ``self_attn.*`` weights (conv + A_log / dt_bias /
        f_b / o_norm / o_proj) load directly through the default path — their
        names match ``KimiLinearKDA``'s modules 1:1 (no fusion).
        """
        config = self.config
        stacked_params_mapping = [
            # KDA q/k/v/g/f_a/b stack into qkvgb_proj; MLA's g_proj falls
            # through (no such param on MLA layers).
            ("self_attn.qkvgb_proj", "self_attn.q_proj", "q"),
            ("self_attn.qkvgb_proj", "self_attn.k_proj", "k"),
            ("self_attn.qkvgb_proj", "self_attn.v_proj", "v"),
            ("self_attn.qkvgb_proj", "self_attn.g_proj", "g"),
            ("self_attn.qkvgb_proj", "self_attn.f_a_proj", "f_a"),
            ("self_attn.qkvgb_proj", "self_attn.b_proj", "b"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        fuse_qkv_a_proj = config.q_lora_rank is not None

        params_dict = dict(self.named_parameters())
        audit = _weight_audit or _KimiK3WeightLoadAudit(
            stage_plan=self.model.stage_plan,
            mapping=self.mapping,
            config=config,
        )
        audit.require_targets("text", params_dict)
        moe_loader = build_moe_checkpoint_loader(
            params_dict=params_dict,
            expert_schema=ExpertCheckpointSchema(
                gate_proj_name="w1", up_proj_name="w3", down_proj_name="w2"
            ),
            num_experts=config.num_experts,
            ep_rank=self.mapping.moe.ep_rank,
            ep_size=self.mapping.moe.ep_size,
        )

        for checkpoint_name, loaded_weight in weights:
            disposition = _classify_kimi_k3_checkpoint_weight(
                checkpoint_name,
                stage_plan=self.model.stage_plan,
                include_vision=False,
                num_hidden_layers=config.num_hidden_layers,
                num_nextn_predict_layers=getattr(config, "num_nextn_predict_layers", 0),
            )
            if disposition == "stage_unowned":
                audit.record_stage_unowned(checkpoint_name)
                continue
            if disposition == "auxiliary":
                audit.record_auxiliary(checkpoint_name)
                continue
            if disposition == "unexpected":
                audit.record_unexpected(checkpoint_name)
                continue

            checkpoint_name = _strip_kimi_k3_language_prefix(checkpoint_name)
            name = checkpoint_name
            # Compressed-tensors MXFP4 routed experts ship the packed weight as
            # ``...w{1,2,3}.weight_packed``; the mxfp4 MoE param is
            # ``w13_weight`` / ``w2_weight`` (packed uint8), so drop the
            # ``_packed`` suffix for the expert loader (scale keeps ``weight_scale``).
            if "experts." in name and name.endswith(".weight_packed"):
                name = name[: -len(".weight_packed")] + ".weight"
            # KDA conv weights are plain params named ``<qkv>_conv1d_weight``.
            if "_conv1d.weight" in name:
                name = name.replace("_conv1d.weight", "_conv1d_weight")

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if ".experts." in name and name not in params_dict:
                    continue  # routed-expert weights handled by moe_loader below
                    # NB: ``.experts.`` (leading dot) so ``shared_experts`` is
                    # NOT matched -- its gate_proj/up_proj must stack here.
                mapped = name.replace(weight_name, param_name)
                if mapped not in params_dict:
                    continue
                param = params_dict[mapped]
                slot = audit.prepare_slot(
                    checkpoint_name,
                    loaded_weight,
                    namespace="text",
                    target_name=mapped,
                    shard_id=shard_id,
                )
                if slot is None:
                    break
                param.weight_loader(param, loaded_weight, shard_id)
                audit.record_consumed(
                    checkpoint_name,
                    namespace="text",
                    target_name=mapped,
                    slot=slot,
                )
                break
            else:
                if moe_loader.matches(name):
                    slot = audit.prepare_moe_slot(checkpoint_name, loaded_weight)
                    if slot is None:
                        continue
                    mapped = moe_loader.load(name, loaded_weight)
                    if not audit.confirm_loader_target(
                        checkpoint_name,
                        slot,
                        mapped,
                    ):
                        continue
                    audit.record_consumed(
                        checkpoint_name,
                        namespace="text",
                        target_name=mapped,
                        slot=slot,
                    )
                    continue
                if moe_loader.is_expert_checkpoint_weight(name):
                    audit.record_rank_unowned(checkpoint_name)
                    continue

                if fuse_qkv_a_proj and ".g_proj" in name:
                    # MLA output gate (KDA g_proj stacked into qkvgb above):
                    # the per-rank shard sits after [q_a | kv_a+rope] in the
                    # widened fused a-projection.
                    mapped = name.replace("g_proj", "fused_qkv_a_proj_with_mqa")
                    param = params_dict.get(mapped)
                    if param is not None:
                        slot = audit.prepare_slot(
                            checkpoint_name,
                            loaded_weight,
                            namespace="text",
                            target_name=mapped,
                            component="g",
                        )
                        if slot is None:
                            continue
                        gate_offset = (
                            config.q_lora_rank
                            + config.kv_lora_rank
                            + config.qk_rope_head_dim
                        )
                        # The checkpoint gate is globally head-sharded; load
                        # this attention rank's rows into the fused tail.
                        gate_rows = loaded_weight.shape[0] // self.mapping.attn.tp_size
                        gate_start = self.mapping.attn.tp_rank * gate_rows
                        gate_shard = loaded_weight[gate_start : gate_start + gate_rows]
                        param.weight_loader(param, gate_shard, begin_size=gate_offset)
                        audit.record_consumed(
                            checkpoint_name,
                            namespace="text",
                            target_name=mapped,
                            slot=slot,
                        )
                        continue
                    audit.record_unexpected(checkpoint_name)
                    continue

                if fuse_qkv_a_proj and (
                    "q_a_proj" in name or "kv_a_proj_with_mqa" in name
                ):
                    # Single targeted replace: chaining ``.replace`` corrupts the
                    # q_a case because ``fused_qkv_a_proj_with_mqa`` (the q_a
                    # result) itself contains ``kv_a_proj_with_mqa`` as a
                    # substring, so a second replace would mangle it.
                    if "q_a_proj" in name:
                        begin_size = 0
                        mapped = name.replace("q_a_proj", "fused_qkv_a_proj_with_mqa")
                    else:
                        begin_size = config.q_lora_rank
                        mapped = name.replace(
                            "kv_a_proj_with_mqa", "fused_qkv_a_proj_with_mqa"
                        )
                    param = params_dict.get(mapped)
                    if param is None:
                        audit.record_unexpected(checkpoint_name)
                        continue
                    component = "q_a" if "q_a_proj" in name else "kv_a"
                    slot = audit.prepare_slot(
                        checkpoint_name,
                        loaded_weight,
                        namespace="text",
                        target_name=mapped,
                        component=component,
                    )
                    if slot is None:
                        continue
                    param.weight_loader(param, loaded_weight, begin_size=begin_size)
                    audit.record_consumed(
                        checkpoint_name,
                        namespace="text",
                        target_name=mapped,
                        slot=slot,
                    )
                    continue

                param = params_dict.get(name)
                if param is None:
                    audit.record_unexpected(checkpoint_name)
                    continue
                slot = audit.prepare_slot(
                    checkpoint_name,
                    loaded_weight,
                    namespace="text",
                    target_name=name,
                )
                if slot is None:
                    continue
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                audit.record_consumed(
                    checkpoint_name,
                    namespace="text",
                    target_name=name,
                    slot=slot,
                )

        audit.finish()
        self.post_load_weights()

    def post_load_weights(self) -> None:
        """Prepare the absorbed MLA weights (``w_kc``/``w_vc``) per MLA layer.

        Kimi-K3's attention is unquantized (MXFP4 ignores ``self_attn.*``), so
        ``kv_b_proj.weight`` is bf16 and no block dequant is needed. KDA layers
        have no ``kv_b_proj`` (not ``KimiLinearMLAAttention``) and are skipped.
        """
        local_layers = self.model.layers[
            self.model.stage_plan.first_layer : self.model.stage_plan.end_layer
        ]
        for layer in local_layers:
            assert isinstance(layer, KimiLinearDecoderLayer)
            self_attn = layer.self_attn
            if isinstance(self_attn, KimiLinearMLAAttention):
                self_attn.w_kc, self_attn.w_vc = _prepare_mla_kv_b_proj_weights(
                    self_attn.kv_b_proj.weight, self_attn
                )
            elif isinstance(self_attn, KimiLinearKDA):
                self_attn.fuse_conv_weights()

        # Fold the AttnRes rms_w * res_w products once; the split kernels take a single wp pointer.
        for layer in local_layers:
            layer._attn_wp = (
                layer.self_attention_res_norm.weight.float()
                * layer.self_attention_res_proj.weight.reshape(-1).float()
            ).to(torch.bfloat16)
            layer._mlp_wp = (
                layer.mlp_res_norm.weight.float()
                * layer.mlp_res_proj.weight.reshape(-1).float()
            ).to(torch.bfloat16)

        for layer in local_layers:
            if getattr(layer, "is_moe_layer", False):
                layer.block_sparse_moe.pack_input_projection_weights()


# ===----------------------------------------------------------------------=== #
# Registered multimodal wrapper
# ===----------------------------------------------------------------------=== #


class KimiK3ForConditionalGeneration(nn.Module):
    """Kimi-K3 top-level model (registered architecture).

    Construction mirrors ``KimiK25ForConditionalGeneration``: a vision path plus
    a text ``language_model``. The text path is ``KimiLinearForCausalLM``.
    """

    def __init__(
        self,
        config: KimiK3Config,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        is_multimodal_active: bool = True,
        mm_attention_backend: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.quant_config = quant_config
        self.is_multimodal_active = is_multimodal_active and (
            mapping.pipeline.stage_count == 1 or mapping.pipeline.is_first_stage
        )

        # EPD encode workers own only the vision tower.
        self.language_model = None
        if not getattr(config, "encoder_only", False):
            self.language_model = KimiLinearForCausalLM(
                config.text_config,
                mapping=mapping,
                quant_config=quant_config,
            )

        # Multimodal path. ``image_encoder`` may later be replaced by
        # ModelExecutor with the encoder CUDA-graph wrapper.
        if self.is_multimodal_active:
            self.vision = KimiK3Vision(
                config.vision_config,
                mapping=mapping,
                quant_config=quant_config,
                mm_attention_backend=mm_attention_backend,
            )
            # Normal serving follows the text embedding dtype. Encoder-only
            # construction uses ModelLoader's configured default dtype.
            if self.language_model is not None:
                target_dtype = self.get_input_embeddings().weight.dtype
                self.vision = self.vision.to(dtype=target_dtype)
            self.vision_embedder = VisionEmbedder(encoder_mapping=mapping.vision)
            self.image_encoder = self.vision.embed_media
        else:
            self.vision = None
            self.vision_embedder = None
            self.image_encoder = None

    def get_input_embeddings(self) -> nn.Module:
        if self.language_model is None:
            raise AttributeError(
                "Kimi-K3 encoder-only mode does not expose text embeddings."
            )
        return self.language_model.model.get_input_embeddings()

    @property
    def pipeline_stage_plan(self):
        if self.language_model is None:
            raise AttributeError("encoder-only Kimi-K3 has no language pipeline stage")
        return self.language_model.model.stage_plan

    @property
    def pipeline_plan(self):
        if self.language_model is None:
            raise AttributeError("encoder-only Kimi-K3 has no language pipeline plan")
        return self.language_model.model.pipeline_plan

    def get_embed_and_head(self):
        if self.mapping.pipeline.stage_count > 1:
            raise AttributeError(
                "Kimi-K3 pipeline stages do not colocate embeddings and LM head"
            )
        return self.language_model.get_embed_and_head()

    def set_dflash_layers_to_capture(
        self,
        layer_ids: list[int],
        incremental_callback=None,
        slot_bufs: list | None = None,
    ) -> None:
        if self.language_model is None:
            raise AttributeError(
                "Kimi-K3 encoder-only mode cannot capture target hidden states."
            )
        self.language_model.set_dflash_layers_to_capture(
            layer_ids,
            incremental_callback=incremental_callback,
            slot_bufs=slot_bufs,
        )

    def set_eagle3_layers_to_capture(self, layer_ids: list[int] | None = None) -> None:
        if self.language_model is None:
            raise AttributeError(
                "Kimi-K3 encoder-only mode does not support EAGLE3 speculative decoding."
            )
        self.language_model.set_eagle3_layers_to_capture(layer_ids)

    @property
    def logits_processor(self):
        # The runtime reads ``model.logits_processor`` on the top-level model
        # (model_executor.py) to build its sampling topology; delegate to the
        # text backbone, which owns it (BaseCausalLM).
        if self.language_model is None:
            raise AttributeError(
                "Kimi-K3 encoder-only mode does not expose a logits processor."
            )
        logits_processor = self.language_model.logits_processor
        if logits_processor is None:
            raise AttributeError(
                "only the final Kimi-K3 pipeline stage owns logits processing"
            )
        return logits_processor

    @property
    def lm_head(self):
        if self.language_model is None:
            raise AttributeError(
                "Kimi-K3 encoder-only mode does not expose an LM head."
            )
        lm_head = self.language_model.lm_head
        if lm_head is None:
            raise AttributeError(
                "only the final Kimi-K3 pipeline stage owns the LM head"
            )
        return lm_head

    @property
    def vision_tower(self):
        """Expose the shared MoonViT attribute expected by EPD prefill."""
        return self.vision.vision_tower if self.vision is not None else None

    def get_multimodal_encoder_specs(self) -> dict[Modality, EncoderSpec]:
        if self.vision is None or self.image_encoder is None:
            return {}
        return {
            Modality.IMAGE: EncoderSpec(
                self.image_encoder,
                make_warmup_items=self.vision.make_image_warmup_items,
            )
        }

    def make_encoder_cudagraph_wrapper(
        self, mapping: Mapping
    ) -> EncoderCudaGraphWrapper:
        return self.vision.make_encoder_cudagraph_wrapper(mapping)

    def make_encoder_cudagraph_wrappers(self, mapping: Mapping) -> dict:
        if self.vision is None:
            return {}
        return {"image_encoder": self.make_encoder_cudagraph_wrapper(mapping)}

    def pad_input_ids(
        self, input_ids: list[int], mm_inputs: MultimodalInputs
    ) -> list[int]:
        return pad_input_tokens(input_ids, mm_inputs)

    @torch.no_grad()
    def multimodal_input_embeds(
        self,
        input_ids: torch.Tensor,
        ctx: "ForwardContext",
        multimodal_context,
    ) -> torch.Tensor | None:
        if (
            multimodal_context is None
            or self.vision_embedder is None
            or not multimodal_context.has_extend_inputs()
            or ctx.forward_mode.is_decode_or_idle()
        ):
            return None
        input_embeds, model_kwargs = self.vision_embedder.apply(
            input_ids=input_ids,
            text_embedding=self.get_input_embeddings(),
            ctx=multimodal_context,
            encoders=self.get_multimodal_encoder_specs(),
            multimodal_model=self,
        )
        assert not model_kwargs, "Kimi-K3 multimodal path must stay embeds-only"
        return input_embeds

    @torch.no_grad()
    def forward_pipeline_stage(
        self,
        ctx: "ForwardContext",
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        out_cache_loc: torch.Tensor,
        incoming: StageActivation | None,
        **kwargs,
    ) -> StageOutput:
        if self.language_model is None:
            raise RuntimeError(
                "Kimi-K3 encoder-only mode cannot execute language-model forward."
            )
        input_embeds = kwargs.pop("input_embeds", None)
        if self.mapping.pipeline.is_first_stage:
            multimodal_context = kwargs.pop("multimodal_context", None)
            if input_embeds is None:
                input_embeds = self.multimodal_input_embeds(
                    input_ids, ctx, multimodal_context
                )
        return self.language_model.forward_pipeline_stage(
            ctx,
            input_ids,
            positions,
            out_cache_loc,
            incoming=incoming,
            input_embeds=input_embeds,
        )

    @torch.no_grad()
    def forward(
        self,
        ctx: "ForwardContext",
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        out_cache_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        if self.language_model is None:
            raise RuntimeError(
                "Kimi-K3 encoder-only mode cannot execute language-model forward."
            )
        if self.mapping.pipeline.stage_count > 1:
            raise RuntimeError(
                "multi-stage Kimi-K3 must run through the pipeline executor"
            )
        multimodal_context = kwargs.pop("multimodal_context", None)
        input_embeds = self.multimodal_input_embeds(input_ids, ctx, multimodal_context)
        if input_embeds is not None:
            kwargs["input_embeds"] = input_embeds
        return self.language_model.forward(
            ctx,
            input_ids,
            positions,
            out_cache_loc,
            **kwargs,
        )

    def post_load_weights(self) -> None:
        """Prepare text-model derived weights for loaders that skip checkpoints."""
        if self.language_model is not None:
            self.language_model.post_load_weights()

    def checkpoint_weight_name_filter(self, name: str) -> bool:
        """Select only shards containing parameters owned by this PP stage."""

        if self.language_model is None:
            return name.startswith(("vision_tower.", "mm_projector.")) and (
                self.vision is not None
            )
        return kimi_k3_stage_checkpoint_weight_filter(
            name,
            stage_plan=self.pipeline_stage_plan,
            include_vision=self.vision is not None,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Route checkpoint weights by top-level prefix.

        The checkpoint stores ``language_model.*`` for the text model (whose
        params are named ``model.*`` / ``lm_head.*``) and
        ``vision_tower.*`` / ``mm_projector.*`` for the multimodal path.

        Routing stays streaming: K3's text checkpoint is too large to retain
        every yielded tensor in a temporary list. Vision tensors are loaded as
        the language loader advances the source iterator.
        """
        loaded_vision_weights = 0
        dropped_vision_weights = 0
        vision_params = (
            dict(self.vision.named_parameters(remove_duplicate=False))
            if self.vision is not None
            else None
        )
        strict_audit = bool(
            self.language_model is not None
            and getattr(
                self.language_model,
                "_supports_kimi_k3_strict_weight_audit",
                False,
            )
        )
        audit = (
            _KimiK3WeightLoadAudit(
                stage_plan=self.pipeline_stage_plan,
                mapping=self.mapping,
                config=self.config.text_config,
            )
            if strict_audit
            else None
        )
        if audit is not None and vision_params is not None:
            audit.require_targets("vision", vision_params)

        def language_weights():
            nonlocal loaded_vision_weights, dropped_vision_weights
            for name, weight in weights:
                if audit is not None:
                    disposition = _classify_kimi_k3_checkpoint_weight(
                        name,
                        stage_plan=self.pipeline_stage_plan,
                        include_vision=self.vision is not None,
                        num_hidden_layers=self.config.text_config.num_hidden_layers,
                        num_nextn_predict_layers=getattr(
                            self.config.text_config,
                            "num_nextn_predict_layers",
                            0,
                        ),
                    )
                    if disposition == "stage_unowned":
                        if name.startswith(("vision_tower.", "mm_projector.")):
                            dropped_vision_weights += 1
                        audit.record_stage_unowned(name)
                        continue
                    if disposition == "auxiliary":
                        audit.record_auxiliary(name)
                        continue
                    if disposition == "unexpected":
                        audit.record_unexpected(name)
                        continue

                if name.startswith("vision_tower.") or name.startswith("mm_projector."):
                    if self.vision is None:
                        dropped_vision_weights += 1
                    else:
                        assert vision_params is not None
                        mapped = self.vision.checkpoint_parameter_name(name)
                        if audit is not None and mapped not in vision_params:
                            audit.record_unexpected(name)
                            continue
                        slot = (
                            audit.prepare_slot(
                                name,
                                weight,
                                namespace="vision",
                                target_name=mapped,
                            )
                            if audit is not None
                            else None
                        )
                        if audit is not None and slot is None:
                            continue
                        self.vision.load_weight(name, weight, vision_params)
                        if audit is not None:
                            audit.record_consumed(
                                name,
                                namespace="vision",
                                target_name=mapped,
                                slot=slot,
                            )
                        loaded_vision_weights += 1
                    continue
                name = _strip_kimi_k3_language_prefix(name)
                yield name, weight

        if self.language_model is not None:
            if audit is None:
                self.language_model.load_weights(language_weights())
            else:
                self.language_model.load_weights(
                    language_weights(),
                    _weight_audit=audit,
                )
        else:
            # Exhaust the stream so interleaved vision weights are still routed.
            for _ in language_weights():
                pass
        if dropped_vision_weights and self.mapping.pipeline.stage_count == 1:
            logger.warning(
                "Dropping %d vision weights: multimodal path is inactive.",
                dropped_vision_weights,
            )
        logger.debug("Loaded %d Kimi-K3 vision tensors.", loaded_vision_weights)


EntryClass = [KimiK3ForConditionalGeneration]
