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

"""Cache setup results and model-recipe dispatch."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Literal

import torch

from tokenspeed.runtime.layers.attention.configs.base import BaseAttnConfig
from tokenspeed.runtime.layers.attention.kv_cache.recipes.deepseek_v4 import (
    prepare_deepseek_v4_cache,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.inkling import (
    prepare_inkling_cache,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.kimi_k3 import (
    prepare_kimi_k3_cache,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.ordinary import (
    prepare_ordinary_cache,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import CacheMemoryPlan
from tokenspeed.runtime.layers.attention.kv_cache.recipes.qwen35 import (
    prepare_qwen35_cache,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    PagedCacheGroupSpec,
)
from tokenspeed.runtime.pipeline.contracts import canonical_digest

CacheModelFamily = Literal[
    "mha",
    "mla",
    "dsa",
    "msa",
    "qwen_gdn",
    "inkling",
    "kimi_k3",
    "deepseek_v4",
]


@dataclass(frozen=True)
class CachePoolSpec:
    """Everything needed to bind one model's compute views to a cache buffer."""

    family: CacheModelFamily
    memory_plan: CacheMemoryPlan
    layer_types: tuple[str, ...]
    layer_group_ids: tuple[str, ...]
    # Scheduler group specs, computed once by the recipe. The pool aligns
    # their physical fields (packing) with the memory plan and publishes the
    # runtime contract from the pair.
    paged_cache_group_specs: tuple[PagedCacheGroupSpec, ...]
    state_field_dtypes: Mapping[str, torch.dtype]
    token_capacity: int
    # Concrete storage dtype for fields whose representation differs from the
    # pool-wide default. K3 DSpark uses this for BF16 draft latent KV beside
    # FP8 target latent KV in the same arena.
    field_dtypes: Mapping[str, torch.dtype] = field(default_factory=dict)
    layer_kv_head_counts: tuple[int, ...] | None = None
    pool_options: object | None = None
    # ``None`` means the legacy identity mapping. Pipeline stages carry their
    # global model layer ids here while the concrete pool remains densely
    # indexed from zero.
    logical_layer_ids: tuple[int, ...] | None = None
    pipeline_plan_digest: str | None = None
    cache_abi_digest: str | None = None
    cache_manifest_digest: str | None = None
    # Model/runtime values that affect allocation or kernel interpretation but
    # are not already represented by the concrete memory/group plans.
    runtime_parameters: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        total_layers = len(self.layer_group_ids)
        if not total_layers:
            raise ValueError("cache spec must contain at least one layer")
        if self.layer_types and len(self.layer_types) != total_layers:
            raise ValueError("cache layer types must be empty or cover every layer")
        if self.layer_kv_head_counts is not None and (
            len(self.layer_kv_head_counts) != total_layers
        ):
            raise ValueError("cache KV head counts must cover every layer")
        if self.logical_layer_ids is not None:
            if len(self.logical_layer_ids) != total_layers:
                raise ValueError("logical cache layer ids must cover every layer")
            if len(set(self.logical_layer_ids)) != total_layers or any(
                isinstance(layer_id, bool)
                or not isinstance(layer_id, int)
                or layer_id < 0
                for layer_id in self.logical_layer_ids
            ):
                raise ValueError(
                    "logical cache layer ids must be unique non-negative integers"
                )
        planned_fields = {field.field_id: field for field in self.memory_plan.fields}
        for field_id, dtype in self.field_dtypes.items():
            planned = planned_fields.get(field_id)
            if planned is None:
                raise ValueError(
                    f"cache field dtype names an unplanned field {field_id!r}"
                )
            if not isinstance(dtype, torch.dtype):
                raise TypeError(
                    f"cache field dtype for {field_id!r} must be a torch.dtype"
                )
            if torch.empty((), dtype=dtype).element_size() != planned.element_size:
                raise ValueError(
                    f"cache field dtype for {field_id!r} does not match its plan"
                )
        for name, digest in (
            ("pipeline_plan_digest", self.pipeline_plan_digest),
            ("cache_abi_digest", self.cache_abi_digest),
            ("cache_manifest_digest", self.cache_manifest_digest),
        ):
            if digest is not None and (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        object.__setattr__(self, "runtime_parameters", tuple(self.runtime_parameters))
        parameter_names = tuple(name for name, _ in self.runtime_parameters)
        if any(not isinstance(name, str) or not name for name in parameter_names):
            raise ValueError("cache runtime parameter names must be non-empty strings")
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("cache runtime parameter names must be unique")

    @property
    def pool_size(self) -> int:
        max_packing = max(
            group.cache_blocks_per_lcm_block for group in self.memory_plan.groups
        )
        return (
            self.memory_plan.num_lcm_blocks
            * max_packing
            * self.memory_plan.logical_block_tokens
        )

    @property
    def runtime_abi_digest(self) -> str:
        """Digest every concrete value needed to bind and consume this pool."""

        if self.pool_options is not None:
            raise ValueError("cache runtime ABI cannot project opaque pool_options")
        plan = self.memory_plan
        return canonical_digest(
            {
                "family": self.family,
                "memory_plan": {
                    "logical_block_tokens": plan.logical_block_tokens,
                    "lcm_block_bytes": plan.lcm_block_bytes,
                    "num_lcm_blocks": plan.num_lcm_blocks,
                    "groups": tuple(
                        (
                            group.group_id,
                            group.cache_blocks_per_lcm_block,
                            group.page_count,
                        )
                        for group in plan.groups
                    ),
                    "planes": tuple(
                        (
                            plane.plane_id,
                            plane.bytes_per_lcm_block,
                            plane.arena_offset_bytes,
                        )
                        for plane in plan.planes
                    ),
                    "fields": tuple(
                        (
                            field.group_id,
                            field.field_id,
                            field.plane_id,
                            field.shape,
                            field.element_size,
                            field.field_offset_bytes,
                            field.page_stride_bytes,
                        )
                        for field in plan.fields
                    ),
                },
                "layer_types": self.layer_types,
                "layer_group_ids": self.layer_group_ids,
                "paged_cache_group_specs": tuple(
                    (
                        spec.group_id,
                        spec.retention,
                        spec.rows_per_page,
                        spec.entry_stride_tokens,
                        spec.sliding_window_tokens,
                        spec.family,
                        spec.cache_blocks_per_lcm_block,
                        spec.transfer_policy,
                    )
                    for spec in self.paged_cache_group_specs
                ),
                "state_field_dtypes": tuple(
                    (field_id, str(dtype).removeprefix("torch."))
                    for field_id, dtype in sorted(self.state_field_dtypes.items())
                ),
                "field_dtypes": tuple(
                    (field_id, str(dtype).removeprefix("torch."))
                    for field_id, dtype in sorted(self.field_dtypes.items())
                ),
                "token_capacity": self.token_capacity,
                "layer_kv_head_counts": self.layer_kv_head_counts,
                "logical_layer_ids": self.logical_layer_ids,
                "pipeline_plan_digest": self.pipeline_plan_digest,
                "cache_abi_digest": self.cache_abi_digest,
                "cache_manifest_digest": self.cache_manifest_digest,
                "runtime_parameters": tuple(sorted(self.runtime_parameters)),
            }
        )

    @property
    def global_runtime_abi_digest(self) -> str:
        """Digest runtime values that every pipeline stage must share."""

        if self.cache_abi_digest is None:
            raise ValueError("global cache runtime ABI requires a static cache ABI")
        return canonical_digest(
            {
                "cache_abi_digest": self.cache_abi_digest,
                "token_capacity": self.token_capacity,
                "runtime_parameters": tuple(sorted(self.runtime_parameters)),
            }
        )

    def layer_view(
        self,
        *,
        first_layer: int,
        num_layers: int,
        family: CacheModelFamily | None = None,
        publish_runtime_contract: bool = True,
    ) -> CachePoolSpec:
        """Describe one concrete compute view over this spec's shared arena.

        The memory plan and scheduler geometry stay merged. Only per-layer
        compute metadata is sliced; a secondary view inherits the target's
        published contract instead of publishing the same groups again.
        """
        total_layers = len(self.layer_group_ids)
        if first_layer < 0 or num_layers < 0:
            raise ValueError("cache layer view bounds must be non-negative")
        last_layer = first_layer + num_layers
        if last_layer > total_layers:
            raise ValueError(
                f"cache layer view [{first_layer}, {last_layer}) exceeds "
                f"the merged {total_layers}-layer spec"
            )
        if self.layer_types and len(self.layer_types) != total_layers:
            raise ValueError("cache layer types must be empty or cover every layer")
        if self.layer_kv_head_counts is not None and (
            len(self.layer_kv_head_counts) != total_layers
        ):
            raise ValueError("cache KV head counts must cover every layer")
        if self.logical_layer_ids is not None and (
            len(self.logical_layer_ids) != total_layers
        ):
            raise ValueError("logical cache layer ids must cover every layer")
        return replace(
            self,
            family=family or self.family,
            layer_types=(
                self.layer_types[first_layer:last_layer] if self.layer_types else ()
            ),
            layer_group_ids=self.layer_group_ids[first_layer:last_layer],
            layer_kv_head_counts=(
                self.layer_kv_head_counts[first_layer:last_layer]
                if self.layer_kv_head_counts is not None
                else None
            ),
            logical_layer_ids=(
                self.logical_layer_ids[first_layer:last_layer]
                if self.logical_layer_ids is not None
                else None
            ),
            paged_cache_group_specs=(
                self.paged_cache_group_specs if publish_runtime_contract else ()
            ),
        )


@dataclass(frozen=True)
class CacheSetup:
    """One big model, one spec: target and draft layers share everything.

    Draft layers are continuation layers of the one merged model (global
    layer ids ``num_target_layers..``, the DeepSeek-V4 MTP convention
    generalized): one plan, one arena, one contract, one pool.
    ``num_draft_layers`` is the only draft-specific fact, consumed at
    model-runner wiring time (a draft model's local layer ``i`` maps to
    global layer ``num_target_layers + i``); the spec itself is
    draft-oblivious.
    """

    spec: CachePoolSpec
    num_draft_layers: int
    cache_budget_bytes: int
    fixed_workspace_bytes: int
    # Pipeline stages can hold a local target subset plus draft continuation
    # layers whose logical ids remain global. ``None`` preserves the legacy
    # continuation convention (immediately after this spec's target layers).
    draft_logical_layer_offset: int | None = None

    @property
    def num_target_layers(self) -> int:
        return len(self.spec.layer_group_ids) - self.num_draft_layers


_PREPARE_CACHE = {
    "mha": partial(prepare_ordinary_cache, family="mha"),
    "mla": partial(prepare_ordinary_cache, family="mla"),
    "dsa": partial(prepare_ordinary_cache, family="dsa"),
    "msa": partial(prepare_ordinary_cache, family="msa"),
    "qwen_gdn": prepare_qwen35_cache,
    "inkling": prepare_inkling_cache,
    "kimi_k3": prepare_kimi_k3_cache,
    "deepseek_v4": prepare_deepseek_v4_cache,
}


def prepare_cache_setup(
    *,
    family: CacheModelFamily,
    server_args,
    model_config,
    attn_config: BaseAttnConfig,
    draft_model_config,
    draft_attn_config: BaseAttnConfig | None,
    cache_budget_bytes: int,
    decode_input_tokens: int,
    overlap_schedule_depth: int,
) -> CacheSetup:
    """Apply one model recipe and size target/draft arenas from one budget."""
    prepare = _PREPARE_CACHE.get(family)
    if prepare is None:
        raise ValueError(f"unsupported cache model family: {family}")
    return prepare(
        server_args=server_args,
        model_config=model_config,
        attn_config=attn_config,
        draft_model_config=draft_model_config,
        draft_attn_config=draft_attn_config,
        cache_budget_bytes=cache_budget_bytes,
        decode_input_tokens=decode_input_tokens,
        overlap_schedule_depth=overlap_schedule_depth,
    )
