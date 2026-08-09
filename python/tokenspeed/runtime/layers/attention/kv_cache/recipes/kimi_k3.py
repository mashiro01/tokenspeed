"""Kimi K3 cache layout, capacity, and setup recipe."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from tokenspeed.runtime.distributed.consensus import raise_on_rank_error
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes import (
    configured_token_limit,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.ordinary import (
    mla_cache_fields,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    CacheMemoryPlan,
    continue_layer_fields,
    solve_cache_layout,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.stage_layout import (
    CacheStagePlacement,
    LogicalCacheFieldSpec,
    pipeline_cache_abi_digest,
    solve_stage_cache_layout,
)
from tokenspeed.runtime.pipeline.adapters.kimi_k3 import (
    build_balanced_kimi_k3_pipeline_plan,
)

_KIMI_K3_LAYERS = 93
_KIMI_K3_KDA_LAYERS = 69
_KIMI_K3_MLA_LAYERS = 24
_KIMI_K3_LOGICAL_BLOCK_TOKENS = 128
_KIMI_K3_STATE_GROUPS = 3
_KIMI_K3_MLA_PACKING = 12
FULL_ATTENTION = "full_attention"
LINEAR_ATTENTION = "linear_attention"

if TYPE_CHECKING:
    from tokenspeed.runtime.configs.kimi_k3_config import KimiLinearConfig


def _require_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _require_non_negative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return value


def kimi_k3_pipeline_workspace_bytes(
    *,
    pipeline_plan,
    stage_id: int,
    num_layers: int,
    attn_res_block_size: int,
    hidden_size: int,
    max_step_tokens: int,
    activation_element_size: int,
) -> int:
    """Reserve the stage-local AttnRes slab and received Wire tensors."""

    num_layers = _require_positive_int("num_layers", num_layers)
    attn_res_block_size = _require_positive_int(
        "attn_res_block_size", attn_res_block_size
    )
    hidden_size = _require_positive_int("hidden_size", hidden_size)
    max_step_tokens = _require_positive_int("max_step_tokens", max_step_tokens)
    activation_element_size = _require_positive_int(
        "activation_element_size", activation_element_size
    )
    if isinstance(stage_id, bool) or not isinstance(stage_id, int):
        raise ValueError("stage_id must be an integer")
    if not 0 <= stage_id < len(pipeline_plan.stages):
        raise ValueError("stage_id is outside the pipeline plan")

    stage = pipeline_plan.stages[stage_id]
    incoming_fields = (
        0 if stage.input_schema is None else len(stage.input_schema.fields)
    )
    attn_res_blocks = math.ceil(num_layers / attn_res_block_size)
    return (
        (attn_res_blocks + incoming_fields)
        * max_step_tokens
        * hidden_size
        * activation_element_size
    )


def _one_based_layers(value: object, name: str, num_layers: int) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple of 1-based layer numbers")
    layers = tuple(value)
    if any(isinstance(layer, bool) or not isinstance(layer, int) for layer in layers):
        raise ValueError(f"{name} must contain integer layer numbers")
    if len(layers) != len(set(layers)):
        raise ValueError(f"{name} contains duplicate layer numbers")
    if any(layer < 1 or layer > num_layers for layer in layers):
        raise ValueError(f"{name} contains a layer outside 1..{num_layers}")
    return tuple(sorted(layers))


def kimi_k3_layer_group_ids(text_config: KimiLinearConfig) -> tuple[str, ...]:
    """Map every Kimi-K3 layer to one full-attention or KDA cache group."""
    num_layers = _require_positive_int(
        "num_hidden_layers", text_config.num_hidden_layers
    )
    linear = text_config.linear_attn_config
    if not isinstance(linear, Mapping):
        raise TypeError("linear_attn_config must be a mapping")
    kda_layers = _one_based_layers(
        linear.get("kda_layers"), "linear_attn_config.kda_layers", num_layers
    )
    kda_layer_ids = tuple(layer - 1 for layer in kda_layers)
    kda_layer_id_set = set(kda_layer_ids)
    full_layer_ids = tuple(
        layer_id for layer_id in range(num_layers) if layer_id not in kda_layer_id_set
    )
    if not kda_layer_ids or not full_layer_ids:
        raise ValueError(
            "Kimi-K3 cache requires both KDA and full-attention layers, got "
            f"{len(kda_layer_ids)} and {len(full_layer_ids)}"
        )
    if num_layers == _KIMI_K3_LAYERS and (
        len(kda_layer_ids) != _KIMI_K3_KDA_LAYERS
        or len(full_layer_ids) != _KIMI_K3_MLA_LAYERS
    ):
        raise ValueError(
            "93-layer Kimi-K3 requires 69 KDA and 24 MLA layers, got "
            f"{len(kda_layer_ids)} and {len(full_layer_ids)}"
        )
    if len(kda_layer_ids) % _KIMI_K3_STATE_GROUPS:
        raise ValueError(
            "Kimi-K3 cache requires the KDA layer count to divide into "
            f"{_KIMI_K3_STATE_GROUPS} state groups, got {len(kda_layer_ids)}"
        )
    if "full_attn_layers" in linear:
        declared = [
            layer
            for layer in linear["full_attn_layers"]
            if not (isinstance(layer, int) and layer > num_layers)
        ]
        declared_full = _one_based_layers(
            declared,
            "linear_attn_config.full_attn_layers",
            num_layers,
        )
        if declared_full != tuple(layer_id + 1 for layer_id in full_layer_ids):
            raise ValueError(
                "linear_attn_config.full_attn_layers must equal the "
                "kda_layers complement"
            )

    group_ids = [FULL_ATTENTION] * num_layers
    per_group = len(kda_layer_ids) // _KIMI_K3_STATE_GROUPS
    for index, layer_id in enumerate(kda_layer_ids):
        group_ids[layer_id] = f"{LINEAR_ATTENTION}_{index // per_group}"
    return tuple(group_ids)


def kimi_k3_cache_fields(
    *,
    layer_group_ids,
    logical_block_tokens,
    latent_width,
    mla_element_size,
    conv_shape,
    conv_element_size,
    recurrent_shape,
    recurrent_element_size,
) -> tuple[CacheFieldSpec, ...]:
    """Describe Kimi-K3 full-attention MLA KV and KDA state."""
    if logical_block_tokens <= 0 or latent_width <= 0 or mla_element_size <= 0:
        raise ValueError("Kimi-K3 MLA geometry must be positive")
    if (
        not conv_shape
        or not recurrent_shape
        or conv_element_size <= 0
        or recurrent_element_size <= 0
    ):
        raise ValueError("Kimi-K3 KDA state geometry must be positive")

    occurrences: dict[str, int] = {}
    fields = []
    for layer_id, group_id in enumerate(layer_group_ids):
        slot = occurrences.get(group_id, 0)
        occurrences[group_id] = slot + 1
        plane_id = f"slot.{slot}"
        if group_id == "full_attention":
            fields.append(
                CacheFieldSpec(
                    group_id,
                    f"layer.{layer_id}.latent_kv",
                    plane_id,
                    (logical_block_tokens, 1, latent_width),
                    mla_element_size,
                )
            )
            continue
        fields.extend(
            (
                CacheFieldSpec(
                    group_id,
                    f"layer.{layer_id}.conv_state",
                    plane_id,
                    tuple(conv_shape),
                    conv_element_size,
                    exact_page_stride=False,
                ),
                CacheFieldSpec(
                    group_id,
                    f"layer.{layer_id}.recurrent_state",
                    plane_id,
                    tuple(recurrent_shape),
                    recurrent_element_size,
                    exact_page_stride=False,
                ),
            )
        )
    return tuple(fields)


def build_kimi_k3_cache_fields(
    text_config: KimiLinearConfig,
    *,
    tp_size: int,
    mla_cache_dtype: torch.dtype,
    mla_quant_method: str | None,
) -> tuple[CacheFieldSpec, ...]:
    """Build target fields from the Kimi-K3 model configuration."""
    tp_size = _require_positive_int("tp_size", tp_size)
    # fp8_e4m3 is the memory-lean default (matches the Blackwell tokenspeed_mla
    # kernels). bf16 is the Hopper path: FlashMLA has no SM90 dense-fp8 MLA
    # kernel, so on SM90 the MLA layers run bf16 (flashinfer ragged prefill +
    # bf16 FlashMLA decode), mirroring how vLLM/sglang serve K3 on Hopper.
    if mla_cache_dtype not in (torch.float8_e4m3fn, torch.bfloat16):
        raise ValueError(
            "Kimi-K3 cache requires mla_cache_dtype in "
            "{torch.float8_e4m3fn, torch.bfloat16}, got "
            f"{mla_cache_dtype}"
        )
    if mla_quant_method == "per_token_head":
        raise ValueError("Kimi-K3 cache does not support per_token_head MLA cache")
    if getattr(text_config, "mla_use_nope", None) is not True:
        raise ValueError("Kimi-K3 cache requires mla_use_nope=True")

    linear = text_config.linear_attn_config
    num_heads = _require_positive_int(
        "linear_attn_config.num_heads", linear.get("num_heads")
    )
    head_dim = _require_positive_int(
        "linear_attn_config.head_dim", linear.get("head_dim")
    )
    kernel_size = _require_positive_int(
        "linear_attn_config.short_conv_kernel_size",
        linear.get("short_conv_kernel_size"),
    )
    if num_heads % tp_size:
        raise ValueError(
            f"KDA num_heads={num_heads} must be divisible by tp_size={tp_size}"
        )
    kv_lora_rank = _require_positive_int("kv_lora_rank", text_config.kv_lora_rank)
    rope_dim = _require_positive_int("qk_rope_head_dim", text_config.qk_rope_head_dim)
    return kimi_k3_cache_fields(
        layer_group_ids=kimi_k3_layer_group_ids(text_config),
        logical_block_tokens=_KIMI_K3_LOGICAL_BLOCK_TOKENS,
        latent_width=kv_lora_rank + rope_dim,
        mla_element_size=mla_cache_dtype.itemsize,
        conv_shape=(3 * num_heads * head_dim // tp_size, kernel_size - 1),
        conv_element_size=torch.bfloat16.itemsize,
        recurrent_shape=(num_heads // tp_size, head_dim, head_dim),
        recurrent_element_size=torch.float32.itemsize,
    )


def build_kimi_k3_logical_cache_fields(
    text_config: KimiLinearConfig,
    *,
    tp_size: int,
    mla_cache_dtype: torch.dtype,
    mla_quant_method: str | None,
) -> tuple[LogicalCacheFieldSpec, ...]:
    """Attach explicit global layer ownership to K3 cache fields."""

    fields = build_kimi_k3_cache_fields(
        text_config,
        tp_size=tp_size,
        mla_cache_dtype=mla_cache_dtype,
        mla_quant_method=mla_quant_method,
    )
    group_ids = kimi_k3_layer_group_ids(text_config)
    logical_fields = []
    cursor = 0
    for logical_layer_id, group_id in enumerate(group_ids):
        field_count = 1 if group_id == FULL_ATTENTION else 2
        for field in fields[cursor : cursor + field_count]:
            logical_fields.append(LogicalCacheFieldSpec(logical_layer_id, field))
        cursor += field_count
    if cursor != len(fields):
        raise ValueError("Kimi-K3 cache field ownership is incomplete")
    return tuple(logical_fields)


def solve_kimi_k3_cache_layout(
    text_config: KimiLinearConfig,
    *,
    tp_size: int,
    mla_cache_dtype: torch.dtype,
    mla_quant_method: str | None,
    draft_fields=None,
):
    """Solve Kimi-K3's capacity-independent P=128 LCM layout.

    ``draft_fields`` join the same solve as continuation layers of the one
    big model: their field ids are renumbered after the target's 93 layers
    (``layer.93``, ``layer.94``, ...) so they share the full-attention
    group's packing and page-id space with globally unique ids.
    """
    fields = build_kimi_k3_cache_fields(
        text_config,
        tp_size=tp_size,
        mla_cache_dtype=mla_cache_dtype,
        mla_quant_method=mla_quant_method,
    )
    layer_group_ids = kimi_k3_layer_group_ids(text_config)
    num_draft_layers = 0
    if draft_fields is not None:
        continued = continue_layer_fields(
            draft_fields, first_layer_id=len(layer_group_ids)
        )
        num_draft_layers = len(continued)
        fields += continued
    mla_plane_bytes = _KIMI_K3_MLA_PACKING * next(
        field.payload_bytes for field in fields if field.group_id == FULL_ATTENTION
    )
    linear_plane_bytes = sum(
        field.payload_bytes
        for field in fields
        if field.group_id == f"{LINEAR_ATTENTION}_0" and field.plane_id == "slot.0"
    )
    linear_packing = max(1, mla_plane_bytes // linear_plane_bytes)
    packing = {
        FULL_ATTENTION: _KIMI_K3_MLA_PACKING,
        f"{LINEAR_ATTENTION}_0": linear_packing,
        f"{LINEAR_ATTENTION}_1": linear_packing,
        f"{LINEAR_ATTENTION}_2": linear_packing,
    }

    max_padding_fraction = float("inf") if num_draft_layers else 0.25
    layout = solve_cache_layout(
        fields,
        logical_block_tokens=_KIMI_K3_LOGICAL_BLOCK_TOKENS,
        cache_blocks_per_lcm_block=packing,
        alignment=256,
        max_padding_fraction=max_padding_fraction,
    )
    if dict(layout.group_packing) != packing:
        raise ValueError(
            f"Kimi-K3 LCM packing must be {packing}, got {dict(layout.group_packing)}"
        )
    num_mla_layers = sum(gid == FULL_ATTENTION for gid in layer_group_ids)
    if len(layout.plane_bytes) != num_mla_layers + num_draft_layers:
        raise ValueError(
            f"Kimi-K3 LCM requires {num_mla_layers} target planes (one per "
            f"MLA layer) plus {num_draft_layers} draft planes, got "
            f"{len(layout.plane_bytes)}"
        )
    return layout


def kimi_k3_lcm_blocks_needed(
    plan: CacheMemoryPlan,
    *,
    token_capacity: int,
    max_scheduled_tokens: int,
    max_live_requests: int,
    decode_input_tokens: int = 1,
    overlap_schedule_depth: int = 0,
) -> int:
    """Return physical LCM parents needed at the configured concurrency."""
    _require_positive_int("plan.logical_block_tokens", plan.logical_block_tokens)
    _require_positive_int("token_capacity", token_capacity)
    _require_non_negative_int("max_scheduled_tokens", max_scheduled_tokens)
    _require_positive_int("max_live_requests", max_live_requests)
    _require_non_negative_int("decode_input_tokens", decode_input_tokens)
    if overlap_schedule_depth not in (0, 1):
        raise ValueError(
            f"overlap_schedule_depth must be 0 or 1, got {overlap_schedule_depth}"
        )
    if overlap_schedule_depth and decode_input_tokens == 0:
        raise ValueError("overlapped cache sizing requires decode_input_tokens > 0")

    page_tokens = plan.logical_block_tokens
    protected_pages = max_live_requests * math.ceil(
        overlap_schedule_depth * decode_input_tokens / page_tokens
    )
    scheduled_pages = math.ceil(min(max_scheduled_tokens, token_capacity) / page_tokens)
    total_lcm_blocks = 0
    for group in plan.groups:
        if group.group_id == FULL_ATTENTION:
            child_pages = (
                math.ceil(token_capacity / page_tokens)
                + max_live_requests
                - 1
                + protected_pages
            )
        else:
            child_pages = 2 * max_live_requests + scheduled_pages + protected_pages
        total_lcm_blocks += math.ceil(child_pages / group.cache_blocks_per_lcm_block)
    return total_lcm_blocks


def kimi_k3_token_capacity_for_cache_pool(
    plan: CacheMemoryPlan,
    *,
    num_lcm_blocks: int,
    max_scheduled_tokens: int,
    max_live_requests: int,
    decode_input_tokens: int = 1,
    overlap_schedule_depth: int = 0,
    upper_bound_tokens: int | None = None,
) -> int:
    """Invert :func:`kimi_k3_lcm_blocks_needed` by monotonic binary search."""
    _require_positive_int("num_lcm_blocks", num_lcm_blocks)
    if upper_bound_tokens is None:
        full_group = plan.group(FULL_ATTENTION)
        upper_bound_tokens = (
            num_lcm_blocks
            * full_group.cache_blocks_per_lcm_block
            * plan.logical_block_tokens
        )
    _require_positive_int("upper_bound_tokens", upper_bound_tokens)

    low, high = 0, upper_bound_tokens
    while low < high:
        candidate = (low + high + 1) // 2
        required = kimi_k3_lcm_blocks_needed(
            plan,
            token_capacity=candidate,
            max_scheduled_tokens=max_scheduled_tokens,
            max_live_requests=max_live_requests,
            decode_input_tokens=decode_input_tokens,
            overlap_schedule_depth=overlap_schedule_depth,
        )
        if required <= num_lcm_blocks:
            low = candidate
        else:
            high = candidate - 1
    if low == 0:
        raise ValueError(
            f"num_lcm_blocks={num_lcm_blocks} cannot admit one token with "
            "the configured Kimi-K3 cache scheduler limits"
        )
    return low


def _global_min_token_capacity(value: int, mapping) -> int:
    if mapping.world_size == 1:
        return value
    if not dist.is_initialized():
        raise RuntimeError(
            "pipeline cache capacity consensus requires initialized distributed groups"
        )
    tensor = torch.tensor([value], dtype=torch.int64)
    cpu_group = pg_manager.get_process_group("gloo", mapping.world_group)
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=cpu_group)
    result = int(tensor.item())
    if result < 1:
        raise ValueError("global Kimi-K3 cache capacity must be positive")
    return result


def _validate_pipeline_cache_digest_consensus(
    cache_abi_digest: str,
    stage_manifest_digest: str,
    mapping,
) -> None:
    """Validate global and per-stage cache ABIs in one global collective."""

    local = torch.tensor(
        list(bytes.fromhex(cache_abi_digest + stage_manifest_digest)),
        dtype=torch.uint8,
    )
    gathered = [torch.empty_like(local) for _ in range(mapping.world_size)]
    cpu_group = pg_manager.get_process_group("gloo", mapping.world_group)
    dist.all_gather(gathered, local, group=cpu_group)

    global_reference = gathered[0][:32]
    global_divergent = [
        rank
        for rank, candidate in enumerate(gathered)
        if not torch.equal(candidate[:32], global_reference)
    ]
    stage_divergent: list[int] = []
    stage_width = mapping.pipeline.stage_world_size
    for first_rank in range(0, mapping.world_size, stage_width):
        stage_reference = gathered[first_rank][32:]
        stage_divergent.extend(
            rank
            for rank in range(first_rank, first_rank + stage_width)
            if not torch.equal(gathered[rank][32:], stage_reference)
        )
    if global_divergent or stage_divergent:
        raise RuntimeError(
            "Kimi-K3 cache digest consensus failed: "
            f"global ABI ranks={global_divergent}, "
            f"stage ABI ranks={stage_divergent}"
        )


def _prepare_kimi_k3_pipeline_cache(
    *,
    server_args,
    model_config,
    attn_config,
    draft_model_config,
    draft_attn_config,
    cache_budget_bytes: int,
    decode_input_tokens: int,
    overlap_schedule_depth: int,
    mapping,
):
    """Prepare a stage-local K3 cache with collective failure semantics."""

    from tokenspeed.runtime.layers.attention.kv_cache.recipes.setup import (
        CachePoolSpec,
        CacheSetup,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
        build_paged_cache_group_specs,
    )

    prepared = None
    local_error = None
    try:
        if draft_model_config is not None or draft_attn_config is not None:
            raise ValueError(
                "Kimi-K3 pipeline cache does not support speculative draft layers"
            )
        text_config = getattr(
            model_config.hf_config, "text_config", model_config.hf_config
        )
        global_group_ids = kimi_k3_layer_group_ids(text_config)
        global_layer_types = tuple(
            FULL_ATTENTION if group_id == FULL_ATTENTION else LINEAR_ATTENTION
            for group_id in global_group_ids
        )
        global_layout = solve_kimi_k3_cache_layout(
            text_config,
            tp_size=attn_config.attn_tp_size,
            mla_cache_dtype=attn_config.kv_cache_dtype,
            mla_quant_method=attn_config.kv_cache_quant_method or None,
        )
        logical_fields = build_kimi_k3_logical_cache_fields(
            text_config,
            tp_size=attn_config.attn_tp_size,
            mla_cache_dtype=attn_config.kv_cache_dtype,
            mla_quant_method=attn_config.kv_cache_quant_method or None,
        )
        pipeline_plan = build_balanced_kimi_k3_pipeline_plan(
            num_layers=text_config.num_hidden_layers,
            hidden_size=text_config.hidden_size,
            attn_res_block_size=text_config.attn_res_block_size,
            stage_count=mapping.pipeline.stage_count,
            activation_dtype=str(attn_config.dtype).removeprefix("torch."),
        )
        placement = CacheStagePlacement.from_pipeline_plan(
            rank=mapping.rank,
            world_size=mapping.world_size,
            plan=pipeline_plan,
        )
        max_step_tokens = max(
            server_args.chunked_prefill_size,
            attn_config.max_bs * decode_input_tokens,
        )
        fixed_workspace_bytes = kimi_k3_pipeline_workspace_bytes(
            pipeline_plan=pipeline_plan,
            stage_id=placement.stage_id,
            num_layers=text_config.num_hidden_layers,
            attn_res_block_size=text_config.attn_res_block_size,
            hidden_size=text_config.hidden_size,
            max_step_tokens=max_step_tokens,
            activation_element_size=attn_config.dtype.itemsize,
        )
        stage_layout = solve_stage_cache_layout(
            logical_fields,
            placement,
            logical_block_tokens=_KIMI_K3_LOGICAL_BLOCK_TOKENS,
            cache_blocks_per_lcm_block=dict(global_layout.group_packing),
            alignment=256,
            # The global K3 solve retains the 25% guard. After stage
            # projection, local group ratios intentionally differ, so the
            # generic per-group padding ratio no longer models wasted bytes.
            max_padding_fraction=float("inf"),
            compact_group_planes=True,
        )
        logical_layer_ids = tuple(
            binding.logical_layer_id for binding in stage_layout.bindings
        )
        layer_types = tuple(
            global_layer_types[layer_id] for layer_id in logical_layer_ids
        )
        group_ids = tuple(global_group_ids[layer_id] for layer_id in logical_layer_ids)
        _, _, conv_dtype, recurrent_dtype, _ = text_config.mamba2_cache_params
        state_dtypes = {
            f"layer.{logical_layer_id}.conv_state": conv_dtype
            for logical_layer_id, layer_type in zip(logical_layer_ids, layer_types)
            if layer_type == LINEAR_ATTENTION
        } | {
            f"layer.{logical_layer_id}.recurrent_state": recurrent_dtype
            for logical_layer_id, layer_type in zip(logical_layer_ids, layer_types)
            if layer_type == LINEAR_ATTENTION
        }
        logical_field_dtypes = {
            logical_field.field.field_id: state_dtypes.get(
                logical_field.field.field_id,
                attn_config.kv_cache_dtype,
            )
            for logical_field in logical_fields
        }
        reference_plan = stage_layout.layout.with_num_lcm_blocks(1)
        usable_cache_bytes = cache_budget_bytes - fixed_workspace_bytes
        max_num_lcm_blocks = (
            usable_cache_bytes // stage_layout.layout.lcm_block_bytes - 1
        )
        if max_num_lcm_blocks < 1:
            raise ValueError(
                "Kimi-K3 cache budget must hold a null parent and one usable "
                "LCM parent"
            )
        token_limit = configured_token_limit(server_args)
        sizing = {
            "max_scheduled_tokens": server_args.chunked_prefill_size,
            "max_live_requests": attn_config.max_bs,
            "decode_input_tokens": decode_input_tokens,
            "overlap_schedule_depth": overlap_schedule_depth,
        }
        num_lcm_blocks = max_num_lcm_blocks
        if token_limit is not None:
            num_lcm_blocks = min(
                num_lcm_blocks,
                kimi_k3_lcm_blocks_needed(
                    reference_plan,
                    token_capacity=token_limit,
                    **sizing,
                ),
            )
        capacity_upper_bound = token_limit
        if capacity_upper_bound is None:
            capacity_upper_bound = (
                max_num_lcm_blocks
                * max(
                    group.cache_blocks_per_lcm_block for group in reference_plan.groups
                )
                * reference_plan.logical_block_tokens
            )
        local_admitted_tokens = kimi_k3_token_capacity_for_cache_pool(
            reference_plan,
            num_lcm_blocks=num_lcm_blocks,
            upper_bound_tokens=capacity_upper_bound,
            **sizing,
        )
        cache_abi_digest = pipeline_cache_abi_digest(
            logical_fields,
            placement,
            global_layout,
            field_dtypes=logical_field_dtypes,
        )
        prepared = {
            "layout": stage_layout.layout,
            "logical_layer_ids": logical_layer_ids,
            "layer_types": layer_types,
            "group_ids": group_ids,
            "state_dtypes": state_dtypes,
            "reference_plan": reference_plan,
            "max_num_lcm_blocks": max_num_lcm_blocks,
            "local_admitted_tokens": local_admitted_tokens,
            "sizing": sizing,
            "pipeline_plan_digest": pipeline_plan.digest,
            "cache_abi_digest": cache_abi_digest,
            "cache_manifest_digest": stage_layout.manifest.stage_digest,
            "fixed_workspace_bytes": fixed_workspace_bytes,
        }
    except Exception as exc:  # noqa: BLE001 - every rank must join consensus
        local_error = exc

    raise_on_rank_error(local_error, mapping, "Kimi-K3 pipeline cache preflight")
    assert prepared is not None
    _validate_pipeline_cache_digest_consensus(
        prepared["cache_abi_digest"],
        prepared["cache_manifest_digest"],
        mapping,
    )
    admitted_tokens = _global_min_token_capacity(
        prepared["local_admitted_tokens"], mapping
    )

    result = None
    global_runtime_digest = None
    stage_runtime_digest = None
    local_error = None
    try:
        num_lcm_blocks = kimi_k3_lcm_blocks_needed(
            prepared["reference_plan"],
            token_capacity=admitted_tokens,
            **prepared["sizing"],
        )
        if num_lcm_blocks > prepared["max_num_lcm_blocks"]:
            raise ValueError(
                "global Kimi-K3 token capacity exceeds this stage's cache budget"
            )
        merged_plan = prepared["layout"].with_num_lcm_blocks(num_lcm_blocks)
        result = CacheSetup(
            spec=CachePoolSpec(
                family="kimi_k3",
                memory_plan=merged_plan,
                layer_types=prepared["layer_types"],
                layer_group_ids=prepared["group_ids"],
                paged_cache_group_specs=build_paged_cache_group_specs(
                    layer_types=prepared["layer_types"],
                    group_ids=prepared["group_ids"],
                    sliding_window_tokens=None,
                    page_size=merged_plan.logical_block_tokens,
                    pd_disaggregation_enabled=attn_config.pd_disaggregation_enabled,
                ),
                state_field_dtypes=prepared["state_dtypes"],
                token_capacity=admitted_tokens,
                logical_layer_ids=prepared["logical_layer_ids"],
                pipeline_plan_digest=prepared["pipeline_plan_digest"],
                cache_abi_digest=prepared["cache_abi_digest"],
                cache_manifest_digest=prepared["cache_manifest_digest"],
                runtime_parameters=(
                    (
                        "activation_dtype",
                        str(attn_config.dtype).removeprefix("torch."),
                    ),
                    ("attn_tp_size", attn_config.attn_tp_size),
                    (
                        "kv_cache_dtype",
                        str(attn_config.kv_cache_dtype).removeprefix("torch."),
                    ),
                    (
                        "kv_cache_quant_method",
                        attn_config.kv_cache_quant_method or "none",
                    ),
                    (
                        "pd_disaggregation_enabled",
                        attn_config.pd_disaggregation_enabled,
                    ),
                    *tuple(sorted(prepared["sizing"].items())),
                ),
            ),
            num_draft_layers=0,
            cache_budget_bytes=cache_budget_bytes,
            fixed_workspace_bytes=prepared["fixed_workspace_bytes"],
        )
        global_runtime_digest = result.spec.global_runtime_abi_digest
        stage_runtime_digest = result.spec.runtime_abi_digest
    except Exception as exc:  # noqa: BLE001 - every rank must join consensus
        local_error = exc
    raise_on_rank_error(local_error, mapping, "Kimi-K3 pipeline cache finalization")
    assert result is not None
    assert global_runtime_digest is not None
    assert stage_runtime_digest is not None
    _validate_pipeline_cache_digest_consensus(
        global_runtime_digest,
        stage_runtime_digest,
        mapping,
    )
    return result


def prepare_kimi_k3_cache(
    *,
    server_args,
    model_config,
    attn_config,
    draft_model_config,
    draft_attn_config,
    cache_budget_bytes: int,
    decode_input_tokens: int,
    overlap_schedule_depth: int,
):
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.setup import (
        CachePoolSpec,
        CacheSetup,
    )

    mapping = getattr(server_args, "mapping", None)
    stage_count = (
        mapping.pipeline.stage_count
        if mapping is not None and hasattr(mapping, "pipeline")
        else 1
    )
    if stage_count > 1:
        return _prepare_kimi_k3_pipeline_cache(
            server_args=server_args,
            model_config=model_config,
            attn_config=attn_config,
            draft_model_config=draft_model_config,
            draft_attn_config=draft_attn_config,
            cache_budget_bytes=cache_budget_bytes,
            decode_input_tokens=decode_input_tokens,
            overlap_schedule_depth=overlap_schedule_depth,
            mapping=mapping,
        )

    text_config = getattr(model_config.hf_config, "text_config", model_config.hf_config)
    global_group_ids = kimi_k3_layer_group_ids(text_config)
    global_layer_types = tuple(
        FULL_ATTENTION if group_id == FULL_ATTENTION else LINEAR_ATTENTION
        for group_id in global_group_ids
    )
    # One big model, one solve, one spec: draft MLA layers continue the
    # target's layer numbering and join the same solve, sharing the
    # full-attention group's packing and page-id space.
    draft_fields = None
    draft_layer_types = ()
    num_draft_layers = 0
    if draft_attn_config is not None:
        num_draft_layers = draft_model_config.num_attention_layers
        draft_layer_types = (FULL_ATTENTION,) * num_draft_layers
        draft_fields = mla_cache_fields(
            layer_group_ids=draft_layer_types,
            logical_block_tokens=_KIMI_K3_LOGICAL_BLOCK_TOKENS,
            latent_width=(
                draft_attn_config.kv_lora_rank + draft_attn_config.qk_rope_head_dim
            ),
            element_size=draft_attn_config.kv_cache_dtype.itemsize,
        )
    global_layout = solve_kimi_k3_cache_layout(
        text_config,
        tp_size=attn_config.attn_tp_size,
        mla_cache_dtype=attn_config.kv_cache_dtype,
        mla_quant_method=attn_config.kv_cache_quant_method or None,
        draft_fields=draft_fields,
    )
    merged_layout = global_layout
    layer_types = global_layer_types
    group_ids = global_group_ids

    _, _, conv_dtype, recurrent_dtype, _ = text_config.mamba2_cache_params
    logical_ids_for_fields = tuple(range(len(layer_types)))
    state_dtypes = {
        f"layer.{logical_layer_id}.conv_state": conv_dtype
        for logical_layer_id, layer_type in zip(logical_ids_for_fields, layer_types)
        if layer_type == LINEAR_ATTENTION
    } | {
        f"layer.{logical_layer_id}.recurrent_state": recurrent_dtype
        for logical_layer_id, layer_type in zip(logical_ids_for_fields, layer_types)
        if layer_type == LINEAR_ATTENTION
    }
    reference_plan = merged_layout.with_num_lcm_blocks(1)

    max_num_lcm_blocks = cache_budget_bytes // merged_layout.lcm_block_bytes - 1
    num_lcm_blocks = max_num_lcm_blocks
    if num_lcm_blocks < 1:
        raise ValueError(
            "Kimi-K3 cache budget must hold a null parent and one usable LCM parent"
        )

    token_limit = configured_token_limit(server_args)
    sizing = {
        "max_scheduled_tokens": server_args.chunked_prefill_size,
        "max_live_requests": attn_config.max_bs,
        "decode_input_tokens": decode_input_tokens,
        "overlap_schedule_depth": overlap_schedule_depth,
    }
    if token_limit is not None:
        num_lcm_blocks = min(
            num_lcm_blocks,
            kimi_k3_lcm_blocks_needed(
                reference_plan,
                token_capacity=token_limit,
                **sizing,
            ),
        )
    capacity_upper_bound = token_limit
    admitted_tokens = kimi_k3_token_capacity_for_cache_pool(
        reference_plan,
        num_lcm_blocks=num_lcm_blocks,
        upper_bound_tokens=capacity_upper_bound,
        **sizing,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
        build_paged_cache_group_specs,
    )

    merged_plan = merged_layout.with_num_lcm_blocks(num_lcm_blocks)
    # ONE spec derivation over the merged layers (draft MLA layers all
    # join the shared full-attention group).
    merged_layer_types = layer_types + draft_layer_types
    merged_group_ids = group_ids + draft_layer_types
    return CacheSetup(
        spec=CachePoolSpec(
            family="kimi_k3",
            memory_plan=merged_plan,
            layer_types=merged_layer_types,
            layer_group_ids=merged_group_ids,
            paged_cache_group_specs=build_paged_cache_group_specs(
                layer_types=merged_layer_types,
                group_ids=merged_group_ids,
                sliding_window_tokens=None,
                page_size=merged_plan.logical_block_tokens,
                pd_disaggregation_enabled=attn_config.pd_disaggregation_enabled,
            ),
            state_field_dtypes=state_dtypes,
            token_capacity=admitted_tokens,
        ),
        num_draft_layers=num_draft_layers,
        cache_budget_bytes=cache_budget_bytes,
        fixed_workspace_bytes=0,
    )
