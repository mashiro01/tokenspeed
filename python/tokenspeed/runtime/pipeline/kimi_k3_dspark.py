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

"""Static placement contract for Kimi-K3 DSpark on native pipeline parallelism.

K3 DSpark consumes five target-layer streams through one dense ``context_proj``.
For PP, sending all five full hidden tensors to one stage would multiply the
activation traffic and strand the draft on PP7, where K3 has the least memory.
Instead every target stage owns the corresponding input-column slice of
``context_proj`` and forwards one accumulated projected context tensor.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from tokenspeed.runtime.pipeline.contracts import PipelinePlan


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


@dataclass(frozen=True)
class KimiK3DSparkProjectionSlice:
    """One target tap's input-column slice of the DSpark context projection."""

    target_layer_id: int
    stage_id: int
    input_start: int
    input_end: int

    @property
    def input_width(self) -> int:
        return self.input_end - self.input_start


@dataclass(frozen=True)
class KimiK3DSparkPlacement:
    """Stable PP ownership for the K3 target, verify, and DSpark draft paths."""

    draft_owner_stage_id: int
    verify_stage_id: int
    target_hidden_size: int
    context_hidden_size: int
    target_layer_ids: tuple[int, ...]
    projection_slices: tuple[KimiK3DSparkProjectionSlice, ...]

    def projection_slices_for_stage(
        self, stage_id: int
    ) -> tuple[KimiK3DSparkProjectionSlice, ...]:
        """Return the context-projection slices computed by one target stage."""

        return tuple(
            slice_
            for slice_ in self.projection_slices
            if slice_.stage_id == stage_id
        )


def split_kimi_k3_dspark_context_projection(
    context_projection_weight: torch.Tensor,
    placement: KimiK3DSparkPlacement,
) -> dict[int, dict[int, torch.Tensor]]:
    """Split a loaded context projection into stage-local target-tap slices.

    ``context_proj`` is an ordinary linear weight with shape
    ``[context_hidden_size, num_taps * target_hidden_size]``. The returned
    buffers retain the original dtype; :class:`KimiK3DSparkStageProjector`
    accumulates each GEMM into FP32 before the draft applies ``context_norm``.
    """

    if not isinstance(context_projection_weight, torch.Tensor):
        raise TypeError("context_projection_weight must be a torch.Tensor")
    expected_shape = (
        placement.context_hidden_size,
        len(placement.target_layer_ids) * placement.target_hidden_size,
    )
    if tuple(context_projection_weight.shape) != expected_shape:
        raise ValueError(
            "Kimi-K3 DSpark context projection has shape "
            f"{tuple(context_projection_weight.shape)}, expected {expected_shape}"
        )

    by_stage: dict[int, dict[int, torch.Tensor]] = {}
    for slice_ in placement.projection_slices:
        by_stage.setdefault(slice_.stage_id, {})[slice_.target_layer_id] = (
            context_projection_weight[:, slice_.input_start : slice_.input_end]
            .detach()
            .contiguous()
        )
    return by_stage


class KimiK3DSparkStageProjector(nn.Module):
    """Apply one pipeline stage's DSpark context-projection column slices.

    The target hidden stream is BF16 in the usual K3 path. CUDA matmul's
    FP32-output mode retains one FP32 accumulator across all stages, avoiding
    a BF16 round-trip for each of the five target taps. CPU uses an explicit
    FP32 fallback so this contract is unit-testable without a GPU.
    """

    def __init__(
        self,
        placement: KimiK3DSparkPlacement,
        *,
        stage_id: int,
        projection_weights: Mapping[int, torch.Tensor],
    ) -> None:
        super().__init__()
        if isinstance(stage_id, bool) or not isinstance(stage_id, int):
            raise TypeError("stage_id must be an integer")
        local_slices = placement.projection_slices_for_stage(stage_id)
        expected_ids = {slice_.target_layer_id for slice_ in local_slices}
        provided_ids = set(projection_weights)
        if provided_ids != expected_ids:
            raise ValueError(
                "Kimi-K3 DSpark stage projection weights must cover exactly "
                f"{sorted(expected_ids)}, got {sorted(provided_ids)}"
            )

        self.placement = placement
        self.stage_id = stage_id
        self._slices_by_layer = {
            slice_.target_layer_id: slice_ for slice_ in local_slices
        }
        self._weight_names: dict[int, str] = {}
        expected_shape = (
            placement.context_hidden_size,
            placement.target_hidden_size,
        )
        for index, slice_ in enumerate(local_slices):
            weight = projection_weights[slice_.target_layer_id]
            if not isinstance(weight, torch.Tensor):
                raise TypeError(
                    "Kimi-K3 DSpark context projection slices must be tensors"
                )
            if tuple(weight.shape) != expected_shape:
                raise ValueError(
                    "Kimi-K3 DSpark context projection slice for layer "
                    f"{slice_.target_layer_id} has shape {tuple(weight.shape)}, "
                    f"expected {expected_shape}"
                )
            name = f"context_weight_{index}"
            self.register_buffer(name, weight.detach().contiguous(), persistent=True)
            self._weight_names[slice_.target_layer_id] = name

    def new_context(self, num_tokens: int, *, device: torch.device) -> torch.Tensor:
        """Allocate the FP32 projected-context accumulator for one PP step."""

        if isinstance(num_tokens, bool) or not isinstance(num_tokens, int):
            raise TypeError("num_tokens must be an integer")
        if num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")
        return torch.zeros(
            (num_tokens, self.placement.context_hidden_size),
            dtype=torch.float32,
            device=device,
        )

    def owns_target_layer(self, target_layer_id: int) -> bool:
        """Return whether this stage owns a DSpark target-layer tap."""

        return target_layer_id in self._weight_names

    @torch.no_grad()
    def accumulate(
        self,
        context: torch.Tensor,
        *,
        target_layer_id: int,
        target_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Add one local target tap's FP32 partial projection in place."""

        try:
            weight = getattr(self, self._weight_names[target_layer_id])
        except KeyError as exc:
            raise ValueError(
                "Kimi-K3 DSpark stage "
                f"{self.stage_id} does not own target layer {target_layer_id}"
            ) from exc
        if target_hidden.ndim != 2 or target_hidden.shape[1] != self.placement.target_hidden_size:
            raise ValueError(
                "Kimi-K3 DSpark target hidden must have shape "
                f"[tokens, {self.placement.target_hidden_size}], got "
                f"{tuple(target_hidden.shape)}"
            )
        expected_shape = (
            target_hidden.shape[0],
            self.placement.context_hidden_size,
        )
        if tuple(context.shape) != expected_shape:
            raise ValueError(
                "Kimi-K3 DSpark context has shape "
                f"{tuple(context.shape)}, expected {expected_shape}"
            )
        if context.dtype != torch.float32:
            raise ValueError("Kimi-K3 DSpark context accumulator must be float32")
        if context.device != target_hidden.device or context.device != weight.device:
            raise ValueError("Kimi-K3 DSpark context, target, and weight must share a device")

        if context.device.type == "cuda":
            partial = torch.mm(target_hidden, weight.t(), out_dtype=torch.float32)
        else:
            partial = torch.mm(target_hidden.float(), weight.float().t())
        context.add_(partial)
        return context


def resolve_kimi_k3_dspark_placement(
    pipeline_plan: PipelinePlan,
    *,
    target_layer_ids: Sequence[int],
    target_hidden_size: int,
    context_hidden_size: int,
    draft_owner_stage_id: int = 0,
) -> KimiK3DSparkPlacement:
    """Resolve the only supported K3 DSpark PP ownership layout.

    PP0 owns the draft because it already owns K3's embedding and has the
    cluster's available memory headroom. PP7 remains the target verification
    stage. It returns verified output and the accumulated context to PP0 over
    the existing pipeline lane; PP0 then produces the next candidate block.
    """

    if not isinstance(pipeline_plan, PipelinePlan):
        raise TypeError("pipeline_plan must be a PipelinePlan")
    target_hidden_size = _positive_int("target_hidden_size", target_hidden_size)
    context_hidden_size = _positive_int("context_hidden_size", context_hidden_size)
    if (
        isinstance(draft_owner_stage_id, bool)
        or not isinstance(draft_owner_stage_id, int)
        or not 0 <= draft_owner_stage_id < len(pipeline_plan.stages)
    ):
        raise ValueError("draft_owner_stage_id is outside the pipeline plan")
    if draft_owner_stage_id != 0:
        raise ValueError("Kimi-K3 DSpark draft owner must be pipeline stage 0")
    if not pipeline_plan.stages[0].owns_embedding:
        raise ValueError("Kimi-K3 DSpark draft owner must own target embeddings")

    layer_ids = tuple(target_layer_ids)
    if not layer_ids:
        raise ValueError("Kimi-K3 DSpark requires at least one target layer id")
    if any(
        isinstance(layer_id, bool) or not isinstance(layer_id, int)
        for layer_id in layer_ids
    ):
        raise ValueError("Kimi-K3 DSpark target layer ids must be integers")
    if tuple(sorted(layer_ids)) != layer_ids:
        raise ValueError("Kimi-K3 DSpark target layer ids must be sorted ascending")
    if len(set(layer_ids)) != len(layer_ids):
        raise ValueError("Kimi-K3 DSpark target layer ids must be unique")

    projection_slices = []
    for feature_index, layer_id in enumerate(layer_ids):
        owner = next(
            (
                stage
                for stage in pipeline_plan.stages
                if stage.first_layer <= layer_id < stage.end_layer
            ),
            None,
        )
        if owner is None:
            last_layer = pipeline_plan.stages[-1].end_layer - 1
            raise ValueError(
                "Kimi-K3 DSpark target layer id "
                f"{layer_id} is outside [0, {last_layer}]"
            )
        input_start = feature_index * target_hidden_size
        projection_slices.append(
            KimiK3DSparkProjectionSlice(
                target_layer_id=layer_id,
                stage_id=owner.stage_id,
                input_start=input_start,
                input_end=input_start + target_hidden_size,
            )
        )

    return KimiK3DSparkPlacement(
        draft_owner_stage_id=draft_owner_stage_id,
        verify_stage_id=pipeline_plan.stages[-1].stage_id,
        target_hidden_size=target_hidden_size,
        context_hidden_size=context_hidden_size,
        target_layer_ids=layer_ids,
        projection_slices=tuple(projection_slices),
    )
