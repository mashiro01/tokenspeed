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

from collections.abc import Sequence
from dataclasses import dataclass

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
