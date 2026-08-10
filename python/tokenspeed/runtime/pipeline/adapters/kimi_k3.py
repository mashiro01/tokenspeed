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

"""Static pipeline plan for Kimi-K3's AttnRes decoder."""

from __future__ import annotations

from collections.abc import Sequence

from tokenspeed.runtime.pipeline.contracts import (
    ActivationFieldSpec,
    ActivationSchema,
    PipelinePlan,
    StagePlan,
)


def kimi_k3_stage_checkpoint_weight_filter(
    name: str,
    *,
    stage_plan: StagePlan,
    include_vision: bool,
) -> bool:
    """Return whether a checkpoint tensor can be consumed by this K3 stage."""

    if not isinstance(name, str) or not name:
        return False
    if name.startswith(("vision_tower.", "mm_projector.")):
        return include_vision
    language_prefix = "language_model."
    if name.startswith(language_prefix):
        name = name[len(language_prefix) :]
    if name.startswith("model.layers."):
        parts = name.split(".", 3)
        if len(parts) < 4 or not parts[2].isdigit():
            return False
        layer_id = int(parts[2])
        return stage_plan.first_layer <= layer_id < stage_plan.end_layer
    if name.startswith("model.embed_tokens."):
        return stage_plan.owns_embedding
    if name.startswith(
        (
            "model.norm.",
            "model.output_attn_res_norm.",
            "model.output_attn_res_proj.",
            "lm_head.",
        )
    ):
        return stage_plan.owns_head
    return False


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def balanced_kimi_k3_stage_layer_counts(
    *,
    num_layers: int,
    attn_res_block_size: int,
    stage_count: int,
) -> tuple[int, ...]:
    """Choose balanced K3 stage ranges on AttnRes block boundaries.

    Every non-final boundary is constrained to a completed AttnRes block. The
    selected boundary is the legal block edge nearest to the proportional
    model split, while reserving at least one layer for every remaining stage.
    """

    num_layers = _positive_int("num_layers", num_layers)
    attn_res_block_size = _positive_int("attn_res_block_size", attn_res_block_size)
    stage_count = _positive_int("stage_count", stage_count)
    if stage_count == 1:
        return (num_layers,)

    last_boundary_block = (num_layers - 1) // attn_res_block_size
    if stage_count - 1 > last_boundary_block:
        raise ValueError(
            f"stage_count={stage_count} cannot place {stage_count - 1} "
            f"AttnRes-aligned boundaries across {num_layers} layers"
        )

    boundary_blocks: list[int] = []
    previous = 0
    for boundary_index in range(1, stage_count):
        remaining_boundaries = stage_count - 1 - boundary_index
        first_candidate = previous + 1
        last_candidate = last_boundary_block - remaining_boundaries
        candidate = min(
            range(first_candidate, last_candidate + 1),
            key=lambda block_id: (
                abs(
                    block_id * attn_res_block_size * stage_count
                    - boundary_index * num_layers
                ),
                block_id,
            ),
        )
        boundary_blocks.append(candidate)
        previous = candidate

    boundaries = tuple(block_id * attn_res_block_size for block_id in boundary_blocks)
    edges = (0, *boundaries, num_layers)
    return tuple(right - left for left, right in zip(edges, edges[1:]))


def build_balanced_kimi_k3_pipeline_plan(
    *,
    num_layers: int,
    hidden_size: int,
    attn_res_block_size: int,
    stage_count: int,
    activation_dtype: str = "bfloat16",
    dspark_context_hidden_size: int | None = None,
    dspark_context_dtype: str | None = None,
) -> PipelinePlan:
    """Build the default balanced K3 plan for a pipeline stage count."""

    return build_kimi_k3_pipeline_plan(
        num_layers=num_layers,
        hidden_size=hidden_size,
        attn_res_block_size=attn_res_block_size,
        stage_layer_counts=balanced_kimi_k3_stage_layer_counts(
            num_layers=num_layers,
            attn_res_block_size=attn_res_block_size,
            stage_count=stage_count,
        ),
        activation_dtype=activation_dtype,
        dspark_context_hidden_size=dspark_context_hidden_size,
        dspark_context_dtype=dspark_context_dtype,
    )


def build_kimi_k3_pipeline_plan(
    *,
    num_layers: int,
    hidden_size: int,
    attn_res_block_size: int,
    stage_layer_counts: Sequence[int],
    activation_dtype: str = "bfloat16",
    dspark_context_hidden_size: int | None = None,
    dspark_context_dtype: str | None = None,
) -> PipelinePlan:
    """Build K3 stage ownership and chain-full AttnRes boundary schemas.

    Every non-final boundary must align with an AttnRes block. The payload
    carries the current prefix stream and one tensor for every completed block
    snapshot. Keeping snapshots as separate ``[tokens, hidden]`` fields avoids
    making the dynamic token dimension part of a fixed trailing shape.

    Args:
        num_layers: Number of decoder layers in the full text model.
        hidden_size: Width of each activation and residual snapshot.
        attn_res_block_size: Decoder layers represented by one snapshot.
        stage_layer_counts: Positive decoder-layer count for every stage.
        activation_dtype: Runtime dtype string used by activation validation.
        dspark_context_hidden_size: When set, reserve one projected DSpark
            context stream at every PP boundary. The stream is an accumulated
            projection rather than a relay of every raw target hidden state.
        dspark_context_dtype: Wire dtype of the accumulated DSpark context.
            Defaults to float32 so partial projections are accumulated once
            without BF16 rounding at every target stage.

    Returns:
        A validated generic :class:`PipelinePlan`.
    """

    num_layers = _positive_int("num_layers", num_layers)
    hidden_size = _positive_int("hidden_size", hidden_size)
    attn_res_block_size = _positive_int("attn_res_block_size", attn_res_block_size)
    counts = tuple(stage_layer_counts)
    if not counts:
        raise ValueError("stage_layer_counts must not be empty")
    for count in counts:
        _positive_int("stage layer count", count)
    if sum(counts) != num_layers:
        raise ValueError(
            f"stage layer counts sum to {sum(counts)}, expected {num_layers}"
        )
    if not isinstance(activation_dtype, str) or not activation_dtype:
        raise ValueError("activation_dtype must be a non-empty string")
    context_dtype = None
    if dspark_context_hidden_size is not None:
        _positive_int("dspark_context_hidden_size", dspark_context_hidden_size)
        context_dtype = (
            "float32" if dspark_context_dtype is None else dspark_context_dtype
        )
        if not isinstance(context_dtype, str) or not context_dtype:
            raise ValueError("dspark_context_dtype must be a non-empty string")
    elif dspark_context_dtype is not None:
        raise ValueError(
            "dspark_context_dtype requires dspark_context_hidden_size"
        )

    boundaries: list[int] = []
    cursor = 0
    for count in counts[:-1]:
        cursor += count
        if cursor % attn_res_block_size:
            raise ValueError(
                f"K3 boundary after layer {cursor} does not align with "
                f"AttnRes block size {attn_res_block_size}"
            )
        boundaries.append(cursor)

    schemas = tuple(
        _boundary_schema(
            source_stage=stage_id,
            end_layer=end_layer,
            hidden_size=hidden_size,
            attn_res_block_size=attn_res_block_size,
            activation_dtype=activation_dtype,
            dspark_context_hidden_size=dspark_context_hidden_size,
            dspark_context_dtype=context_dtype,
        )
        for stage_id, end_layer in enumerate(boundaries)
    )

    stages = []
    first_layer = 0
    stage_count = len(counts)
    for stage_id, count in enumerate(counts):
        end_layer = first_layer + count
        stages.append(
            StagePlan(
                stage_id=stage_id,
                stage_count=stage_count,
                first_layer=first_layer,
                end_layer=end_layer,
                owns_embedding=stage_id == 0,
                owns_head=stage_id == stage_count - 1,
                input_schema=(None if stage_id == 0 else schemas[stage_id - 1]),
                output_schema=(
                    None if stage_id == stage_count - 1 else schemas[stage_id]
                ),
            )
        )
        first_layer = end_layer
    return PipelinePlan(tuple(stages))


def _boundary_schema(
    *,
    source_stage: int,
    end_layer: int,
    hidden_size: int,
    attn_res_block_size: int,
    activation_dtype: str,
    dspark_context_hidden_size: int | None,
    dspark_context_dtype: str | None,
) -> ActivationSchema:
    completed_blocks = end_layer // attn_res_block_size
    fields = [
        ActivationFieldSpec(
            field_id="prefix_sum",
            dtype=activation_dtype,
            trailing_shape=(hidden_size,),
        )
    ]
    fields.extend(
        ActivationFieldSpec(
            field_id=f"block_residual.{block_id}",
            dtype=activation_dtype,
            trailing_shape=(hidden_size,),
        )
        for block_id in range(completed_blocks)
    )
    if dspark_context_hidden_size is not None:
        assert dspark_context_dtype is not None
        fields.append(
            ActivationFieldSpec(
                field_id="dspark_context",
                dtype=dspark_context_dtype,
                trailing_shape=(dspark_context_hidden_size,),
            )
        )
    return ActivationSchema(
        boundary_id=(
            f"kimi-k3/attn-res-{attn_res_block_size}/after-layer-{end_layer}/"
            f"stage-{source_stage}-to-{source_stage + 1}"
        ),
        fields=tuple(fields),
    )
