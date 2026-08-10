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

"""Model-independent projection of logical cache fields onto one stage."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    CacheLayout,
    solve_cache_layout,
)
from tokenspeed.runtime.pipeline.contracts import PipelinePlan

_MANIFEST_VERSION = 1


def _cache_layout_payload(layout: CacheLayout) -> dict[str, object]:
    return {
        "logical_block_tokens": layout.logical_block_tokens,
        "lcm_block_bytes": layout.lcm_block_bytes,
        "group_packing": [
            [group_id, count] for group_id, count in sorted(layout.group_packing)
        ],
        "plane_bytes": [
            [plane_id, byte_count]
            for plane_id, byte_count in sorted(layout.plane_bytes)
        ],
        "fields": [
            {
                "group_id": field.group_id,
                "field_id": field.field_id,
                "plane_id": field.plane_id,
                "shape": list(field.shape),
                "element_size": field.element_size,
                "field_offset_bytes": field.field_offset_bytes,
                "page_stride_bytes": field.page_stride_bytes,
            }
            for field in sorted(
                layout.fields,
                key=lambda item: (
                    item.plane_id,
                    item.group_id,
                    item.field_id,
                ),
            )
        ],
    }


def _digest_payload(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class LogicalCacheFieldSpec:
    """A cache field with explicit logical-layer ownership.

    ``field_id`` and ``plane_id`` remain opaque recipe-owned identifiers. Stage
    projection uses ``logical_layer_id`` exclusively and never parses either
    identifier.

    Args:
        logical_layer_id: Layer in the unpartitioned logical model.
        field: Existing cache field geometry and placement specification.
    """

    logical_layer_id: int
    field: CacheFieldSpec


@dataclass(frozen=True)
class CacheStagePlacement:
    """Complete cache-layer ownership for one rank's pipeline replica.

    The full partition is carried on every rank so duplicate, omitted, and
    out-of-range ownership can be rejected before a local layout is produced.

    Args:
        rank: Global rank that will own the projected layout.
        world_size: Total global ranks across all stages.
        stage_id: This rank's stage index.
        stage_count: Number of stages in the pipeline replica.
        num_logical_layers: Number of layers in the unpartitioned model.
        logical_layer_ids_by_stage: Explicit logical layer ids owned by every
            stage, indexed by stage id.
        pipeline_plan_digest: Digest of the layer ownership plan.
    """

    rank: int
    world_size: int
    stage_id: int
    stage_count: int
    num_logical_layers: int
    logical_layer_ids_by_stage: tuple[tuple[int, ...], ...]
    pipeline_plan_digest: str

    @classmethod
    def from_pipeline_plan(
        cls,
        *,
        rank: int,
        world_size: int,
        plan: PipelinePlan,
    ) -> "CacheStagePlacement":
        """Derive cache ownership from the validated execution plan."""

        stage_count = len(plan.stages)
        if not _is_int(world_size) or world_size < 1:
            raise ValueError("world_size must be a positive integer")
        if world_size % stage_count:
            raise ValueError("world_size must be divisible by stage_count")
        if not _is_int(rank) or rank < 0 or rank >= world_size:
            raise ValueError("rank is outside the global world")
        stage_world_size = world_size // stage_count
        stage_id = rank // stage_world_size
        return cls(
            rank=rank,
            world_size=world_size,
            stage_id=stage_id,
            stage_count=stage_count,
            num_logical_layers=plan.stages[-1].end_layer,
            logical_layer_ids_by_stage=tuple(
                tuple(range(stage.first_layer, stage.end_layer))
                for stage in plan.stages
            ),
            pipeline_plan_digest=plan.digest,
        )


@dataclass(frozen=True)
class CacheLayerBinding:
    """Translation from a logical model layer to a dense local layer id.

    Args:
        logical_layer_id: Layer id used by the unpartitioned model.
        physical_layer_id: Dense zero-based layer id on the owning stage.
    """

    logical_layer_id: int
    physical_layer_id: int


@dataclass(frozen=True)
class RankCacheLayoutManifest:
    """Canonical description of one rank's projected cache layout.

    Args:
        rank: Global rank owning the layout.
        world_size: Total global ranks across all stages.
        stage_id: Pipeline stage index on the rank.
        stage_count: Number of stages in the pipeline replica.
        num_logical_layers: Number of layers in the logical model.
        bindings: Logical-to-physical layer translations for this stage.
        local_group_ids: Cache groups with at least one local field.
        field_owners: Explicit ``(field_id, logical_layer_id)`` ownership.
        global_group_packing: Shared logical page count multiplier by group.
        pipeline_plan_digest: Digest of the layer ownership plan.
        layout: Capacity-independent local cache geometry.
    """

    rank: int
    world_size: int
    stage_id: int
    stage_count: int
    num_logical_layers: int
    bindings: tuple[CacheLayerBinding, ...]
    local_group_ids: tuple[str, ...]
    field_owners: tuple[tuple[str, int], ...]
    global_group_packing: tuple[tuple[str, int], ...]
    pipeline_plan_digest: str
    layout: CacheLayout

    def _canonical_payload(self, *, include_rank: bool) -> dict[str, object]:
        payload = {
            "version": _MANIFEST_VERSION,
            "world_size": self.world_size,
            "stage_id": self.stage_id,
            "stage_count": self.stage_count,
            "num_logical_layers": self.num_logical_layers,
            "bindings": [
                [binding.logical_layer_id, binding.physical_layer_id]
                for binding in sorted(
                    self.bindings,
                    key=lambda item: (
                        item.logical_layer_id,
                        item.physical_layer_id,
                    ),
                )
            ],
            "local_group_ids": sorted(self.local_group_ids),
            "field_owners": [
                [field_id, logical_layer_id]
                for field_id, logical_layer_id in sorted(self.field_owners)
            ],
            "global_group_packing": [
                [group_id, count]
                for group_id, count in sorted(self.global_group_packing)
            ],
            "pipeline_plan_digest": self.pipeline_plan_digest,
            "layout": _cache_layout_payload(self.layout),
        }
        if include_rank:
            payload["rank"] = self.rank
        return payload

    def canonical_json(self) -> str:
        """Return a stable, whitespace-free rank manifest."""

        return json.dumps(
            self._canonical_payload(include_rank=True),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )

    def stage_canonical_json(self) -> str:
        """Return a stable stage ABI projection shared by its TP ranks."""

        return json.dumps(
            self._canonical_payload(include_rank=False),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )

    @property
    def digest(self) -> str:
        """Return the SHA-256 digest of :meth:`canonical_json`."""

        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @property
    def stage_digest(self) -> str:
        """Return a rank-independent digest for same-stage consensus."""

        return hashlib.sha256(self.stage_canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class StageCacheLayout:
    """A projected stage layout and its logical-to-physical metadata.

    Args:
        layout: Capacity-independent geometry returned by the existing solver.
        logical_fields: Local fields with their explicit logical ownership.
        bindings: Dense logical-to-physical layer translations.
        local_group_ids: Cache groups represented by local fields.
        global_group_packing: Shared logical page count multiplier by group.
        manifest: Canonical per-rank description of this projection.
    """

    layout: CacheLayout
    logical_fields: tuple[LogicalCacheFieldSpec, ...]
    bindings: tuple[CacheLayerBinding, ...]
    local_group_ids: tuple[str, ...]
    global_group_packing: tuple[tuple[str, int], ...]
    manifest: RankCacheLayoutManifest

    @property
    def logical_to_physical(self) -> dict[int, int]:
        """Return the stage's logical-to-physical layer translation."""

        return {
            binding.logical_layer_id: binding.physical_layer_id
            for binding in self.bindings
        }


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_placement(placement: CacheStagePlacement) -> dict[int, int]:
    if not _is_int(placement.rank) or placement.rank < 0:
        raise ValueError("rank must be a non-negative integer")
    if not _is_int(placement.world_size) or placement.world_size < 1:
        raise ValueError("world_size must be a positive integer")
    if not _is_int(placement.stage_count) or placement.stage_count < 1:
        raise ValueError("stage_count must be a positive integer")
    if (
        not _is_int(placement.stage_id)
        or placement.stage_id < 0
        or placement.stage_id >= placement.stage_count
    ):
        raise ValueError("stage_id is out of range")
    if not _is_int(placement.num_logical_layers) or placement.num_logical_layers < 1:
        raise ValueError("num_logical_layers must be a positive integer")
    if len(placement.logical_layer_ids_by_stage) != placement.stage_count:
        raise ValueError("logical_layer_ids_by_stage must have one entry per stage")
    if placement.world_size % placement.stage_count:
        raise ValueError("world_size must be divisible by stage_count")
    if placement.rank >= placement.world_size:
        raise ValueError("rank is outside the global world")
    stage_world_size = placement.world_size // placement.stage_count
    expected_stage_id = placement.rank // stage_world_size
    if placement.stage_id != expected_stage_id:
        raise ValueError(
            f"rank {placement.rank} belongs to stage {expected_stage_id}, "
            f"not stage {placement.stage_id}"
        )
    if (
        not isinstance(placement.pipeline_plan_digest, str)
        or len(placement.pipeline_plan_digest) != 64
        or any(
            character not in "0123456789abcdef"
            for character in placement.pipeline_plan_digest
        )
    ):
        raise ValueError("pipeline_plan_digest must be a lowercase SHA-256 digest")

    owner_by_layer: dict[int, int] = {}
    for stage_id, logical_layer_ids in enumerate(placement.logical_layer_ids_by_stage):
        for logical_layer_id in logical_layer_ids:
            if (
                not _is_int(logical_layer_id)
                or logical_layer_id < 0
                or logical_layer_id >= placement.num_logical_layers
            ):
                raise ValueError(
                    f"ownership layer id {logical_layer_id!r} is out of range"
                )
            previous_stage = owner_by_layer.get(logical_layer_id)
            if previous_stage is not None:
                raise ValueError(
                    f"duplicate ownership for logical layer {logical_layer_id}: "
                    f"stages {previous_stage} and {stage_id}"
                )
            owner_by_layer[logical_layer_id] = stage_id

    missing = sorted(set(range(placement.num_logical_layers)) - owner_by_layer.keys())
    if missing:
        raise ValueError(f"omitted ownership for logical layers {missing}")
    flattened = tuple(
        logical_layer_id
        for logical_layer_ids in placement.logical_layer_ids_by_stage
        for logical_layer_id in logical_layer_ids
    )
    if any(
        not logical_layer_ids
        for logical_layer_ids in placement.logical_layer_ids_by_stage
    ):
        raise ValueError("every stage must own at least one logical layer")
    if flattened != tuple(range(placement.num_logical_layers)):
        raise ValueError("stage layer ownership must be ordered and contiguous")
    return owner_by_layer


def pipeline_cache_abi_digest(
    fields: Sequence[LogicalCacheFieldSpec],
    placement: CacheStagePlacement,
    global_layout: CacheLayout,
    *,
    field_dtypes: Mapping[str, object] | None = None,
) -> str:
    """Digest the complete rank-independent cache and ownership ABI."""

    logical_fields = tuple(fields)
    if not logical_fields:
        raise ValueError("at least one logical cache field is required")
    owner_by_layer = _validate_placement(placement)
    field_owners: list[list[object]] = []
    field_ids: set[str] = set()
    for logical_field in logical_fields:
        if not isinstance(logical_field, LogicalCacheFieldSpec) or not isinstance(
            logical_field.field, CacheFieldSpec
        ):
            raise ValueError("fields must contain logical CacheFieldSpec values")
        logical_layer_id = logical_field.logical_layer_id
        if not _is_int(logical_layer_id) or logical_layer_id not in owner_by_layer:
            raise ValueError(
                f"logical cache field owner {logical_layer_id!r} is unknown"
            )
        field_id = logical_field.field.field_id
        if field_id in field_ids:
            raise ValueError(f"duplicate logical cache field id {field_id!r}")
        field_ids.add(field_id)
        field_owners.append([field_id, logical_layer_id])
    layout_field_ids = {field.field_id for field in global_layout.fields}
    if field_ids != layout_field_ids:
        raise ValueError("global cache layout fields do not match logical fields")
    normalized_field_dtypes = []
    if field_dtypes is not None:
        dtype_field_ids = set(field_dtypes)
        missing = field_ids - dtype_field_ids
        unknown = dtype_field_ids - field_ids
        if missing or unknown:
            raise ValueError(
                "cache field dtype mapping must exactly cover logical fields; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        for field_id, dtype in sorted(field_dtypes.items()):
            dtype_name = str(dtype).removeprefix("torch.")
            if not dtype_name:
                raise ValueError("cache field dtype names must not be empty")
            normalized_field_dtypes.append([field_id, dtype_name])

    return _digest_payload(
        {
            "version": _MANIFEST_VERSION,
            "world_size": placement.world_size,
            "stage_count": placement.stage_count,
            "num_logical_layers": placement.num_logical_layers,
            "logical_layer_ids_by_stage": [
                list(layer_ids) for layer_ids in placement.logical_layer_ids_by_stage
            ],
            "field_owners": sorted(field_owners),
            "field_dtypes": normalized_field_dtypes,
            "pipeline_plan_digest": placement.pipeline_plan_digest,
            "layout": _cache_layout_payload(global_layout),
        }
    )


def solve_stage_cache_layout(
    fields: Sequence[LogicalCacheFieldSpec],
    placement: CacheStagePlacement,
    *,
    logical_block_tokens: int,
    cache_blocks_per_lcm_block: Mapping[str, int] | None = None,
    alignment: int = 1,
    max_padding_fraction: float = 0.25,
    compact_group_planes: bool = False,
) -> StageCacheLayout:
    """Project logical cache fields onto one stage and solve local geometry.

    Args:
        fields: Cache fields with explicit logical-layer ownership.
        placement: Complete pipeline ownership and the rank being projected.
        logical_block_tokens: Scheduler token count represented by one page.
        cache_blocks_per_lcm_block: Optional global per-group packing. Groups
            with no field on this stage are accepted and omitted locally.
        alignment: Byte alignment forwarded to :func:`solve_cache_layout`.
        max_padding_fraction: Padding limit forwarded to the existing solver.
        compact_group_planes: Renumber each local group's planes by occurrence.
            This preserves field and group identities while removing holes left
            by projection from a global interleaved layout.

    Returns:
        The local layout, dense layer bindings, and canonical rank manifest.

    Raises:
        ValueError: If fields or ownership are empty, duplicate, incomplete,
            out of range, or otherwise unknown to the declared placement.
    """

    logical_fields = tuple(fields)
    if not logical_fields:
        raise ValueError("at least one logical cache field is required")

    owner_by_layer = _validate_placement(placement)
    field_ids: set[str] = set()
    all_group_ids: set[str] = set()
    for logical_field in logical_fields:
        if not isinstance(logical_field, LogicalCacheFieldSpec):
            raise ValueError("fields must contain LogicalCacheFieldSpec values")
        if not isinstance(logical_field.field, CacheFieldSpec):
            raise ValueError("logical cache fields must contain CacheFieldSpec values")
        if not _is_int(logical_field.logical_layer_id):
            raise ValueError("logical cache field owner must be an integer")
        if logical_field.logical_layer_id not in owner_by_layer:
            raise ValueError(
                "logical cache field "
                f"{logical_field.field.field_id!r} has unknown ownership for "
                f"layer {logical_field.logical_layer_id}"
            )
        if logical_field.field.field_id in field_ids:
            raise ValueError(
                f"duplicate logical cache field id {logical_field.field.field_id!r}"
            )
        field_ids.add(logical_field.field.field_id)
        all_group_ids.add(logical_field.field.group_id)

    if cache_blocks_per_lcm_block is not None:
        unknown_groups = set(cache_blocks_per_lcm_block) - all_group_ids
        if unknown_groups:
            raise ValueError(
                "cache_blocks_per_lcm_block names unknown groups: "
                f"{sorted(unknown_groups)}"
            )
        if any(
            not _is_int(count) or count < 1
            for count in cache_blocks_per_lcm_block.values()
        ):
            raise ValueError("cache group packing must be a positive integer")

    global_layout = solve_cache_layout(
        (logical_field.field for logical_field in logical_fields),
        logical_block_tokens=logical_block_tokens,
        cache_blocks_per_lcm_block=cache_blocks_per_lcm_block,
        alignment=alignment,
        max_padding_fraction=max_padding_fraction,
    )
    global_group_packing = global_layout.group_packing
    global_packing_by_group = dict(global_group_packing)

    local_layer_ids = tuple(
        sorted(placement.logical_layer_ids_by_stage[placement.stage_id])
    )
    local_layer_id_set = set(local_layer_ids)
    projected_fields = tuple(
        sorted(
            (
                logical_field
                for logical_field in logical_fields
                if logical_field.logical_layer_id in local_layer_id_set
            ),
            key=lambda item: (
                item.field.plane_id,
                item.field.group_id,
                item.field.field_id,
                item.logical_layer_id,
            ),
        )
    )
    if not projected_fields:
        raise ValueError("the selected stage must own at least one cache field")
    if compact_group_planes:
        next_plane_by_group: dict[str, int] = {}
        compact_plane_ids: dict[tuple[str, str], str] = {}
        compacted_fields = []
        for logical_field in sorted(
            projected_fields,
            key=lambda item: (
                item.logical_layer_id,
                item.field.group_id,
                item.field.plane_id,
                item.field.field_id,
            ),
        ):
            field = logical_field.field
            plane_key = (field.group_id, field.plane_id)
            plane_id = compact_plane_ids.get(plane_key)
            if plane_id is None:
                local_slot = next_plane_by_group.get(field.group_id, 0)
                next_plane_by_group[field.group_id] = local_slot + 1
                plane_id = f"stage-slot.{local_slot}"
                compact_plane_ids[plane_key] = plane_id
            compacted_fields.append(
                replace(logical_field, field=replace(field, plane_id=plane_id))
            )
        projected_fields = tuple(compacted_fields)

    bindings = tuple(
        CacheLayerBinding(
            logical_layer_id=logical_layer_id,
            physical_layer_id=physical_layer_id,
        )
        for physical_layer_id, logical_layer_id in enumerate(local_layer_ids)
    )
    local_group_ids = tuple(
        sorted({logical_field.field.group_id for logical_field in projected_fields})
    )
    local_packing = {
        group_id: global_packing_by_group[group_id] for group_id in local_group_ids
    }

    layout = solve_cache_layout(
        (logical_field.field for logical_field in projected_fields),
        logical_block_tokens=logical_block_tokens,
        cache_blocks_per_lcm_block=local_packing,
        alignment=alignment,
        max_padding_fraction=max_padding_fraction,
    )
    field_owners = tuple(
        sorted(
            (
                logical_field.field.field_id,
                logical_field.logical_layer_id,
            )
            for logical_field in projected_fields
        )
    )
    manifest = RankCacheLayoutManifest(
        rank=placement.rank,
        world_size=placement.world_size,
        stage_id=placement.stage_id,
        stage_count=placement.stage_count,
        num_logical_layers=placement.num_logical_layers,
        bindings=bindings,
        local_group_ids=local_group_ids,
        field_owners=field_owners,
        global_group_packing=global_group_packing,
        pipeline_plan_digest=placement.pipeline_plan_digest,
        layout=layout,
    )
    return StageCacheLayout(
        layout=layout,
        logical_fields=projected_fields,
        bindings=bindings,
        local_group_ids=local_group_ids,
        global_group_packing=global_group_packing,
        manifest=manifest,
    )
