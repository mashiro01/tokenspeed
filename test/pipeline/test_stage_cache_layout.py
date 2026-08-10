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

import hashlib
import json
import unittest

from tokenspeed.runtime.layers.attention.kv_cache.recipes.kimi_k3 import (
    kimi_k3_cache_fields,
    kimi_k3_pipeline_workspace_bytes,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    solve_cache_layout,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.stage_layout import (
    CacheLayerBinding,
    CacheStagePlacement,
    LogicalCacheFieldSpec,
    pipeline_cache_abi_digest,
    solve_stage_cache_layout,
)
from tokenspeed.runtime.pipeline import (
    ActivationFieldSpec,
    ActivationSchema,
    PipelinePlan,
    StagePlan,
)
from tokenspeed.runtime.pipeline.adapters.kimi_k3 import (
    build_balanced_kimi_k3_pipeline_plan,
)


def _logical_field(
    logical_layer_id: int,
    group_id: str,
    field_id: str,
    plane_id: str,
) -> LogicalCacheFieldSpec:
    return LogicalCacheFieldSpec(
        logical_layer_id=logical_layer_id,
        field=CacheFieldSpec(
            group_id=group_id,
            field_id=field_id,
            plane_id=plane_id,
            shape=(8,),
            element_size=2,
        ),
    )


def _placement(
    rank: int,
    stage_id: int,
    stage_count: int,
    num_logical_layers: int,
    logical_layer_ids_by_stage: tuple[tuple[int, ...], ...],
    *,
    world_size: int | None = None,
    pipeline_plan_digest: str | None = None,
) -> CacheStagePlacement:
    return CacheStagePlacement(
        rank=rank,
        world_size=world_size or stage_count * 8,
        stage_id=stage_id,
        stage_count=stage_count,
        num_logical_layers=num_logical_layers,
        logical_layer_ids_by_stage=logical_layer_ids_by_stage,
        pipeline_plan_digest=pipeline_plan_digest or "0" * 64,
    )


class StageCacheLayoutTest(unittest.TestCase):
    def test_k3_pp8_workspace_reserves_receive_and_attnres_slabs(self):
        plan = build_balanced_kimi_k3_pipeline_plan(
            num_layers=93,
            hidden_size=7168,
            attn_res_block_size=12,
            stage_count=8,
        )

        stage0 = kimi_k3_pipeline_workspace_bytes(
            pipeline_plan=plan,
            stage_id=0,
            num_layers=93,
            attn_res_block_size=12,
            hidden_size=7168,
            max_step_tokens=8192,
            activation_element_size=2,
        )
        stage7 = kimi_k3_pipeline_workspace_bytes(
            pipeline_plan=plan,
            stage_id=7,
            num_layers=93,
            attn_res_block_size=12,
            hidden_size=7168,
            max_step_tokens=8192,
            activation_element_size=2,
        )

        per_tensor = 8192 * 7168 * 2
        self.assertEqual(stage0, 8 * per_tensor)
        self.assertEqual(stage7, 16 * per_tensor)

    def test_default_k3_pp8_compacts_stage_planes_with_global_packing(self):
        full_layers = set(range(3, 92, 4)) | {92}
        kda_layers = [layer for layer in range(93) if layer not in full_layers]
        kda_position = {layer: index for index, layer in enumerate(kda_layers)}
        group_ids = tuple(
            (
                "full_attention"
                if layer in full_layers
                else f"linear_attention_{kda_position[layer] // 23}"
            )
            for layer in range(93)
        )
        fields = kimi_k3_cache_fields(
            layer_group_ids=group_ids,
            logical_block_tokens=128,
            latent_width=576,
            mla_element_size=1,
            conv_shape=(4608, 3),
            conv_element_size=2,
            recurrent_shape=(12, 128, 128),
            recurrent_element_size=4,
        )
        logical_fields = []
        cursor = 0
        for layer_id, group_id in enumerate(group_ids):
            field_count = 1 if group_id == "full_attention" else 2
            logical_fields.extend(
                LogicalCacheFieldSpec(layer_id, field)
                for field in fields[cursor : cursor + field_count]
            )
            cursor += field_count

        packing = {
            "full_attention": 12,
            "linear_attention_0": 1,
            "linear_attention_1": 1,
            "linear_attention_2": 1,
        }
        global_layout = solve_cache_layout(
            fields,
            logical_block_tokens=128,
            cache_blocks_per_lcm_block=packing,
            alignment=256,
        )
        plan = build_balanced_kimi_k3_pipeline_plan(
            num_layers=93,
            hidden_size=7168,
            attn_res_block_size=12,
            stage_count=8,
        )

        projected = []
        for stage_id in range(8):
            projected.append(
                solve_stage_cache_layout(
                    logical_fields,
                    CacheStagePlacement.from_pipeline_plan(
                        rank=stage_id * 8,
                        world_size=64,
                        plan=plan,
                    ),
                    logical_block_tokens=128,
                    cache_blocks_per_lcm_block=dict(global_layout.group_packing),
                    alignment=256,
                    max_padding_fraction=float("inf"),
                    compact_group_planes=True,
                )
            )

        self.assertEqual(
            tuple(len(stage.bindings) for stage in projected),
            (12, 12, 12, 12, 12, 12, 12, 9),
        )
        self.assertEqual(
            tuple(stage.layout.lcm_block_bytes for stage in projected),
            (
                7_538_688,
                7_538_688,
                4_282_368,
                7_538_688,
                7_538_688,
                6_724_608,
                7_538_688,
                5_096_448,
            ),
        )
        self.assertEqual(
            tuple(len(stage.layout.plane_bytes) for stage in projected),
            (9, 9, 5, 9, 9, 8, 9, 6),
        )
        for stage in projected:
            self.assertTrue(
                set(stage.layout.group_packing).issubset(
                    set(global_layout.group_packing)
                )
            )
            self.assertTrue(
                all(
                    field.plane_id.startswith("stage-slot.")
                    for field in stage.layout.fields
                )
            )

    def test_single_stage_is_identical_to_direct_solver(self):
        logical_fields = (
            _logical_field(2, "history", "opaque-c", "plane/second"),
            _logical_field(0, "history", "opaque-a", "plane/first"),
            _logical_field(1, "state", "opaque-b", "plane/first"),
        )
        packing = {"history": 1, "state": 1}

        projected = solve_stage_cache_layout(
            logical_fields,
            _placement(
                rank=0,
                stage_id=0,
                stage_count=1,
                num_logical_layers=3,
                logical_layer_ids_by_stage=((0, 1, 2),),
                world_size=1,
            ),
            logical_block_tokens=128,
            cache_blocks_per_lcm_block=packing,
            alignment=16,
            max_padding_fraction=float("inf"),
        )
        direct = solve_cache_layout(
            (logical_field.field for logical_field in logical_fields),
            logical_block_tokens=128,
            cache_blocks_per_lcm_block=packing,
            alignment=16,
            max_padding_fraction=float("inf"),
        )

        self.assertEqual(projected.layout, direct)
        self.assertEqual(
            projected.bindings,
            (
                CacheLayerBinding(0, 0),
                CacheLayerBinding(1, 1),
                CacheLayerBinding(2, 2),
            ),
        )
        self.assertEqual(projected.logical_to_physical, {0: 0, 1: 1, 2: 2})

    def test_projects_opaque_fields_and_filters_nonlocal_group_packing(self):
        logical_fields = (
            _logical_field(0, "history", "field[zero]", "opaque:plane-a"),
            _logical_field(1, "state", "field[one]", "opaque:plane-b"),
            _logical_field(2, "history", "field[two]", "opaque:plane-c"),
            _logical_field(3, "remote", "field[three]", "opaque:plane-d"),
        )
        projected = solve_stage_cache_layout(
            logical_fields,
            _placement(
                rank=7,
                stage_id=0,
                stage_count=2,
                num_logical_layers=4,
                logical_layer_ids_by_stage=((0, 1), (2, 3)),
            ),
            logical_block_tokens=128,
            cache_blocks_per_lcm_block={
                "history": 2,
                "state": 4,
                "remote": 8,
            },
            max_padding_fraction=float("inf"),
        )

        self.assertEqual(projected.logical_to_physical, {0: 0, 1: 1})
        self.assertEqual(projected.local_group_ids, ("history", "state"))
        self.assertEqual(
            dict(projected.layout.group_packing), {"history": 2, "state": 4}
        )
        self.assertEqual(
            {field.field_id for field in projected.layout.fields},
            {"field[zero]", "field[one]"},
        )
        self.assertEqual(
            {field.plane_id for field in projected.layout.fields},
            {"opaque:plane-a", "opaque:plane-b"},
        )
        self.assertEqual(projected.manifest.rank, 7)

    def test_cacheless_owned_layer_still_gets_a_dense_binding(self):
        projected = solve_stage_cache_layout(
            (
                _logical_field(0, "history", "field-a", "plane-a"),
                _logical_field(2, "history", "field-c", "plane-c"),
                _logical_field(3, "remote", "field-d", "plane-d"),
            ),
            _placement(
                rank=0,
                stage_id=0,
                stage_count=2,
                num_logical_layers=4,
                logical_layer_ids_by_stage=((0, 1, 2), (3,)),
            ),
            logical_block_tokens=128,
            max_padding_fraction=float("inf"),
        )

        self.assertEqual(projected.logical_to_physical, {0: 0, 1: 1, 2: 2})

    def test_manifest_digest_is_canonical_and_deterministic(self):
        fields = (
            _logical_field(0, "history", "field-a", "plane-a"),
            _logical_field(1, "remote", "field-b", "plane-b"),
            _logical_field(2, "history", "field-c", "plane-c"),
            _logical_field(3, "remote", "field-d", "plane-d"),
        )

        def solve(logical_fields, local_ids, remote_ids):
            return solve_stage_cache_layout(
                logical_fields,
                _placement(
                    rank=4,
                    stage_id=0,
                    stage_count=2,
                    num_logical_layers=4,
                    logical_layer_ids_by_stage=(local_ids, remote_ids),
                ),
                logical_block_tokens=64,
                cache_blocks_per_lcm_block={"history": 1, "remote": 3},
                max_padding_fraction=float("inf"),
            )

        first = solve(fields, (0, 1), (2, 3))
        second = solve(tuple(reversed(fields)), (0, 1), (2, 3))
        canonical_json = first.manifest.canonical_json()

        self.assertEqual(first.manifest.digest, second.manifest.digest)
        self.assertEqual(canonical_json, second.manifest.canonical_json())
        self.assertEqual(
            first.manifest.digest,
            hashlib.sha256(canonical_json.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(len(first.manifest.digest), 64)
        self.assertEqual(json.loads(canonical_json)["version"], 1)
        self.assertNotIn(": ", canonical_json)

        same_stage_other_rank = solve_stage_cache_layout(
            fields,
            _placement(7, 0, 2, 4, ((0, 1), (2, 3))),
            logical_block_tokens=64,
            cache_blocks_per_lcm_block={"history": 1, "remote": 3},
            max_padding_fraction=float("inf"),
        )
        self.assertNotEqual(
            first.manifest.digest, same_stage_other_rank.manifest.digest
        )
        self.assertEqual(
            first.manifest.stage_digest,
            same_stage_other_rank.manifest.stage_digest,
        )

    def test_pipeline_cache_abi_digest_is_global_and_rank_independent(self):
        fields = (
            _logical_field(0, "history", "field-a", "plane-a"),
            _logical_field(1, "remote", "field-b", "plane-b"),
        )
        layout = solve_cache_layout(
            (field.field for field in fields),
            logical_block_tokens=128,
            cache_blocks_per_lcm_block={"history": 1, "remote": 2},
            max_padding_fraction=float("inf"),
        )
        placements = (
            _placement(0, 0, 2, 2, ((0,), (1,))),
            _placement(8, 1, 2, 2, ((0,), (1,))),
        )

        digests = [
            pipeline_cache_abi_digest(fields, placement, layout)
            for placement in placements
        ]
        self.assertEqual(digests[0], digests[1])
        self.assertEqual(len(digests[0]), 64)
        changed_plan = _placement(
            0,
            0,
            2,
            2,
            ((0,), (1,)),
            pipeline_plan_digest="1" * 64,
        )
        self.assertNotEqual(
            digests[0], pipeline_cache_abi_digest(fields, changed_plan, layout)
        )
        bf16 = pipeline_cache_abi_digest(
            fields,
            placements[0],
            layout,
            field_dtypes={"field-a": "bfloat16", "field-b": "float32"},
        )
        fp16 = pipeline_cache_abi_digest(
            fields,
            placements[0],
            layout,
            field_dtypes={"field-a": "float16", "field-b": "float32"},
        )
        self.assertNotEqual(bf16, fp16)
        with self.assertRaisesRegex(ValueError, "exactly cover"):
            pipeline_cache_abi_digest(
                fields,
                placements[0],
                layout,
                field_dtypes={"field-a": "bfloat16"},
            )

    def test_rejects_invalid_ownership(self):
        fields = (_logical_field(0, "history", "field-a", "plane-a"),)
        invalid_placements = (
            (
                "duplicate ownership",
                _placement(0, 0, 2, 3, ((0, 1), (1, 2))),
            ),
            (
                "omitted ownership",
                _placement(0, 0, 2, 3, ((0,), (2,))),
            ),
            (
                "out of range",
                _placement(0, 0, 2, 3, ((0, 3), (1, 2))),
            ),
            (
                "stage_id is out of range",
                _placement(0, 2, 2, 3, ((0, 1), (2,))),
            ),
            (
                "one entry per stage",
                _placement(0, 0, 2, 3, ((0, 1, 2),)),
            ),
        )

        for message, placement in invalid_placements:
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError, message
            ):
                solve_stage_cache_layout(
                    fields,
                    placement,
                    logical_block_tokens=128,
                )

    def test_rejects_unknown_field_ownership(self):
        with self.assertRaisesRegex(ValueError, "unknown ownership"):
            solve_stage_cache_layout(
                (_logical_field(3, "history", "field-a", "plane-a"),),
                _placement(0, 0, 1, 3, ((0, 1, 2),), world_size=1),
                logical_block_tokens=128,
            )

    def test_rejects_duplicate_field_ids_before_projection(self):
        fields = (
            _logical_field(0, "history", "duplicate", "plane-a"),
            _logical_field(1, "remote", "duplicate", "plane-b"),
        )
        with self.assertRaisesRegex(ValueError, "duplicate logical cache field"):
            solve_stage_cache_layout(
                fields,
                _placement(0, 0, 2, 2, ((0,), (1,))),
                logical_block_tokens=128,
            )

    def test_rejects_unknown_packing_group(self):
        with self.assertRaisesRegex(ValueError, "unknown groups"):
            solve_stage_cache_layout(
                (_logical_field(0, "history", "field-a", "plane-a"),),
                _placement(0, 0, 1, 1, ((0,),), world_size=1),
                logical_block_tokens=128,
                cache_blocks_per_lcm_block={"unknown": 1},
            )

    def test_rejects_invalid_packing_for_a_nonlocal_group(self):
        fields = (
            _logical_field(0, "history", "field-a", "plane-a"),
            _logical_field(1, "remote", "field-b", "plane-b"),
        )
        with self.assertRaisesRegex(ValueError, "positive integer"):
            solve_stage_cache_layout(
                fields,
                _placement(0, 0, 2, 2, ((0,), (1,))),
                logical_block_tokens=128,
                cache_blocks_per_lcm_block={"history": 1, "remote": 0},
            )

    def test_requires_global_and_local_fields(self):
        placement = _placement(0, 0, 1, 1, ((0,),), world_size=1)
        with self.assertRaisesRegex(ValueError, "at least one logical cache field"):
            solve_stage_cache_layout((), placement, logical_block_tokens=128)

        with self.assertRaisesRegex(ValueError, "selected stage"):
            solve_stage_cache_layout(
                (_logical_field(0, "history", "field-a", "plane-a"),),
                _placement(8, 1, 2, 2, ((0,), (1,))),
                logical_block_tokens=128,
            )

    def test_derives_rank_and_layer_ownership_from_pipeline_plan(self):
        schema = ActivationSchema(
            "boundary", (ActivationFieldSpec("hidden", "bfloat16", (8,)),)
        )
        plan = PipelinePlan(
            (
                StagePlan(0, 2, 0, 2, True, False, output_schema=schema),
                StagePlan(1, 2, 2, 4, False, True, input_schema=schema),
            )
        )

        placement = CacheStagePlacement.from_pipeline_plan(
            rank=10,
            world_size=16,
            plan=plan,
        )

        self.assertEqual(placement.stage_id, 1)
        self.assertEqual(placement.logical_layer_ids_by_stage, ((0, 1), (2, 3)))
        self.assertEqual(placement.pipeline_plan_digest, plan.digest)

    def test_rejects_rank_stage_mismatch_and_noncontiguous_plan(self):
        field = (_logical_field(0, "history", "field-a", "plane-a"),)
        with self.assertRaisesRegex(ValueError, "belongs to stage"):
            solve_stage_cache_layout(
                field,
                _placement(8, 0, 2, 2, ((0,), (1,))),
                logical_block_tokens=128,
            )
        with self.assertRaisesRegex(ValueError, "ordered and contiguous"):
            solve_stage_cache_layout(
                field,
                _placement(0, 0, 2, 2, ((1,), (0,))),
                logical_block_tokens=128,
            )

    def test_shared_group_packing_and_page_count_are_identical_across_stages(self):
        fields = (
            LogicalCacheFieldSpec(0, CacheFieldSpec("A", "a0", "shared", (8,), 1)),
            LogicalCacheFieldSpec(0, CacheFieldSpec("B", "b0", "shared", (1,), 1)),
            LogicalCacheFieldSpec(1, CacheFieldSpec("A", "a1", "stage-1", (8,), 1)),
            LogicalCacheFieldSpec(1, CacheFieldSpec("B", "b1", "stage-1", (1,), 1)),
        )
        layouts = [
            solve_stage_cache_layout(
                fields,
                _placement(stage_id * 8, stage_id, 2, 2, ((0,), (1,))),
                logical_block_tokens=128,
                max_padding_fraction=float("inf"),
            )
            for stage_id in range(2)
        ]

        self.assertEqual(
            layouts[0].global_group_packing, layouts[1].global_group_packing
        )
        for group_id in ("A", "B"):
            counts = [
                layout.layout.with_num_lcm_blocks(4).group(group_id).page_count
                for layout in layouts
            ]
            self.assertEqual(counts[0], counts[1])


if __name__ == "__main__":
    unittest.main()
