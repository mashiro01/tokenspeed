import pytest

from tokenspeed.runtime.distributed.mapping import (
    AttentionLayerMapping,
    DenseLayerMapping,
    Mapping,
    MappingBase,
    MoeLayerMapping,
    PipelineMapping,
    _make_parallelism_group,
    _make_parallelism_rank,
    _resolve_parallelism_sizes,
)

# =============================================================================
# _resolve_parallelism_sizes
# =============================================================================


class TestResolveParallelismSizes:

    def test_all_provided(self):
        assert _resolve_parallelism_sizes(8, 4, 2) == (4, 2)

    def test_all_provided_three_dims(self):
        assert _resolve_parallelism_sizes(24, 2, 3, 4) == (2, 3, 4)

    def test_infer_last(self):
        assert _resolve_parallelism_sizes(8, 4, None) == (4, 2)

    def test_infer_first(self):
        assert _resolve_parallelism_sizes(8, None, 2) == (4, 2)

    def test_infer_middle(self):
        assert _resolve_parallelism_sizes(24, 2, None, 4) == (2, 3, 4)

    def test_all_none_two_dims(self):
        # First None gets world_size, rest get 1
        assert _resolve_parallelism_sizes(8, None, None) == (8, 1)

    def test_all_none_three_dims(self):
        assert _resolve_parallelism_sizes(8, None, None, None) == (8, 1, 1)

    def test_world_size_one(self):
        assert _resolve_parallelism_sizes(1, None, None) == (1, 1)
        assert _resolve_parallelism_sizes(1, 1, 1) == (1, 1)

    def test_product_mismatch_raises(self):
        with pytest.raises(AssertionError):
            _resolve_parallelism_sizes(8, 4, 4)

    def test_indivisible_raises(self):
        with pytest.raises(AssertionError):
            _resolve_parallelism_sizes(8, 3, None)

    def test_provided_exceeds_world_size_raises(self):
        with pytest.raises(AssertionError):
            _resolve_parallelism_sizes(8, 16, None)

    def test_zero_size_raises(self):
        with pytest.raises(AssertionError):
            _resolve_parallelism_sizes(8, 0, None)


# =============================================================================
# _make_parallelism_rank
# =============================================================================


class TestMakeParallelismRank:

    def test_basic(self):
        # rank=5 in [tp=4, dp=2]: tp_rank=1, dp_rank=1
        assert _make_parallelism_rank(5, size=4, stride=1) == 1
        assert _make_parallelism_rank(5, size=2, stride=4) == 1

    def test_wraps_with_global_rank(self):
        # rank=10, tp=4: tp_rank = 10 % 4 = 2
        assert _make_parallelism_rank(10, size=4, stride=1) == 2
        # dp_rank = (10 // 4) % 2 = 0
        assert _make_parallelism_rank(10, size=2, stride=4) == 0

    def test_consistent_with_group(self):
        """rank should be at position rank_in_dim within its group."""
        for rank in range(24):
            for size, stride in [(2, 1), (3, 2), (4, 6)]:
                r = _make_parallelism_rank(rank, size, stride)
                g = _make_parallelism_group(rank, size, stride)
                assert g[r] == rank


# =============================================================================
# _make_parallelism_group
# =============================================================================


class TestMakeParallelismGroup:

    def test_stride_1(self):
        # TP group for rank 5 in [tp=4, dp=2]
        assert _make_parallelism_group(5, size=4, stride=1) == (4, 5, 6, 7)

    def test_stride_gt_1(self):
        # DP group for rank 5 in [tp=4, dp=2]
        assert _make_parallelism_group(5, size=2, stride=4) == (1, 5)

    def test_size_1(self):
        assert _make_parallelism_group(3, size=1, stride=1) == (3,)

    def test_rank_0(self):
        assert _make_parallelism_group(0, size=4, stride=1) == (0, 1, 2, 3)

    def test_rank_always_in_group(self):
        for rank in range(24):
            for size, stride in [(2, 1), (3, 2), (4, 6)]:
                group = _make_parallelism_group(rank, size, stride)
                assert rank in group


# =============================================================================
# MappingBase
# =============================================================================


class TestMappingBase:

    def test_immediate_rank(self):
        m = MappingBase(rank=3, world_size=8)
        assert m.rank == 3
        assert m.world_size == 8

    def test_deferred_rank(self):
        m = MappingBase(world_size=8)
        m.rank = 5
        assert m.rank == 5

    def test_access_before_set_raises(self):
        m = MappingBase(world_size=8)
        with pytest.raises(AssertionError, match="rank is not initialized"):
            _ = m.rank

    def test_set_once_raises(self):
        m = MappingBase(rank=3, world_size=8)
        with pytest.raises(AssertionError, match="rank is already initialized"):
            m.rank = 5

    def test_deferred_set_once_raises(self):
        m = MappingBase(world_size=8)
        m.rank = 3
        with pytest.raises(AssertionError, match="rank is already initialized"):
            m.rank = 5

    def test_negative_rank_raises(self):
        with pytest.raises(AssertionError):
            MappingBase(rank=-1, world_size=8)

    def test_deferred_negative_rank_raises(self):
        m = MappingBase(world_size=8)
        with pytest.raises(AssertionError):
            m.rank = -1

    def test_zero_world_size_raises(self):
        with pytest.raises(AssertionError):
            MappingBase(world_size=0)


# =============================================================================
# PipelineMapping
# =============================================================================


class TestPipelineMapping:

    def test_pp1(self):
        m = PipelineMapping(rank=3, world_size=8)

        assert m.world_size == 8
        assert m.world_group == tuple(range(8))
        assert m.stage_count == 1
        assert m.stage_index == 0
        assert m.stage_world_size == 8
        assert m.stage_local_rank == 3
        assert m.stage_group == tuple(range(8))
        assert m.pipeline_group == (3,)
        assert m.prev_rank is None
        assert m.next_rank is None
        assert m.is_first_stage
        assert m.is_last_stage

    def test_pp2_stage_contiguous_groups(self):
        mappings = [
            PipelineMapping(rank=rank, world_size=8, stage_count=2) for rank in range(8)
        ]

        assert {m.stage_group for m in mappings} == {
            (0, 1, 2, 3),
            (4, 5, 6, 7),
        }
        assert {m.pipeline_group for m in mappings} == {
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        }

        first = mappings[2]
        assert first.stage_index == 0
        assert first.stage_local_rank == 2
        assert first.prev_rank is None
        assert first.next_rank == 6
        assert first.is_first_stage
        assert not first.is_last_stage

        last = mappings[6]
        assert last.stage_index == 1
        assert last.stage_local_rank == 2
        assert last.prev_rank == 2
        assert last.next_rank is None
        assert not last.is_first_stage
        assert last.is_last_stage

    def test_pp4_groups_and_neighbors(self):
        m = PipelineMapping(rank=10, world_size=16, stage_count=4)

        assert m.stage_index == 2
        assert m.stage_world_size == 4
        assert m.stage_local_rank == 2
        assert m.stage_group == (8, 9, 10, 11)
        assert m.pipeline_group == (2, 6, 10, 14)
        assert m.prev_rank == 6
        assert m.next_rank == 14
        assert not m.is_first_stage
        assert not m.is_last_stage

    def test_deferred_rank(self):
        m = PipelineMapping(world_size=8, stage_count=2)

        assert m.stage_count == 2
        assert m.stage_world_size == 4
        with pytest.raises(AssertionError, match="rank is not initialized"):
            _ = m.stage_index

        m.rank = 5
        assert m.rank == 5
        assert m.stage_index == 1
        assert m.stage_local_rank == 1
        assert m.stage_group == (4, 5, 6, 7)
        assert m.pipeline_group == (1, 5)

    @pytest.mark.parametrize("value", [True, False, 1.0, "1", None])
    def test_invalid_stage_count_type(self, value):
        with pytest.raises(TypeError, match="stage_count must be an int"):
            PipelineMapping(world_size=8, stage_count=value)

    @pytest.mark.parametrize("value", [True, False, 8.0, "8", None])
    def test_invalid_world_size_type(self, value):
        with pytest.raises(TypeError, match="world_size must be an int"):
            PipelineMapping(world_size=value)

    @pytest.mark.parametrize("value", [0, -1])
    def test_non_positive_stage_count(self, value):
        with pytest.raises(ValueError, match="stage_count must be positive"):
            PipelineMapping(world_size=8, stage_count=value)

    def test_indivisible_world_size(self):
        with pytest.raises(ValueError, match="must be divisible"):
            PipelineMapping(world_size=8, stage_count=3)

    @pytest.mark.parametrize("rank", [True, 1.0, "1"])
    def test_invalid_rank_type(self, rank):
        with pytest.raises(TypeError, match="rank must be an int"):
            PipelineMapping(rank=rank, world_size=8, stage_count=2)

    @pytest.mark.parametrize("rank", [-1, 8])
    def test_rank_out_of_global_world(self, rank):
        with pytest.raises(ValueError, match=r"rank must be in \[0, 8\)"):
            PipelineMapping(rank=rank, world_size=8, stage_count=2)

    def test_deferred_rank_is_strict(self):
        m = PipelineMapping(world_size=8, stage_count=2)
        with pytest.raises(TypeError, match="rank must be an int"):
            m.rank = True


# =============================================================================
# DenseLayerMapping
# =============================================================================


class TestDenseLayerMapping:

    def test_tp_only(self):
        m = DenseLayerMapping(rank=3, world_size=8, tp_size=8)
        assert m.tp_size == 8
        assert m.dp_size == 1
        assert m.tp_rank == 3
        assert m.dp_rank == 0
        assert m.tp_group == tuple(range(8))
        assert m.dp_group == (3,)

    def test_dp_only(self):
        m = DenseLayerMapping(rank=3, world_size=8, tp_size=1)
        assert m.tp_size == 1
        assert m.dp_size == 8
        assert m.tp_rank == 0
        assert m.dp_rank == 3
        assert m.tp_group == (3,)
        assert m.dp_group == tuple(range(8))

    def test_combined(self):
        # ws=8, tp=4, dp=2: ranks [0..3] are dp0, [4..7] are dp1
        m = DenseLayerMapping(rank=5, world_size=8, tp_size=4)
        assert m.tp_size == 4
        assert m.dp_size == 2
        assert m.tp_rank == 1
        assert m.dp_rank == 1
        assert m.tp_group == (4, 5, 6, 7)
        assert m.dp_group == (1, 5)

    def test_infer_dp(self):
        m = DenseLayerMapping(rank=0, world_size=8, tp_size=4)
        assert m.dp_size == 2

    def test_infer_tp(self):
        m = DenseLayerMapping(rank=0, world_size=8, dp_size=2)
        assert m.tp_size == 4

    def test_deferred_rank(self):
        m = DenseLayerMapping(world_size=8, tp_size=4)
        assert m.tp_size == 4
        assert m.dp_size == 2
        m.rank = 5
        assert m.tp_rank == 1
        assert m.dp_rank == 1
        assert m.tp_group == (4, 5, 6, 7)
        assert m.dp_group == (1, 5)

    def test_deferred_rank_access_before_set_raises(self):
        m = DenseLayerMapping(world_size=8, tp_size=4)
        with pytest.raises(AssertionError):
            _ = m.tp_rank

    def test_groups_partition_world(self):
        """Every rank's TP group should partition the world into dp_size groups,
        and every rank's DP group should partition the world into tp_size groups."""
        ws = 8
        tp = 4
        mappings = [
            DenseLayerMapping(rank=r, world_size=ws, tp_size=tp) for r in range(ws)
        ]

        tp_groups = set()
        dp_groups = set()
        for m in mappings:
            tp_groups.add(tuple(m.tp_group))
            dp_groups.add(tuple(m.dp_group))

        # Should have dp_size distinct TP groups and tp_size distinct DP groups
        assert len(tp_groups) == 2  # dp_size
        assert len(dp_groups) == 4  # tp_size

        # TP groups should be disjoint and cover all ranks
        all_tp_ranks = sorted(r for g in tp_groups for r in g)
        assert all_tp_ranks == list(range(ws))

        # DP groups should be disjoint and cover all ranks
        all_dp_ranks = sorted(r for g in dp_groups for r in g)
        assert all_dp_ranks == list(range(ws))

    def test_global_rank_beyond_world_size(self):
        """Layer mappings accept global ranks beyond the local mapping size."""
        m = DenseLayerMapping(rank=10, world_size=8, tp_size=4)
        assert m.tp_rank == 2  # 10 % 4
        assert m.dp_rank == 0  # (10 // 4) % 2
        assert m.tp_group == (8, 9, 10, 11)
        assert m.dp_group == (10, 14)

    def test_invalid_rank_raises(self):
        with pytest.raises(AssertionError):
            DenseLayerMapping(rank=-1, world_size=8, tp_size=4)

    def test_invalid_world_size_raises(self):
        with pytest.raises(AssertionError):
            DenseLayerMapping(rank=0, world_size=0, tp_size=1)


# =============================================================================
# AttentionLayerMapping
# =============================================================================


class TestAttentionLayerMapping:

    def test_tp_only(self):
        m = AttentionLayerMapping(rank=2, world_size=8, tp_size=8, cp_size=1, dp_size=1)
        assert m.tp_rank == 2
        assert m.cp_rank == 0
        assert m.dp_rank == 0
        assert m.tp_group == tuple(range(8))
        assert m.cp_group == (2,)
        assert m.dp_group == (2,)

    def test_tp_cp(self):
        # ws=8, tp=2, cp=4, dp=1
        # rank layout: rank = dp_rank*(tp*cp) + cp_rank*tp + tp_rank
        m = AttentionLayerMapping(rank=5, world_size=8, tp_size=2, cp_size=4, dp_size=1)
        assert m.tp_rank == 1  # 5 % 2
        assert m.cp_rank == 2  # (5 // 2) % 4
        assert m.dp_rank == 0  # 5 // 8
        assert m.tp_group == (4, 5)
        assert m.cp_group == (1, 3, 5, 7)

    def test_tp_cp_dp(self):
        # ws=16, tp=2, cp=2, dp=4
        m = AttentionLayerMapping(
            rank=7, world_size=16, tp_size=2, cp_size=2, dp_size=4
        )
        assert m.tp_rank == 1  # 7 % 2
        assert m.cp_rank == 1  # (7 // 2) % 2
        assert m.dp_rank == 1  # 7 // 4
        assert m.tp_group == (6, 7)
        assert m.cp_group == (5, 7)
        assert m.dp_group == (3, 7, 11, 15)

    def test_infer_cp(self):
        m = AttentionLayerMapping(rank=0, world_size=16, tp_size=2, dp_size=4)
        assert m.cp_size == 2

    def test_infer_dp(self):
        m = AttentionLayerMapping(rank=0, world_size=16, tp_size=2, cp_size=4)
        assert m.dp_size == 2

    def test_deferred_rank(self):
        m = AttentionLayerMapping(world_size=16, tp_size=2, cp_size=2, dp_size=4)
        assert m.tp_size == 2
        assert m.cp_size == 2
        assert m.dp_size == 4
        m.rank = 7
        assert m.tp_rank == 1
        assert m.cp_rank == 1
        assert m.dp_rank == 1
        assert m.tp_group == (6, 7)
        assert m.cp_group == (5, 7)
        assert m.dp_group == (3, 7, 11, 15)

    def test_cp_size_1_matches_dense(self):
        """With cp_size=1, AttentionLayerMapping should produce the same
        tp/dp ranks and groups as DenseLayerMapping."""
        ws = 8
        tp = 4
        for r in range(ws):
            attn = AttentionLayerMapping(rank=r, world_size=ws, tp_size=tp, cp_size=1)
            dense = DenseLayerMapping(rank=r, world_size=ws, tp_size=tp)
            assert attn.tp_rank == dense.tp_rank
            assert attn.dp_rank == dense.dp_rank
            assert attn.tp_group == dense.tp_group
            assert attn.dp_group == dense.dp_group

    def test_groups_partition_world(self):
        """All three group types should partition the world correctly."""
        ws = 24
        tp, cp, dp = 2, 3, 4
        mappings = [
            AttentionLayerMapping(
                rank=r, world_size=ws, tp_size=tp, cp_size=cp, dp_size=dp
            )
            for r in range(ws)
        ]

        for attr, expected_count in [
            ("tp_group", ws // tp),
            ("cp_group", ws // cp),
            ("dp_group", ws // dp),
        ]:
            groups = set()
            for m in mappings:
                groups.add(tuple(getattr(m, attr)))
            assert (
                len(groups) == expected_count
            ), f"{attr}: expected {expected_count} groups, got {len(groups)}"
            all_ranks = sorted(r for g in groups for r in g)
            assert all_ranks == list(range(ws)), f"{attr}: groups don't cover all ranks"

    def test_all_ranks_consistent(self):
        """For every rank, rank == dp_rank * (tp*cp) + cp_rank * tp + tp_rank."""
        ws = 24
        tp, cp, dp = 2, 3, 4
        for r in range(ws):
            m = AttentionLayerMapping(
                rank=r, world_size=ws, tp_size=tp, cp_size=cp, dp_size=dp
            )
            reconstructed = m.dp_rank * (tp * cp) + m.cp_rank * tp + m.tp_rank
            assert reconstructed == r, f"rank={r}: reconstructed={reconstructed}"

    def test_invalid_product_raises(self):
        with pytest.raises(AssertionError):
            AttentionLayerMapping(
                rank=0, world_size=16, tp_size=2, cp_size=3, dp_size=4
            )


# =============================================================================
# MoeLayerMapping
# =============================================================================


class TestMoeLayerMapping:

    def test_tp_only(self):
        m = MoeLayerMapping(rank=3, world_size=8, tp_size=8, ep_size=1, dp_size=1)
        assert m.tp_rank == 3
        assert m.ep_rank == 0
        assert m.dp_rank == 0
        assert m.tp_group == tuple(range(8))
        assert m.ep_group == (3,)
        assert m.dp_group == (3,)

    def test_ep_only(self):
        m = MoeLayerMapping(rank=3, world_size=8, tp_size=1, ep_size=8, dp_size=1)
        assert m.tp_rank == 0
        assert m.ep_rank == 3
        assert m.dp_rank == 0
        assert m.tp_group == (3,)
        assert m.ep_group == tuple(range(8))
        assert m.dp_group == (3,)

    def test_tp_ep(self):
        # ws=8, tp=2, ep=4, dp=1
        m = MoeLayerMapping(rank=5, world_size=8, tp_size=2, ep_size=4, dp_size=1)
        assert m.tp_rank == 1  # 5 % 2
        assert m.ep_rank == 2  # (5 // 2) % 4
        assert m.dp_rank == 0  # 5 // 8
        assert m.tp_group == (4, 5)
        assert m.ep_group == (1, 3, 5, 7)

    def test_tp_ep_dp(self):
        # ws=16, tp=2, ep=2, dp=4
        m = MoeLayerMapping(rank=7, world_size=16, tp_size=2, ep_size=2, dp_size=4)
        assert m.tp_rank == 1  # 7 % 2
        assert m.ep_rank == 1  # (7 // 2) % 2
        assert m.dp_rank == 1  # 7 // 4
        assert m.tp_group == (6, 7)
        assert m.ep_group == (5, 7)
        assert m.dp_group == (3, 7, 11, 15)

    def test_infer_ep(self):
        m = MoeLayerMapping(rank=0, world_size=16, tp_size=2, dp_size=4)
        assert m.ep_size == 2

    def test_infer_dp(self):
        m = MoeLayerMapping(rank=0, world_size=16, tp_size=2, ep_size=4)
        assert m.dp_size == 2

    def test_deferred_rank(self):
        m = MoeLayerMapping(world_size=16, tp_size=2, ep_size=2, dp_size=4)
        assert m.tp_size == 2
        assert m.ep_size == 2
        assert m.dp_size == 4
        m.rank = 7
        assert m.tp_rank == 1
        assert m.ep_rank == 1
        assert m.dp_rank == 1
        assert m.tp_group == (6, 7)
        assert m.ep_group == (5, 7)
        assert m.dp_group == (3, 7, 11, 15)

    def test_ep_size_1_matches_dense(self):
        """With ep_size=1, MoeLayerMapping should produce the same
        tp/dp ranks and groups as DenseLayerMapping."""
        ws = 8
        tp = 4
        for r in range(ws):
            moe = MoeLayerMapping(rank=r, world_size=ws, tp_size=tp, ep_size=1)
            dense = DenseLayerMapping(rank=r, world_size=ws, tp_size=tp)
            assert moe.tp_rank == dense.tp_rank
            assert moe.dp_rank == dense.dp_rank
            assert moe.tp_group == dense.tp_group
            assert moe.dp_group == dense.dp_group

    def test_structure_mirrors_attention(self):
        """MoeLayerMapping(tp, ep, dp) should have the same rank/group structure
        as AttentionLayerMapping(tp, cp, dp) when sizes match, since both are
        3-dim inner-to-outer layouts."""
        ws = 24
        tp, middle, dp = 2, 3, 4
        for r in range(ws):
            attn = AttentionLayerMapping(
                rank=r, world_size=ws, tp_size=tp, cp_size=middle, dp_size=dp
            )
            moe = MoeLayerMapping(
                rank=r, world_size=ws, tp_size=tp, ep_size=middle, dp_size=dp
            )
            assert attn.tp_rank == moe.tp_rank
            assert attn.cp_rank == moe.ep_rank
            assert attn.dp_rank == moe.dp_rank
            assert attn.tp_group == moe.tp_group
            assert attn.cp_group == moe.ep_group
            assert attn.dp_group == moe.dp_group

    def test_groups_partition_world(self):
        """All three group types should partition the world correctly."""
        ws = 24
        tp, ep, dp = 2, 3, 4
        mappings = [
            MoeLayerMapping(rank=r, world_size=ws, tp_size=tp, ep_size=ep, dp_size=dp)
            for r in range(ws)
        ]

        for attr, expected_count in [
            ("tp_group", ws // tp),
            ("ep_group", ws // ep),
            ("dp_group", ws // dp),
        ]:
            groups = set()
            for m in mappings:
                groups.add(tuple(getattr(m, attr)))
            assert (
                len(groups) == expected_count
            ), f"{attr}: expected {expected_count} groups, got {len(groups)}"
            all_ranks = sorted(r for g in groups for r in g)
            assert all_ranks == list(range(ws)), f"{attr}: groups don't cover all ranks"

    def test_all_ranks_consistent(self):
        """For every rank, rank == dp_rank * (tp*ep) + ep_rank * tp + tp_rank."""
        ws = 24
        tp, ep, dp = 2, 3, 4
        for r in range(ws):
            m = MoeLayerMapping(
                rank=r, world_size=ws, tp_size=tp, ep_size=ep, dp_size=dp
            )
            reconstructed = m.dp_rank * (tp * ep) + m.ep_rank * tp + m.tp_rank
            assert reconstructed == r, f"rank={r}: reconstructed={reconstructed}"

    def test_invalid_product_raises(self):
        with pytest.raises(AssertionError):
            MoeLayerMapping(rank=0, world_size=16, tp_size=2, ep_size=3, dp_size=4)


# =============================================================================
# Mapping (global)
# =============================================================================


class TestMapping:

    def test_pp1_explicit_is_legacy_parity(self):
        kwargs = dict(
            rank=3,
            world_size=8,
            attn_tp_size=4,
            attn_cp_size=1,
            dense_tp_size=8,
            moe_tp_size=2,
            moe_ep_size=4,
            nprocs_per_node=8,
            nnodes=1,
        )
        implicit = Mapping(**kwargs)
        explicit = Mapping(**kwargs, pipeline_parallel_size=1)

        assert repr(implicit) == repr(explicit)
        assert repr(explicit) == (
            "Mapping(rank=3, world_size=8)\n"
            "  Cluster : 1 node(s) x 8 proc(s)\n"
            "  Attention: tp=4  cp=1  dp=2\n"
            "    Vision: tp=4  item_dp=1\n"
            "  Dense   : tp=8  dp=1\n"
            "  MoE     : tp=2  ep=4  dp=1"
        )
        assert implicit.attn.tp_group == explicit.attn.tp_group == (0, 1, 2, 3)
        assert implicit.attn.dp_group == explicit.attn.dp_group == (3, 7)
        assert implicit.dense.tp_group == explicit.dense.tp_group == tuple(range(8))
        assert implicit.moe.tp_ep_group == explicit.moe.tp_ep_group == tuple(range(8))

    def test_pp2_layer_sizes_are_stage_local_groups_are_global(self):
        m = Mapping(
            rank=10,
            world_size=16,
            pipeline_parallel_size=2,
            attn_tp_size=4,
            attn_cp_size=1,
            attn_dp_size=2,
            dense_tp_size=8,
            dense_dp_size=1,
            moe_tp_size=2,
            moe_ep_size=4,
            moe_dp_size=1,
        )

        assert m.world_size == 16
        assert m.world_group == tuple(range(16))
        assert m.pipeline.stage_world_size == 8
        assert m.pipeline.stage_group == tuple(range(8, 16))
        assert m.pipeline.pipeline_group == (2, 10)

        assert m.attn.world_size == 8
        assert m.dense.world_size == 8
        assert m.moe.world_size == 8
        assert m.vision.world_size == 4
        assert m.attn.tp_group == (8, 9, 10, 11)
        assert m.attn.dp_group == (10, 14)
        assert m.dense.tp_group == tuple(range(8, 16))
        assert m.moe.tp_ep_group == tuple(range(8, 16))
        assert m.vision.tp_group == (8, 9, 10, 11)

    def test_pp8_all_rank_groups_stay_inside_stage_and_lanes_cover_world(self):
        mappings = [
            Mapping(
                rank=rank,
                world_size=64,
                pipeline_parallel_size=8,
                attn_tp_size=8,
                dense_tp_size=8,
                moe_tp_size=1,
                moe_ep_size=8,
            )
            for rank in range(64)
        ]

        stage_groups = {mapping.pipeline.stage_group for mapping in mappings}
        lanes = {mapping.pipeline.pipeline_group for mapping in mappings}
        assert stage_groups == {
            tuple(range(stage_id * 8, (stage_id + 1) * 8)) for stage_id in range(8)
        }
        assert lanes == {
            tuple(stage_local_rank + stage_id * 8 for stage_id in range(8))
            for stage_local_rank in range(8)
        }
        assert mappings[3].pipeline.pipeline_group == (
            3,
            11,
            19,
            27,
            35,
            43,
            51,
            59,
        )
        for mapping in mappings:
            stage_group = set(mapping.pipeline.stage_group)
            for group in (
                mapping.attn.tp_group,
                mapping.attn.cp_group,
                mapping.attn.dp_group,
                mapping.dense.tp_group,
                mapping.dense.dp_group,
                mapping.moe.tp_group,
                mapping.moe.ep_group,
                mapping.moe.dp_group,
                mapping.vision.tp_group,
                mapping.vision.dp_group,
            ):
                assert set(group) <= stage_group

    def test_pipeline_deferred_rank_propagates_global_rank(self):
        m = Mapping(
            world_size=16,
            pipeline_parallel_size=4,
            attn_tp_size=2,
            dense_tp_size=4,
            moe_tp_size=2,
            moe_ep_size=2,
        )

        assert m.pipeline.stage_count == 4
        assert m.pipeline.stage_world_size == 4
        with pytest.raises(AssertionError, match="rank is not initialized"):
            _ = m.pipeline.pipeline_group

        m.rank = 10
        assert m.pipeline.rank == 10
        assert m.pipeline.stage_index == 2
        assert m.pipeline.stage_group == (8, 9, 10, 11)
        assert m.pipeline.pipeline_group == (2, 6, 10, 14)
        assert m.attn.rank == 10
        assert m.attn.tp_group == (10, 11)
        assert m.dense.rank == 10
        assert m.dense.tp_group == (8, 9, 10, 11)
        assert m.moe.rank == 10
        assert m.moe.tp_ep_group == (8, 9, 10, 11)
        assert m.vision.rank == 10

    @pytest.mark.parametrize("value", [True, False, 2.0, "2", None])
    def test_invalid_pipeline_parallel_size_type(self, value):
        with pytest.raises(TypeError, match="stage_count must be an int"):
            Mapping(world_size=8, pipeline_parallel_size=value)

    def test_pipeline_parallel_size_must_divide_world(self):
        with pytest.raises(ValueError, match="must be divisible"):
            Mapping(world_size=8, pipeline_parallel_size=3)

    def test_deferred_pipeline_rank_validation_does_not_initialize(self):
        m = Mapping(world_size=8, pipeline_parallel_size=2)

        with pytest.raises(TypeError, match="rank must be an int"):
            m.rank = True
        with pytest.raises(AssertionError, match="rank is not initialized"):
            _ = m.rank

        m.rank = 5
        assert m.rank == 5
        assert m.pipeline.rank == 5

    def test_parallel_groups(self):
        m = Mapping(
            rank=3,
            world_size=8,
            attn_tp_size=4,
            dense_tp_size=8,
            moe_tp_size=2,
            moe_ep_size=4,
        )
        assert m.attn.tp_rank == 3
        assert m.dense.tp_rank == 3
        assert m.moe.ep_rank == 1

    def test_layer_mappings_use_global_rank(self):
        """Layer mappings should use the global rank so groups contain global ranks."""
        m = Mapping(rank=10, world_size=16, dense_tp_size=4)
        # rank=10, world_size=16, dense tp=4, dp=4
        assert m.dense.tp_group == (8, 9, 10, 11)
        assert m.dense.dp_group == (2, 6, 10, 14)
        # All group members should be valid global ranks
        assert all(0 <= r < 16 for r in m.dense.tp_group)
        assert all(0 <= r < 16 for r in m.dense.dp_group)

    def test_nprocs_per_node_and_nnodes(self):
        m = Mapping(rank=0, world_size=16, nprocs_per_node=8, nnodes=2)
        assert m.nprocs_per_node == 8
        assert m.nnodes == 2

    def test_nprocs_per_node_infer_nnodes(self):
        m = Mapping(rank=0, world_size=16, nprocs_per_node=8)
        assert m.nprocs_per_node == 8
        assert m.nnodes == 2

    def test_nnodes_infer_nprocs(self):
        m = Mapping(rank=0, world_size=16, nnodes=4)
        assert m.nprocs_per_node == 4
        assert m.nnodes == 4

    def test_nprocs_default_single_node(self):
        m = Mapping(rank=0, world_size=8)
        assert m.nprocs_per_node == 8
        assert m.nnodes == 1

    def test_nprocs_nnodes_mismatch_raises(self):
        with pytest.raises(AssertionError):
            Mapping(rank=0, world_size=16, nprocs_per_node=3, nnodes=2)

    def test_deferred_rank(self):
        """Create Mapping without rank, set it later, verify propagation."""
        m = Mapping(
            world_size=8,
            attn_tp_size=4,
            dense_tp_size=8,
            moe_tp_size=2,
            moe_ep_size=4,
        )
        # Sizes resolved eagerly
        assert m.attn.tp_size == 4
        assert m.dense.tp_size == 8
        assert m.moe.tp_size == 2
        assert m.moe.ep_size == 4

        # Set rank later
        m.rank = 3
        assert m.rank == 3
        # Propagated to sub-mappings
        assert m.attn.rank == 3
        assert m.dense.rank == 3
        assert m.moe.rank == 3
        # Derived ranks work
        assert m.attn.tp_rank == 3
        assert m.dense.tp_rank == 3
        assert m.moe.ep_rank == 1

    def test_deferred_rank_access_before_set_raises(self):
        m = Mapping(world_size=8, dense_tp_size=4)
        with pytest.raises(AssertionError, match="rank is not initialized"):
            _ = m.rank
        with pytest.raises(AssertionError, match="rank is not initialized"):
            _ = m.dense.tp_rank

    def test_deferred_rank_set_once_raises(self):
        m = Mapping(world_size=8, dense_tp_size=4)
        m.rank = 3
        with pytest.raises(AssertionError, match="rank is already initialized"):
            m.rank = 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
