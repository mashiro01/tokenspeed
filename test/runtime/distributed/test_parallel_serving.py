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

import pytest

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.distributed.parallel_serving import ParallelServingPlan
from tokenspeed.runtime.execution import distributed_initializer


def _mapping(*, rank=0, tp=4, cp=1, dp=1):
    return Mapping(
        rank=rank,
        world_size=tp * cp * dp,
        attn_tp_size=tp,
        attn_cp_size=cp,
        attn_dp_size=dp,
        dense_tp_size=tp * cp,
        dense_dp_size=dp,
        moe_tp_size=tp * cp,
        moe_ep_size=1,
        moe_dp_size=dp,
        vision_tp_size=tp,
        vision_dp_size=1,
    )


@pytest.mark.parametrize("dcp", [2, 4])
def test_dcp_is_nested_in_tp_without_changing_world_size(dcp):
    mapping = _mapping(tp=4)
    plan = ParallelServingPlan.resolve(
        mapping,
        decode_context_parallel_size=dcp,
        cp_kv_cache_interleave_size=1,
    )

    assert mapping.world_size == 4
    assert mapping.attn.tp_size == 4
    assert mapping.attn.cp_size == 1
    assert plan.decode_context_parallel_size == dcp


def test_dcp_subgroups_are_contiguous_inside_each_adp_replica():
    groups = []
    for rank in range(8):
        mapping = _mapping(rank=rank, tp=4, dp=2)
        plan = ParallelServingPlan.resolve(
            mapping,
            decode_context_parallel_size=2,
            cp_kv_cache_interleave_size=1,
        )
        groups.append(plan.dcp_group(mapping))

    assert groups == [
        (0, 1),
        (0, 1),
        (2, 3),
        (2, 3),
        (4, 5),
        (4, 5),
        (6, 7),
        (6, 7),
    ]


def test_dcp_rejects_legacy_cp_and_nondivisible_tp():
    with pytest.raises(ValueError, match="legacy"):
        ParallelServingPlan.resolve(
            _mapping(tp=1, cp=4),
            decode_context_parallel_size=2,
            cp_kv_cache_interleave_size=1,
        )
    with pytest.raises(ValueError, match="divisible"):
        ParallelServingPlan.resolve(
            _mapping(tp=4),
            decode_context_parallel_size=3,
            cp_kv_cache_interleave_size=1,
        )


def test_initializer_registers_adp_and_exact_dcp_groups(monkeypatch):
    mapping = _mapping(rank=5, tp=4, dp=2)
    plan = ParallelServingPlan.resolve(
        mapping,
        decode_context_parallel_size=2,
        cp_kv_cache_interleave_size=1,
    )
    calls = []

    def record(group, **kwargs):
        calls.append((group, kwargs))

    monkeypatch.setattr(
        distributed_initializer.pg_manager,
        "init_process_group",
        record,
    )
    config = type(
        "Config",
        (),
        {
            "mapping": mapping,
            "dcp_size": plan.decode_context_parallel_size,
            "dcp_group": plan.dcp_group(mapping),
        },
    )()

    distributed_initializer._initialize_model_process_groups(config)

    assert (mapping.attn.dp_group, {}) in calls
    assert (plan.dcp_group(mapping), {"backend": "nccl", "role": "dcp"}) in calls
