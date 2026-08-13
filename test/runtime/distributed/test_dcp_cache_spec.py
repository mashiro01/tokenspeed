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

from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.distributed.dcp import shard_dcp_logical_rows
from tokenspeed.runtime.layers.attention.kv_cache.recipes.deepseek_v4_cache_spec import (
    V4_INDEXER_COMPRESSOR_STATE_GROUP_ID,
    V4_SWA_KV_GROUP_ID,
    build_v4_cache_specs,
    deepseek_v4_scheduler_block_tokens,
    v4_compressed_kv_group_id,
    v4_compressor_state_group_id,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    compute_paged_cache_group_page_counts,
)


def _specs(dcp: int):
    return build_v4_cache_specs(
        SimpleNamespace(sliding_window=128),
        layer_ratio=(4, 128),
        decode_context_parallel_size=dcp,
    )


def test_dcp_shards_history_and_swa_but_keeps_tail_state_replicated():
    specs = {spec.group_id: spec for spec in _specs(4)}

    assert specs[V4_SWA_KV_GROUP_ID].entry_stride_tokens == 4
    assert specs[v4_compressed_kv_group_id(4)].entry_stride_tokens == 16
    assert specs[v4_compressed_kv_group_id(128)].entry_stride_tokens == 512
    assert specs[v4_compressor_state_group_id(4)].entry_stride_tokens == 1
    assert specs[v4_compressor_state_group_id(128)].entry_stride_tokens == 1
    assert specs[V4_INDEXER_COMPRESSOR_STATE_GROUP_ID].entry_stride_tokens == 1


def test_dcp_reduces_one_million_token_history_page_budget():
    sizing = dict(
        max_live_requests=1,
        max_scheduled_tokens=8192,
        max_total_tokens=1_048_576,
        max_context_len=1_048_576,
    )
    tp = compute_paged_cache_group_page_counts(_specs(1), **sizing)
    dcp4 = compute_paged_cache_group_page_counts(_specs(4), **sizing)

    for group_id in (
        V4_SWA_KV_GROUP_ID,
        v4_compressed_kv_group_id(4),
        v4_compressed_kv_group_id(128),
    ):
        assert dcp4[group_id] < tp[group_id]
    for group_id in (
        v4_compressor_state_group_id(4),
        v4_compressor_state_group_id(128),
        V4_INDEXER_COMPRESSOR_STATE_GROUP_ID,
    ):
        assert dcp4[group_id] == tp[group_id]


@pytest.mark.parametrize("dcp_size", [1, 2, 4])
def test_dcp_scheduler_domain_scales_with_context_parallelism(dcp_size):
    assert deepseek_v4_scheduler_block_tokens(dcp_size) == 256 * dcp_size


def test_packed_rows_have_exactly_one_physical_dcp_owner():
    rows = torch.arange(64, dtype=torch.int64)
    ownership = []
    local_rows = []
    for rank in range(4):
        local, owned = shard_dcp_logical_rows(
            rows,
            dcp_size=4,
            dcp_rank=rank,
            interleave_size=2,
        )
        ownership.append(owned)
        local_rows.append(local)

    assert torch.stack(ownership).sum(dim=0).eq(1).all()
    assert torch.equal(local_rows[0][:10], torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 2, 3]))


@pytest.mark.parametrize(
    ("rows", "kwargs", "error"),
    [
        (torch.tensor([0.0]), dict(dcp_size=2, dcp_rank=0, interleave_size=1), TypeError),
        (torch.tensor([-1]), dict(dcp_size=2, dcp_rank=0, interleave_size=1), ValueError),
        (torch.tensor([0]), dict(dcp_size=0, dcp_rank=0, interleave_size=1), ValueError),
        (torch.tensor([0]), dict(dcp_size=2, dcp_rank=2, interleave_size=1), ValueError),
        (torch.tensor([0]), dict(dcp_size=2, dcp_rank=0, interleave_size=0), ValueError),
    ],
)
def test_dcp_row_sharding_rejects_invalid_contracts(rows, kwargs, error):
    with pytest.raises(error):
        shard_dcp_logical_rows(rows, **kwargs)


def test_cache_spec_rejects_interleave_that_splits_physical_pages():
    with pytest.raises(ValueError, match="must divide every sharded"):
        build_v4_cache_specs(
            SimpleNamespace(sliding_window=128),
            layer_ratio=(4, 128),
            decode_context_parallel_size=2,
            cp_kv_cache_interleave_size=3,
        )
