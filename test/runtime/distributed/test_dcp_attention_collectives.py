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

from unittest.mock import patch

import torch

from tokenspeed.runtime.distributed.dcp import (
    DcpAttentionWorkspace,
    gather_dcp_queries,
    gather_dcp_sinks,
    merge_dcp_attention_states,
)


def _workspace(local_heads: int = 1) -> DcpAttentionWorkspace:
    return DcpAttentionWorkspace.allocate(
        dcp_size=2,
        max_rows=1,
        local_heads=local_heads,
        head_dim=1,
        dtype=torch.float32,
        device="cpu",
    )


def test_gather_queries_preserves_dcp_major_head_order() -> None:
    workspace = _workspace(local_heads=2)

    def fake_gather(out, _local, _group):
        out.copy_(torch.tensor([[[[1.0], [2.0]]], [[[3.0], [4.0]]]]))

    with patch(
        "tokenspeed.runtime.distributed.dcp._all_gather_into_dcp_major",
        side_effect=fake_gather,
    ):
        gathered = gather_dcp_queries(
            torch.zeros(1, 2, 1), group=(0, 1), workspace=workspace
        )

    torch.testing.assert_close(gathered.flatten(), torch.tensor([1.0, 2.0, 3.0, 4.0]))


def test_sink_is_present_on_exactly_one_kv_partition() -> None:
    workspace = _workspace(local_heads=2)

    def fake_gather(out, local, _group):
        out.copy_(torch.stack((local, local + 2)))

    with patch(
        "tokenspeed.runtime.distributed.dcp._all_gather_into_dcp_major",
        side_effect=fake_gather,
    ):
        owner = gather_dcp_sinks(
            torch.tensor([1.0, 2.0]),
            group=(0, 1),
            dcp_rank=0,
            workspace=workspace,
        ).clone()
        non_owner = gather_dcp_sinks(
            torch.tensor([1.0, 2.0]),
            group=(0, 1),
            dcp_rank=1,
            workspace=workspace,
        ).clone()

    torch.testing.assert_close(owner, torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.isneginf(non_owner).all()


def test_merge_uses_natural_lse_and_reduce_scatters_heads() -> None:
    workspace = _workspace()
    gathered_lse = torch.log(torch.tensor([[[1.0, 3.0]], [[3.0, 1.0]]]))
    partial_out_by_rank = (
        torch.tensor([[[1.0], [10.0]]]),
        torch.tensor([[[3.0], [30.0]]]),
    )

    def fake_gather(out, _local, _group):
        out.copy_(gathered_lse)

    def fake_reduce_scatter(out, current, _group):
        other_rank = 1 - active_rank
        total = current[active_rank].clone()
        other_weight = torch.exp(
            gathered_lse[other_rank] - torch.logsumexp(gathered_lse, dim=0)
        )
        other = partial_out_by_rank[other_rank] * other_weight.unsqueeze(-1)
        total += other[:, active_rank : active_rank + 1]
        out.copy_(total)

    for active_rank, expected in ((0, 2.5), (1, 15.0)):
        with (
            patch(
                "tokenspeed.runtime.distributed.dcp._all_gather_into_dcp_major",
                side_effect=fake_gather,
            ),
            patch(
                "tokenspeed.runtime.distributed.dcp._reduce_scatter_dcp_heads",
                side_effect=fake_reduce_scatter,
            ),
        ):
            out, lse = merge_dcp_attention_states(
                partial_out_by_rank[active_rank],
                gathered_lse[active_rank],
                group=(0, 1),
                dcp_rank=active_rank,
                local_output_heads=1,
                workspace=workspace,
            )

        torch.testing.assert_close(out, torch.tensor([[[expected]]]))
        torch.testing.assert_close(lse, torch.log(torch.tensor([[4.0]])))


def test_merge_empty_shards_produces_zero_output_and_negative_infinite_lse() -> None:
    workspace = _workspace()

    def fake_gather(out, _local, _group):
        out.fill_(-torch.inf)

    def fake_reduce_scatter(out, current, _group):
        out.copy_(current.sum(dim=0))

    with (
        patch(
            "tokenspeed.runtime.distributed.dcp._all_gather_into_dcp_major",
            side_effect=fake_gather,
        ),
        patch(
            "tokenspeed.runtime.distributed.dcp._reduce_scatter_dcp_heads",
            side_effect=fake_reduce_scatter,
        ),
    ):
        out, lse = merge_dcp_attention_states(
            torch.ones(1, 2, 1),
            torch.full((1, 2), -torch.inf),
            group=(0, 1),
            dcp_rank=0,
            local_output_heads=1,
            workspace=workspace,
        )

    torch.testing.assert_close(out, torch.zeros_like(out))
    assert torch.isneginf(lse).all()
