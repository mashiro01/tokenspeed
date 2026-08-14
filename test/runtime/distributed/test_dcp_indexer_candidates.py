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

import pytest
import torch

import tokenspeed.runtime.distributed.dcp as dcp


@pytest.mark.parametrize("dcp_size", [2, 4])
@pytest.mark.parametrize("interleave_size", [1, 2, 4])
def test_owned_prefix_lengths_match_exact_interleaved_owner_counts(
    dcp_size: int,
    interleave_size: int,
) -> None:
    cycle = dcp_size * interleave_size
    global_lengths = torch.arange(0, 3 * cycle + interleave_size + 1)
    global_rows = torch.arange(int(global_lengths.max()))
    owners = torch.remainder(
        torch.div(global_rows, interleave_size, rounding_mode="floor"),
        dcp_size,
    )

    for dcp_rank in range(dcp_size):
        expected = (
            (global_rows.unsqueeze(0) < global_lengths.unsqueeze(1))
            & owners.unsqueeze(0).eq(dcp_rank)
        ).sum(dim=1)
        actual = dcp.dcp_owned_prefix_lengths(
            global_lengths,
            dcp_size=dcp_size,
            dcp_rank=dcp_rank,
            interleave_size=interleave_size,
        )

        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("dcp_size", [2, 4])
@pytest.mark.parametrize("interleave_size", [1, 2, 4])
def test_restore_global_rows_is_inverse_with_nonzero_local_base(
    dcp_size: int,
    interleave_size: int,
) -> None:
    local_row_base = torch.tensor(
        [[2 * interleave_size + 1], [5 * interleave_size]],
        dtype=torch.int64,
    )
    relative_local_rows = torch.arange(2 * interleave_size + 3).repeat(2, 1)

    for dcp_rank in range(dcp_size):
        absolute_local_rows = relative_local_rows + local_row_base
        expected = (
            torch.div(
                absolute_local_rows,
                interleave_size,
                rounding_mode="floor",
            )
            * (dcp_size * interleave_size)
            + dcp_rank * interleave_size
            + torch.remainder(absolute_local_rows, interleave_size)
        )
        restored = dcp.restore_dcp_global_rows(
            relative_local_rows,
            dcp_size=dcp_size,
            dcp_rank=dcp_rank,
            interleave_size=interleave_size,
            local_row_base=local_row_base,
        )

        torch.testing.assert_close(restored, expected)
        roundtrip_local, owned = dcp.shard_dcp_logical_rows(
            restored,
            dcp_size=dcp_size,
            dcp_rank=dcp_rank,
            interleave_size=interleave_size,
        )
        assert owned.all()
        torch.testing.assert_close(roundtrip_local, absolute_local_rows)


def test_restore_global_rows_preserves_invalid_candidate_sentinel() -> None:
    restored = dcp.restore_dcp_global_rows(
        torch.tensor([[-1, 0, 1]], dtype=torch.int32),
        dcp_size=4,
        dcp_rank=2,
        interleave_size=2,
        local_row_base=5,
    )

    torch.testing.assert_close(restored, torch.tensor([[-1, 21, 28]]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_restore_global_rows_is_cuda_graph_safe_with_scalar_base() -> None:
    local_rows = torch.tensor([[-1, 0, 1]], dtype=torch.int32, device="cuda")
    capture_stream = torch.cuda.Stream()

    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            restored = dcp.restore_dcp_global_rows(
                local_rows,
                dcp_size=2,
                dcp_rank=1,
                interleave_size=1,
                local_row_base=3,
            )
    capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        restored = dcp.restore_dcp_global_rows(
            local_rows,
            dcp_size=2,
            dcp_rank=1,
            interleave_size=1,
            local_row_base=3,
        )
    graph.replay()

    torch.testing.assert_close(
        restored,
        torch.tensor([[-1, 7, 9]], dtype=torch.int64, device="cuda"),
    )


def test_gather_indexer_candidates_materializes_only_dcp_times_local_topk() -> None:
    group = (4, 7, 9, 12)
    local_scores = torch.tensor([[4.0, 3.0], [2.0, 1.0]])
    local_rows = torch.tensor([[40, 41], [50, 51]], dtype=torch.int32)
    scores_by_rank = torch.tensor(
        [
            [[4.0, 3.0], [2.0, 1.0]],
            [[8.0, 7.0], [6.0, 5.0]],
            [[12.0, 11.0], [10.0, 9.0]],
            [[16.0, 15.0], [14.0, 13.0]],
        ]
    )
    rows_by_rank = torch.tensor(
        [
            [[40, 41], [50, 51]],
            [[140, 141], [150, 151]],
            [[240, 241], [250, 251]],
            [[340, 341], [350, 351]],
        ],
        dtype=torch.int32,
    )
    workspace = dcp.DcpIndexerCandidateWorkspace.allocate(
        dcp_size=len(group),
        max_rows=3,
        topk=local_scores.shape[1],
        device="cpu",
    )
    gathered_inputs: list[torch.Tensor] = []

    def fake_gather(
        output: torch.Tensor,
        current: torch.Tensor,
        actual_group: tuple[int, ...],
    ) -> None:
        assert actual_group == group
        gathered_inputs.append(current.clone())
        source = scores_by_rank if current.is_floating_point() else rows_by_rank
        if current.is_floating_point():
            output.fill_(-torch.inf)
        else:
            output.fill_(-1)
        output[:, : source.shape[1]].copy_(source)

    with patch(
        "tokenspeed.runtime.distributed.dcp._all_gather_into_dcp_major",
        side_effect=fake_gather,
    ) as gather:
        candidate_scores, candidate_rows = dcp.gather_dcp_indexer_candidates(
            local_scores,
            local_rows,
            group=group,
            workspace=workspace,
        )

    assert gather.call_count == 2
    torch.testing.assert_close(gathered_inputs[0][:2], local_scores)
    torch.testing.assert_close(gathered_inputs[1][:2], local_rows)
    assert torch.isneginf(gathered_inputs[0][2]).all()
    assert gathered_inputs[1][2].eq(-1).all()
    assert candidate_scores.shape == (2, len(group) * local_scores.shape[1])
    assert candidate_rows.shape == candidate_scores.shape
    assert workspace.gathered_scores.numel() == len(group) * 3 * local_scores.shape[1]
    assert workspace.gathered_rows.numel() == len(group) * 3 * local_rows.shape[1]
    assert workspace.candidate_scores.numel() == len(group) * 3 * local_scores.shape[1]
    assert workspace.candidate_rows.numel() == len(group) * 3 * local_rows.shape[1]
    torch.testing.assert_close(
        candidate_scores,
        scores_by_rank.permute(1, 0, 2).reshape(2, -1),
    )
    torch.testing.assert_close(
        candidate_rows,
        rows_by_rank.permute(1, 0, 2).reshape(2, -1),
    )
