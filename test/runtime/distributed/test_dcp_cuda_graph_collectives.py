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

"""Real NCCL CUDA-graph replay coverage for DCP fixed-shape collectives."""

from __future__ import annotations

import math
import os
import socket
import sys
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

import tokenspeed.runtime.distributed.dcp as dcp  # noqa: E402
from tokenspeed.runtime.distributed.process_group_manager import (  # noqa: E402
    process_group_manager as pg_manager,
)

register_cuda_ci(est_time=30, suite="runtime-2gpu")

_WORLD_SIZE = 2
_REPLAYS = 64


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def _seed_inputs(
    *,
    rank: int,
    pattern: int,
    q: torch.Tensor,
    partial_out: torch.Tensor,
    partial_lse: torch.Tensor,
    candidate_scores: torch.Tensor,
    candidate_rows: torch.Tensor,
) -> None:
    q.copy_(torch.arange(q.numel(), device=q.device).reshape(q.shape))
    q.add_(rank * 100 + pattern * 10)
    partial_out.copy_(
        torch.arange(partial_out.numel(), device=q.device).reshape(partial_out.shape)
    )
    partial_out.add_(rank * 100 + pattern * 10)
    lse_weight = (1.0, 3.0)[rank] + pattern * (1.0, 2.0)[rank]
    partial_lse.fill_(math.log(lse_weight))
    candidate_scores.copy_(
        torch.arange(candidate_scores.numel(), device=q.device).reshape(
            candidate_scores.shape
        )
    )
    candidate_scores.add_(rank * 100 + pattern * 10)
    candidate_rows.copy_(
        torch.arange(
            candidate_rows.numel(),
            device=q.device,
            dtype=candidate_rows.dtype,
        ).reshape(candidate_rows.shape)
    )
    candidate_rows.add_(rank * 1000 + pattern * 100)


def _expected(
    *,
    rank: int,
    pattern: int,
    rows: int,
    local_heads: int,
    head_dim: int,
    topk: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    q_by_rank = torch.stack(
        [
            torch.arange(rows * local_heads * head_dim, device=device).reshape(
                rows, local_heads, head_dim
            ).to(torch.float32)
            + source_rank * 100
            + pattern * 10
            for source_rank in range(_WORLD_SIZE)
        ]
    )
    expected_q = q_by_rank.permute(1, 0, 2, 3).reshape(
        rows, _WORLD_SIZE * local_heads, head_dim
    )

    group_heads = _WORLD_SIZE * local_heads
    out_by_rank = torch.stack(
        [
            torch.arange(rows * group_heads * head_dim, device=device).reshape(
                rows, group_heads, head_dim
            ).to(torch.float32)
            + source_rank * 100
            + pattern * 10
            for source_rank in range(_WORLD_SIZE)
        ]
    )
    weights = torch.tensor(
        [1.0 + pattern, 3.0 + pattern * 2.0], device=device
    ).reshape(_WORLD_SIZE, 1, 1, 1)
    expected_merged = (out_by_rank * weights).sum(dim=0) / weights.sum()
    head_start = rank * local_heads
    expected_local_out = expected_merged[:, head_start : head_start + local_heads]
    expected_lse = torch.full(
        (rows, local_heads),
        math.log(float(weights.sum())),
        device=device,
    )

    score_by_rank = torch.stack(
        [
            torch.arange(rows * topk, device=device).reshape(rows, topk)
            .to(torch.float32)
            + source_rank * 100
            + pattern * 10
            for source_rank in range(_WORLD_SIZE)
        ]
    )
    row_by_rank = torch.stack(
        [
            torch.arange(rows * topk, device=device, dtype=torch.int32).reshape(
                rows, topk
            )
            + source_rank * 1000
            + pattern * 100
            for source_rank in range(_WORLD_SIZE)
        ]
    )
    expected_scores = score_by_rank.permute(1, 0, 2).reshape(rows, -1)
    expected_rows = row_by_rank.permute(1, 0, 2).reshape(rows, -1)
    expected_top_scores, expected_offsets = torch.topk(
        expected_scores, k=topk, dim=-1, sorted=True
    )
    expected_top_rows = expected_rows.gather(1, expected_offsets)
    return (
        expected_q,
        expected_local_out,
        expected_lse,
        expected_scores,
        expected_rows,
        expected_top_scores,
        expected_top_rows,
    )


def _run_graph_replay_case(
    *,
    rank: int,
    device: torch.device,
    group: tuple[int, ...],
) -> None:
    rows, local_heads, head_dim, topk = 2, 2, 4, 3
    attention_workspace = dcp.DcpAttentionWorkspace.allocate(
        dcp_size=_WORLD_SIZE,
        max_rows=rows,
        local_heads=local_heads,
        head_dim=head_dim,
        dtype=torch.float32,
        device=device,
    )
    indexer_workspace = dcp.DcpIndexerCandidateWorkspace.allocate(
        dcp_size=_WORLD_SIZE,
        max_rows=rows,
        topk=topk,
        device=device,
    )
    q = torch.empty((rows, local_heads, head_dim), device=device)
    partial_out = torch.empty(
        (rows, _WORLD_SIZE * local_heads, head_dim), device=device
    )
    partial_lse = torch.empty((rows, _WORLD_SIZE * local_heads), device=device)
    local_scores = torch.empty((rows, topk), device=device)
    global_rows = torch.empty((rows, topk), dtype=torch.int32, device=device)

    def forward() -> tuple[torch.Tensor, ...]:
        gathered_q = dcp.gather_dcp_queries(
            q,
            group=group,
            workspace=attention_workspace,
        )
        merged_out, merged_lse = dcp.merge_dcp_attention_states(
            partial_out,
            partial_lse,
            group=group,
            dcp_rank=rank,
            local_output_heads=local_heads,
            workspace=attention_workspace,
        )
        candidate_scores, candidate_rows = dcp.gather_dcp_indexer_candidates(
            local_scores,
            global_rows,
            group=group,
            workspace=indexer_workspace,
        )
        top_scores, top_offsets = torch.topk(
            candidate_scores, k=topk, dim=-1, sorted=True
        )
        top_rows = candidate_rows.gather(1, top_offsets)
        return (
            gathered_q,
            merged_out,
            merged_lse,
            candidate_scores,
            candidate_rows,
            top_scores,
            top_rows,
        )

    _seed_inputs(
        rank=rank,
        pattern=0,
        q=q,
        partial_out=partial_out,
        partial_lse=partial_lse,
        candidate_scores=local_scores,
        candidate_rows=global_rows,
    )
    warmup_stream = torch.cuda.Stream(device=device)
    warmup_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            forward()
    warmup_stream.synchronize()
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=warmup_stream):
        graph_outputs = forward()
    torch.cuda.synchronize(device)
    dist.barrier()

    fingerprints: dict[int, tuple[torch.Tensor, ...]] = {}
    failures: list[str] = []
    names = (
        "gathered_q",
        "merged_out",
        "merged_lse",
        "candidate_scores",
        "candidate_rows",
        "top_scores",
        "top_rows",
    )
    for replay in range(_REPLAYS):
        pattern = replay % 2
        _seed_inputs(
            rank=rank,
            pattern=pattern,
            q=q,
            partial_out=partial_out,
            partial_lse=partial_lse,
            candidate_scores=local_scores,
            candidate_rows=global_rows,
        )
        graph.replay()
        torch.cuda.synchronize(device)
        actual = tuple(tensor.clone() for tensor in graph_outputs)
        expected = _expected(
            rank=rank,
            pattern=pattern,
            rows=rows,
            local_heads=local_heads,
            head_dim=head_dim,
            topk=topk,
            device=device,
        )
        for name, current, reference in zip(names, actual, expected, strict=True):
            if current.is_floating_point():
                matches = torch.allclose(current, reference, rtol=1e-6, atol=1e-6)
            else:
                matches = torch.equal(current, reference)
            if not matches and len(failures) < 8:
                failures.append(f"replay={replay} tensor={name}")

        prior = fingerprints.get(pattern)
        if prior is None:
            fingerprints[pattern] = actual
        else:
            for name, current, first in zip(names, actual, prior, strict=True):
                if not torch.equal(current, first) and len(failures) < 8:
                    failures.append(
                        f"replay={replay} tensor={name} differs bitwise from "
                        f"the first replay of pattern={pattern}"
                    )

    dist.barrier()
    if failures:
        raise AssertionError("; ".join(failures))


def _worker_main(
    rank: int,
    world_size: int,
    port: int,
    error_dict,
) -> None:
    try:
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://localhost:{port}",
            rank=rank,
            world_size=world_size,
        )
        group = tuple(range(world_size))
        pg_manager.init_process_group(group, backend="nccl", role="dcp")
        _run_graph_replay_case(rank=rank, device=device, group=group)
    except Exception:
        error_dict[rank] = traceback.format_exc()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_dcp2_collectives_are_deterministic_across_repeated_cuda_graph_replay() -> None:
    if torch.cuda.device_count() < _WORLD_SIZE:
        pytest.skip(f"Need {_WORLD_SIZE} GPUs, have {torch.cuda.device_count()}")
    port = _get_open_port()
    error_dict = mp.Manager().dict()
    mp.spawn(
        _worker_main,
        args=(_WORLD_SIZE, port, error_dict),
        nprocs=_WORLD_SIZE,
        join=True,
    )
    if error_dict:
        raise RuntimeError(
            "\n".join(f"Rank {rank}: {error}" for rank, error in error_dict.items())
        )
