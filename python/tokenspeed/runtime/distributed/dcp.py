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

"""Physical cache ownership for TP-nested decode context parallelism."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.attention import dcp_attn_merge_rs

from tokenspeed.runtime.distributed.mapping import Group
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)


@dataclass(frozen=True)
class DcpAttentionWorkspace:
    """Fixed-shape buffers for capture-safe DCP attention collectives."""

    local_q: torch.Tensor
    gathered_q: torch.Tensor
    local_lse: torch.Tensor
    gathered_lse: torch.Tensor
    merged_lse: torch.Tensor
    reduce_scatter_input: torch.Tensor
    reduce_scatter_output: torch.Tensor

    @classmethod
    def allocate(
        cls,
        *,
        dcp_size: int,
        max_rows: int,
        local_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> "DcpAttentionWorkspace":
        """Allocate all DCP decode buffers before CUDA graph capture."""

        if min(dcp_size, max_rows, local_heads, head_dim) <= 0:
            raise ValueError("DCP workspace dimensions must be positive")
        group_heads = dcp_size * local_heads
        return cls(
            local_q=torch.empty(
                (max_rows, local_heads, head_dim), dtype=dtype, device=device
            ),
            gathered_q=torch.empty(
                (dcp_size, max_rows, local_heads, head_dim),
                dtype=dtype,
                device=device,
            ),
            local_lse=torch.empty(
                (max_rows, group_heads), dtype=torch.float32, device=device
            ),
            gathered_lse=torch.empty(
                (dcp_size, max_rows, group_heads),
                dtype=torch.float32,
                device=device,
            ),
            merged_lse=torch.empty(
                (max_rows, group_heads), dtype=torch.float32, device=device
            ),
            reduce_scatter_input=torch.empty(
                (dcp_size, max_rows, local_heads, head_dim),
                dtype=dtype,
                device=device,
            ),
            reduce_scatter_output=torch.empty(
                (max_rows, local_heads, head_dim), dtype=dtype, device=device
            ),
        )


@dataclass(frozen=True)
class DcpIndexerCandidateWorkspace:
    """Fixed-shape buffers for O(DCP x K) sparse-indexer candidate exchange."""

    local_candidates: torch.Tensor
    gathered_candidates: torch.Tensor
    candidate_records: torch.Tensor
    candidate_scores: torch.Tensor
    candidate_rows: torch.Tensor
    candidate_offsets: torch.Tensor
    safe_candidate_offsets: torch.Tensor
    candidate_lengths: torch.Tensor
    final_rows: torch.Tensor

    @classmethod
    def allocate(
        cls,
        *,
        dcp_size: int,
        max_rows: int,
        topk: int,
        device: torch.device | str,
    ) -> "DcpIndexerCandidateWorkspace":
        """Allocate candidate buffers before decode graph warmup/capture."""

        if min(dcp_size, max_rows, topk) <= 0:
            raise ValueError("DCP indexer workspace dimensions must be positive")
        return cls(
            local_candidates=torch.empty(
                (max_rows, topk), dtype=torch.int64, device=device
            ),
            gathered_candidates=torch.empty(
                (dcp_size, max_rows, topk), dtype=torch.int64, device=device
            ),
            candidate_records=torch.empty(
                (max_rows, dcp_size * topk), dtype=torch.int64, device=device
            ),
            candidate_scores=torch.empty(
                (max_rows, dcp_size * topk), dtype=torch.float32, device=device
            ),
            candidate_rows=torch.empty(
                (max_rows, dcp_size * topk), dtype=torch.int32, device=device
            ),
            candidate_offsets=torch.empty(
                (max_rows, topk), dtype=torch.int32, device=device
            ),
            safe_candidate_offsets=torch.empty(
                (max_rows, topk), dtype=torch.int64, device=device
            ),
            candidate_lengths=torch.full(
                (max_rows,), dcp_size * topk, dtype=torch.int32, device=device
            ),
            final_rows=torch.empty((max_rows, topk), dtype=torch.int32, device=device),
        )


def dcp_owned_prefix_lengths(
    global_lengths: torch.Tensor,
    *,
    dcp_size: int,
    dcp_rank: int,
    interleave_size: int,
) -> torch.Tensor:
    """Count rows owned by one DCP rank in each global prefix ``[0, length)``."""

    _validate_dcp_sharding(dcp_size, dcp_rank, interleave_size)
    lengths = global_lengths.clamp_min(0)
    cycle = dcp_size * interleave_size
    cycles = torch.div(lengths, cycle, rounding_mode="floor")
    remainder = torch.remainder(lengths, cycle)
    tail = (remainder - dcp_rank * interleave_size).clamp(0, interleave_size)
    return cycles * interleave_size + tail


def restore_dcp_global_rows(
    local_rows: torch.Tensor,
    *,
    dcp_size: int,
    dcp_rank: int,
    interleave_size: int,
    local_row_base: int | torch.Tensor = 0,
) -> torch.Tensor:
    """Restore absolute global rows from one rank's local logical coordinates."""

    _validate_dcp_sharding(dcp_size, dcp_rank, interleave_size)
    valid = local_rows >= 0
    if isinstance(local_row_base, torch.Tensor):
        if local_row_base.device != local_rows.device:
            raise ValueError("DCP local row base must be on the candidate-row device")
        local_row_base = local_row_base.to(torch.int64)
    else:
        local_row_base = int(local_row_base)
    absolute_local = local_rows.to(torch.int64) + local_row_base
    cycle = dcp_size * interleave_size
    global_rows = (
        torch.div(absolute_local, interleave_size, rounding_mode="floor") * cycle
        + dcp_rank * interleave_size
        + torch.remainder(absolute_local, interleave_size)
    )
    return torch.where(valid, global_rows, torch.full_like(global_rows, -1))


def gather_dcp_indexer_candidates(
    local_scores: torch.Tensor,
    local_rows: torch.Tensor,
    *,
    group: Group,
    workspace: DcpIndexerCandidateWorkspace,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exchange score/row records with one DCP all-gather.

    Each candidate is transferred as one opaque 64-bit record containing the
    exact float32 score bits followed by the int32 global row. This halves the
    latency-sensitive indexer collectives without changing score ordering or
    row values.
    """

    _validate_group(group)
    if local_scores.ndim != 2 or local_rows.shape != local_scores.shape:
        raise ValueError(
            "DCP indexer candidates must have matching [rows, topk] shapes"
        )
    if local_scores.dtype != torch.float32 or local_rows.dtype != torch.int32:
        raise TypeError("DCP indexer candidates require float32 scores and int32 rows")
    rows, topk = local_scores.shape
    if len(group) != workspace.gathered_candidates.shape[0]:
        raise ValueError("DCP group and indexer workspace sizes disagree")
    if (
        rows > workspace.local_candidates.shape[0]
        or topk != workspace.local_candidates.shape[1]
    ):
        raise ValueError("DCP indexer candidates exceed the preallocated workspace")

    local_fields = workspace.local_candidates.view(torch.int32).view(
        *workspace.local_candidates.shape, 2
    )
    local_fields[:rows, :, 0].copy_(local_scores.view(torch.int32))
    local_fields[:rows, :, 1].copy_(local_rows)
    if rows < workspace.local_candidates.shape[0]:
        # IEEE-754 float32 -inf bit pattern followed by the invalid row sentinel.
        local_fields[rows:, :, 0].fill_(-0x800000)
        local_fields[rows:, :, 1].fill_(-1)
    _all_gather_into_dcp_major(
        workspace.gathered_candidates,
        workspace.local_candidates,
        group,
    )
    for rank in range(len(group)):
        start = rank * topk
        end = start + topk
        workspace.candidate_records[:, start:end].copy_(
            workspace.gathered_candidates[rank]
        )
    candidate_fields = workspace.candidate_records.view(torch.int32).view(
        *workspace.candidate_records.shape, 2
    )
    workspace.candidate_scores.view(torch.int32).copy_(candidate_fields[:, :, 0])
    workspace.candidate_rows.copy_(candidate_fields[:, :, 1])
    return workspace.candidate_scores[:rows], workspace.candidate_rows[:rows]


def gather_dcp_queries(
    q: torch.Tensor,
    *,
    group: Group,
    workspace: DcpAttentionWorkspace,
) -> torch.Tensor:
    """All-gather local TP query heads for attention over one local KV shard."""

    _validate_group(group)
    if q.ndim != 3:
        raise ValueError(f"DCP q must have shape [rows, heads, dim], got {q.shape}")
    rows, local_heads, head_dim = q.shape
    local = workspace.local_q[:, :local_heads, :head_dim]
    if rows > local.shape[0]:
        raise ValueError("DCP query rows exceed the preallocated workspace")
    local[:rows].copy_(q)
    if rows < local.shape[0]:
        local[rows:].zero_()
    gathered = workspace.gathered_q[:, :, :local_heads, :head_dim]
    _all_gather_into_dcp_major(gathered, local, group)
    return gathered[:, :rows].permute(1, 0, 2, 3).reshape(rows, -1, head_dim)


def merge_dcp_attention_states(
    local_out: torch.Tensor,
    local_lse: torch.Tensor,
    *,
    group: Group,
    dcp_rank: int,
    local_output_heads: int,
    workspace: DcpAttentionWorkspace,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge natural-log attention states with LSE all-gather and output RS."""

    _validate_group(group)
    if local_out.ndim != 3 or local_lse.ndim != 2:
        raise ValueError("DCP partial state must be [rows, heads, dim] + [rows, heads]")
    if local_out.shape[:2] != local_lse.shape:
        raise ValueError("DCP output and LSE shapes disagree")
    if local_lse.dtype != torch.float32:
        raise TypeError(f"DCP LSE must be float32, got {local_lse.dtype}")
    if not 0 <= dcp_rank < len(group):
        raise ValueError(f"dcp_rank must be in [0, {len(group)}), got {dcp_rank}")
    rows, group_heads, head_dim = local_out.shape
    if group_heads != len(group) * local_output_heads:
        raise ValueError("DCP local attention must cover every gathered query head")
    if rows > workspace.local_lse.shape[0]:
        raise ValueError("DCP state rows exceed the preallocated workspace")

    staged_lse = workspace.local_lse[:, :group_heads]
    staged_lse[:rows].copy_(local_lse)
    if rows < staged_lse.shape[0]:
        staged_lse[rows:].fill_(-torch.inf)
    gathered_lse = workspace.gathered_lse[:, :, :group_heads]
    _all_gather_into_dcp_major(gathered_lse, staged_lse, group)

    merged_lse = workspace.merged_lse[:, :group_heads]
    rs_input = workspace.reduce_scatter_input[:, :, :local_output_heads, :head_dim]
    dcp_attn_merge_rs(
        local_out,
        gathered_lse,
        dcp_rank=dcp_rank,
        rs_staging=rs_input,
        merged_lse_out=merged_lse,
    )
    rs_output = workspace.reduce_scatter_output[:, :local_output_heads, :head_dim]
    _reduce_scatter_dcp_heads(rs_output, rs_input, group)
    head_start = dcp_rank * local_output_heads
    head_end = head_start + local_output_heads
    return rs_output[:rows], merged_lse[:rows, head_start:head_end]


def _all_gather_into_dcp_major(
    output: torch.Tensor,
    input: torch.Tensor,
    group: Group,
) -> None:
    expected = (len(group), *input.shape)
    if tuple(output.shape) != expected:
        raise ValueError(
            f"DCP collective output must have shape {expected}, got {tuple(output.shape)}"
        )
    process_group = pg_manager.get_process_group("nccl", group, role="dcp")
    dist.all_gather_into_tensor(output, input, group=process_group)


def _reduce_scatter_dcp_heads(
    output: torch.Tensor,
    input: torch.Tensor,
    group: Group,
) -> None:
    expected = (len(group), *output.shape)
    if tuple(input.shape) != expected:
        raise ValueError(
            f"DCP reduce-scatter input must have shape {expected}, got {tuple(input.shape)}"
        )
    process_group = pg_manager.get_process_group("nccl", group, role="dcp")
    dist.reduce_scatter_tensor(
        output,
        input.reshape(len(group) * output.shape[0], *output.shape[1:]),
        op=dist.ReduceOp.SUM,
        group=process_group,
    )


def _validate_group(group: Group) -> None:
    if len(group) <= 1:
        raise ValueError("DCP collective requires a group of at least two ranks")


def _validate_dcp_sharding(
    dcp_size: int,
    dcp_rank: int,
    interleave_size: int,
) -> None:
    if isinstance(dcp_size, bool) or not isinstance(dcp_size, int) or dcp_size <= 0:
        raise ValueError("DCP size must be a positive integer")
    if (
        isinstance(dcp_rank, bool)
        or not isinstance(dcp_rank, int)
        or not 0 <= dcp_rank < dcp_size
    ):
        raise ValueError("DCP rank must be an integer inside the DCP group")
    if (
        isinstance(interleave_size, bool)
        or not isinstance(interleave_size, int)
        or interleave_size <= 0
    ):
        raise ValueError("DCP interleave size must be a positive integer")


def shard_dcp_logical_rows(
    logical_rows: torch.Tensor,
    *,
    dcp_size: int,
    dcp_rank: int,
    interleave_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map global cache rows into one rank's interleaved physical row domain.

    Args:
        logical_rows: Global non-negative cache-row coordinates.
        dcp_size: Number of ranks sharing the history-token dimension.
        dcp_rank: Local rank inside the DCP group.
        interleave_size: Adjacent rows assigned to one owner before rotating.

    Returns:
        ``(local_rows, is_local)`` with shapes matching ``logical_rows``.
    """

    if (
        logical_rows.dtype == torch.bool
        or logical_rows.is_floating_point()
        or logical_rows.is_complex()
    ):
        raise TypeError("DCP logical rows must use an integer dtype")
    _validate_dcp_sharding(dcp_size, dcp_rank, interleave_size)

    rows = logical_rows.to(torch.int64)
    if rows.device.type == "cpu" and not bool(rows.ge(0).all().item()):
        raise ValueError("DCP logical rows must be non-negative")

    owner = torch.remainder(
        torch.div(rows, interleave_size, rounding_mode="floor"),
        dcp_size,
    )
    local_rows = torch.div(
        rows,
        dcp_size * interleave_size,
        rounding_mode="floor",
    ) * interleave_size + torch.remainder(rows, interleave_size)
    return local_rows, owner.eq(dcp_rank)


__all__ = [
    "DcpAttentionWorkspace",
    "DcpIndexerCandidateWorkspace",
    "dcp_owned_prefix_lengths",
    "gather_dcp_indexer_candidates",
    "gather_dcp_queries",
    "merge_dcp_attention_states",
    "restore_dcp_global_rows",
    "shard_dcp_logical_rows",
]
