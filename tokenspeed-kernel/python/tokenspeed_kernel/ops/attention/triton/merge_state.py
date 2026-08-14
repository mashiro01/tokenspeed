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

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


@triton.jit
def attn_merge_state_kernel(
    OutA,
    LseA,
    OutB,
    LseB,
    Out,
    Lse,
    head_dim: tl.constexpr,
    lse_scale_log2: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < head_dim
    value_offsets = row * head_dim + offs_d

    lse_a = tl.load(LseA + row).to(tl.float32)
    lse_b = tl.load(LseB + row).to(tl.float32)
    lse_a_log2 = lse_a * lse_scale_log2
    lse_b_log2 = lse_b * lse_scale_log2
    lse_max_log2 = tl.maximum(lse_a_log2, lse_b_log2)

    weight_a = tl.exp2(lse_a_log2 - lse_max_log2)
    weight_b = tl.exp2(lse_b_log2 - lse_max_log2)
    denom = weight_a + weight_b

    out_a = tl.load(OutA + value_offsets, mask=mask_d, other=0.0).to(tl.float32)
    out_b = tl.load(OutB + value_offsets, mask=mask_d, other=0.0).to(tl.float32)
    out = (out_a * weight_a + out_b * weight_b) / denom
    merged_lse = (lse_max_log2 + tl.log2(denom)) / lse_scale_log2

    tl.store(Out + value_offsets, out, mask=mask_d)
    tl.store(Lse + row, merged_lse)


@triton.jit
def dcp_attn_merge_rs_kernel(
    LocalOut,
    GatheredLse,
    RsStaging,
    MergedLse,
    active_rows: tl.constexpr,
    group_heads: tl.constexpr,
    local_heads: tl.constexpr,
    head_dim: tl.constexpr,
    dcp_size: tl.constexpr,
    dcp_rank: tl.constexpr,
    lse_stride_cp: tl.constexpr,
    lse_stride_row: tl.constexpr,
    staging_stride_dst: tl.constexpr,
    staging_stride_row: tl.constexpr,
    merged_lse_stride_row: tl.constexpr,
    BLOCK_CP: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row_head = tl.program_id(0)
    row = row_head // group_heads
    head = row_head % group_heads
    active = row < active_rows
    cp_offsets = tl.arange(0, BLOCK_CP)
    cp_mask = cp_offsets < dcp_size
    lse_offsets = cp_offsets * lse_stride_cp + row * lse_stride_row + head
    lse = tl.load(GatheredLse + lse_offsets, mask=cp_mask, other=-float("inf")).to(
        tl.float32
    )
    finite = (lse == lse) & (tl.abs(lse) < float("inf"))
    clean_lse = tl.where(finite & cp_mask, lse, -float("inf"))
    max_lse = tl.max(clean_lse, axis=0)
    safe_max = tl.where(max_lse > -float("inf"), max_lse, 0.0)
    weights = tl.exp(clean_lse - safe_max)
    denom = tl.sum(weights, axis=0)
    merged_lse = tl.log(denom) + safe_max
    source_weight = tl.sum(
        tl.where(cp_offsets == dcp_rank, weights, 0.0), axis=0
    ) / tl.maximum(denom, 1.1754943508222875e-38)
    tl.store(MergedLse + row * merged_lse_stride_row + head, merged_lse)

    destination = head // local_heads
    local_head = head % local_heads
    dim_offsets = tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < head_dim
    local_out_offsets = (row * group_heads + head) * head_dim + dim_offsets
    out = tl.load(LocalOut + local_out_offsets, mask=active & dim_mask, other=0.0).to(
        tl.float32
    )
    staging_offsets = (
        destination * staging_stride_dst
        + row * staging_stride_row
        + local_head * head_dim
        + dim_offsets
    )
    tl.store(
        RsStaging + staging_offsets,
        tl.where(active & (source_weight > 0.0), out * source_weight, 0.0),
        mask=dim_mask,
    )


@register_kernel(
    "attention",
    "attn_merge_state",
    name="triton_attn_merge_state",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=format_signatures(
        ("out_a", "out_b"), "dense", {torch.float16, torch.bfloat16}
    ),
    priority=Priority.PORTABLE,
    traits={},
    tags={"portability"},
)
def triton_attn_merge_state(
    out_a: torch.Tensor,
    lse_a: torch.Tensor,
    out_b: torch.Tensor,
    lse_b: torch.Tensor,
    lse_scale_log2: float,
    inplace: bool = False,
    enable_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    del enable_pdl
    out = out_a if inplace else torch.empty_like(out_a)
    lse = lse_a if inplace else torch.empty_like(lse_a)
    total_rows = out_a.shape[0] * out_a.shape[1]
    head_dim = out_a.shape[2]
    block_d = triton.next_power_of_2(head_dim)
    attn_merge_state_kernel[(total_rows,)](
        out_a,
        lse_a,
        out_b,
        lse_b,
        out,
        lse,
        head_dim,
        float(lse_scale_log2),
        BLOCK_D=block_d,
    )
    return out, lse


@register_kernel(
    "attention",
    "dcp_attn_merge_rs",
    name="triton_dcp_attn_merge_rs",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=format_signatures("local_out", "dense", {torch.float16, torch.bfloat16}),
    priority=Priority.PORTABLE,
    traits={},
    tags={"determinism", "portability"},
)
def triton_dcp_attn_merge_rs(
    local_out: torch.Tensor,
    gathered_lse: torch.Tensor,
    *,
    dcp_rank: int,
    rs_staging: torch.Tensor,
    merged_lse_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse DCP LSE merge and local-output correction into RS staging."""

    dcp_size, _, group_heads = gathered_lse.shape
    active_rows = local_out.shape[0]
    capacity_rows = rs_staging.shape[1]
    local_heads = group_heads // dcp_size
    head_dim = local_out.shape[-1]
    dcp_attn_merge_rs_kernel[(capacity_rows * group_heads,)](
        local_out,
        gathered_lse,
        rs_staging,
        merged_lse_out,
        active_rows=active_rows,
        group_heads=group_heads,
        local_heads=local_heads,
        head_dim=head_dim,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
        lse_stride_cp=gathered_lse.stride(0),
        lse_stride_row=gathered_lse.stride(1),
        staging_stride_dst=rs_staging.stride(0),
        staging_stride_row=rs_staging.stride(1),
        merged_lse_stride_row=merged_lse_out.stride(0),
        BLOCK_CP=triton.next_power_of_2(dcp_size),
        BLOCK_D=triton.next_power_of_2(head_dim),
    )
    return rs_staging, merged_lse_out
