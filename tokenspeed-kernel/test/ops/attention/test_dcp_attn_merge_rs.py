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
import torch
from tokenspeed_kernel.ops.attention import dcp_attn_merge_rs

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)


def _reference(
    local_out: torch.Tensor,
    gathered_lse: torch.Tensor,
    *,
    dcp_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    dcp_size, rows, group_heads = gathered_lse.shape
    local_heads = group_heads // dcp_size
    clean_lse = torch.where(
        torch.isfinite(gathered_lse),
        gathered_lse,
        torch.full_like(gathered_lse, -torch.inf),
    )
    max_lse = clean_lse.amax(dim=0)
    safe_max = torch.nan_to_num(max_lse, nan=0.0, posinf=0.0, neginf=0.0)
    weights = torch.exp(clean_lse - safe_max.unsqueeze(0))
    denom = weights.sum(dim=0)
    merged_lse = torch.log(denom) + safe_max
    correction = weights[dcp_rank] / denom.clamp_min(torch.finfo(torch.float32).tiny)
    corrected = torch.where(
        correction.unsqueeze(-1) > 0,
        local_out.float() * correction.unsqueeze(-1),
        torch.zeros_like(local_out, dtype=torch.float32),
    )
    staging = corrected.view(rows, dcp_size, local_heads, local_out.shape[-1])
    return staging.permute(1, 0, 2, 3).to(local_out.dtype), merged_lse


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dcp_size", [2, 4])
def test_dcp_attn_merge_rs_matches_reference_and_destination_layout(
    dtype: torch.dtype,
    dcp_size: int,
) -> None:
    rows, local_heads, head_dim = 3, 32, 512
    group_heads = dcp_size * local_heads
    generator = torch.Generator(device="cuda").manual_seed(100 + dcp_size)
    local_out = torch.randn(
        rows,
        group_heads,
        head_dim,
        dtype=dtype,
        device="cuda",
        generator=generator,
    )
    gathered_lse = torch.randn(
        dcp_size,
        rows,
        group_heads,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    gathered_lse[0, 0, 0] = torch.nan
    gathered_lse[-1, 0, 1] = torch.inf
    gathered_lse[:, 1, 2] = -torch.inf
    local_out[1, 2].fill_(torch.nan)
    dcp_rank = dcp_size - 1
    rs_staging = torch.full(
        (dcp_size, rows, local_heads, head_dim),
        torch.nan,
        dtype=dtype,
        device="cuda",
    )
    merged_lse_out = torch.full(
        (rows, group_heads), torch.nan, dtype=torch.float32, device="cuda"
    )

    actual_staging, actual_lse = dcp_attn_merge_rs(
        local_out,
        gathered_lse,
        dcp_rank=dcp_rank,
        rs_staging=rs_staging,
        merged_lse_out=merged_lse_out,
        solution="triton",
    )
    expected_staging, expected_lse = _reference(
        local_out,
        gathered_lse,
        dcp_rank=dcp_rank,
    )

    assert actual_staging.data_ptr() == rs_staging.data_ptr()
    assert actual_lse.data_ptr() == merged_lse_out.data_ptr()
    torch.testing.assert_close(actual_staging, expected_staging, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(actual_lse, expected_lse, rtol=1e-5, atol=1e-5)
    assert actual_staging[0, 1, 2].eq(0).all()
    assert torch.isneginf(actual_lse[1, 2])


def test_dcp_attn_merge_rs_replays_with_caller_owned_buffers() -> None:
    dcp_size, rows, capacity_rows, local_heads, head_dim = 2, 2, 4, 32, 512
    group_heads = dcp_size * local_heads
    local_out = torch.empty(
        rows, group_heads, head_dim, dtype=torch.bfloat16, device="cuda"
    )
    gathered_lse = torch.empty(
        dcp_size, capacity_rows, group_heads, dtype=torch.float32, device="cuda"
    )
    rs_staging = torch.empty(
        dcp_size,
        capacity_rows,
        local_heads,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    merged_lse = torch.empty(
        capacity_rows, group_heads, dtype=torch.float32, device="cuda"
    )

    def run() -> None:
        dcp_attn_merge_rs(
            local_out,
            gathered_lse,
            dcp_rank=1,
            rs_staging=rs_staging,
            merged_lse_out=merged_lse,
            solution="triton",
        )

    local_out.normal_()
    gathered_lse.normal_()
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    for seed in range(4):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        local_out.normal_(generator=generator)
        gathered_lse.normal_(generator=generator)
        expected_staging, expected_lse = _reference(
            local_out,
            gathered_lse[:, :rows],
            dcp_rank=1,
        )
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            rs_staging[:, :rows], expected_staging, rtol=2e-3, atol=2e-3
        )
        assert rs_staging[:, rows:].eq(0).all()
        torch.testing.assert_close(
            merged_lse[:rows], expected_lse, rtol=1e-5, atol=1e-5
        )
