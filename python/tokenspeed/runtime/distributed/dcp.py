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

import torch


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
    if (
        isinstance(dcp_size, bool)
        or not isinstance(dcp_size, int)
        or dcp_size <= 0
    ):
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


__all__ = ["shard_dcp_logical_rows"]
