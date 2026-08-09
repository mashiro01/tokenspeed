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

import pytest

latent_input_small_batch = pytest.importorskip(
    "tokenspeed_kernel_amd.ops.gfx950.moe.fp16.latent_input_small_batch",
    reason="tokenspeed-kernel-amd is required for Gluon latent input tests",
)


@pytest.mark.parametrize("hidden_size", [64, 128, 192, 256, 7168])
def test_split_k_covers_every_tile(hidden_size: int) -> None:
    split_k = latent_input_small_batch._split_k(
        tokens=2, total_n=6016, hidden=hidden_size, block_m=16
    )
    assert (hidden_size // latent_input_small_batch._BLOCK_K) % split_k == 0


def test_split_k_does_not_drop_k_tiles() -> None:
    assert (
        latent_input_small_batch._split_k(
            tokens=2, total_n=6016, hidden=192, block_m=16
        )
        == 1
    )
