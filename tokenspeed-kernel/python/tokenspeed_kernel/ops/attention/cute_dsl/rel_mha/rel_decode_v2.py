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

"""Relative-bias decode v2 using the standalone TokenSpeed MHA decode kernel.

This implementation replaces the three-kernel sequence of the FA4-fork
forward pass, Triton shear, and split-KV combine with a single kernel. Its
CUTLASS ``gqa_decode`` data flow performs the K @ Q GEMM first, packs grouped
heads, fuses relative bias into the log2 online softmax, culls sliding-window
attention tiles, and performs a deterministic split-KV reduction in the
kernel. It consumes the compact per-head relative-bias table directly, without
a shear prepass.

The kernel handles serving page sizes 64, 128, and 256 natively. It assembles
K/V MMA tiles from subpage TMA boxes, allowing it to consume the engine's flat
per-step group table without reinterpreting the cache, expanding the page
table, or rewriting the table for each call. Two independent mechanisms make
``-1`` window holes safe; both are verified bit-for-bit and with Compute
Sanitizer. First, the page index is a bounds-checked TMA coordinate, so
out-of-range loads return zeros instead of faulting. Second, the in-kernel
window mask uses only index arithmetic to set out-of-window columns to
negative infinity, regardless of their loaded contents. The mask alone does
not neutralize NaNs, so the zero-fill is essential: an out-of-window table
entry must never point to real, potentially uninitialized memory.

TVM FFI compiles the kernel once per static configuration, including page
size. Runtime calls accept PyTorch tensors and dynamic batch sizes, page
counts, and sequence lengths. They are safe for CUDA graph capture because
``reduction_mode="kernel"`` rewrites the workspaces on every launch.
"""

from __future__ import annotations

import math
import os as _os

import torch

# Sliding-window attention scans at most five tiles, where splitting does not
# help. Full attention splits and reduces within the kernel (an A/B knob).
_V2_FULL_SPLITS = max(1, int(_os.environ.get("TSMHA_V2_FULL_SPLITS", "8")))


class _CompiledCache:
    def __init__(self):
        self._cache = {}

    def get(self, *key):
        if key not in self._cache:
            self._cache[key] = _compile_decode(*key)
        return self._cache[key]


_DECODE = _CompiledCache()
# Contiguous split-KV workspaces sized to the exact batch; see the allocation
# comment in the wrapper.
_WORKSPACES: dict = {}


def _prediction_tile(
    prediction: int, grouped_head_tile: int, window_left: int, dtype: torch.dtype
) -> int:
    """Choose query-token packing for each CTA.

    Sliding-window attention uses the kernel's packing model because short
    BF16 windows benefit from fewer output rows per CTA. Full attention uses
    the maximum legal packing: ``grouped_head_tile * tile <= 32``.
    """
    if prediction == 1:
        return 1
    if window_left >= 0:
        from .flash_fwd_sm100_bias_decode import (
            choose_swa_prediction_tile,
        )

        return choose_swa_prediction_tile(
            prediction,
            grouped_head_tile,
            window_left,
            bf16_eager=dtype == torch.bfloat16,
        )
    tile = 1 << (prediction - 1).bit_length()
    return min(32 // grouped_head_tile, tile)


def _compile_decode(
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    extent: int,
    window_left: int,
    kv_splits: int,
    dtype: torch.dtype,
    device_index: int,
    page_size: int,
    use_pdl: bool,
    prediction: int = 1,
):
    from cutlass import cute
    from flash_attn.cute.cute_dsl_utils import to_cute_tensor

    from .flash_fwd_sm100_bias_decode import (
        FlashAttentionDecodeSm100Bias,
    )

    grouped = num_q_heads // num_kv_heads
    grouped_head_tile = 1 << (grouped - 1).bit_length()  # next pow2
    kernel = FlashAttentionDecodeSm100Bias(
        head_dim,
        grouped_head_tile,
        prediction_tile=_prediction_tile(
            prediction, grouped_head_tile, window_left, dtype
        ),
        sequence_tile=256,
        reduction_mode="kernel",
        window_size_left=window_left if window_left >= 0 else None,
        page_size=page_size,
        use_pdl=use_pdl,
    )
    dev = torch.device("cuda", device_index)
    FB, FPAGES, P = 2, 8, prediction
    q = torch.empty(FB, P, num_q_heads, head_dim, device=dev, dtype=dtype)
    k = torch.empty(FPAGES, page_size, num_kv_heads, head_dim, device=dev, dtype=dtype)
    v = torch.empty_like(k)
    o = torch.empty(FB, P, num_q_heads, head_dim, device=dev, dtype=dtype)
    bias = torch.empty(FB, P, num_q_heads, extent, device=dev, dtype=dtype)
    op = torch.empty(
        kv_splits, FB, P, num_q_heads, head_dim, device=dev, dtype=torch.float32
    )
    mp = torch.empty(kv_splits, FB, P, num_q_heads, device=dev, dtype=torch.float32)
    lp = torch.empty_like(mp)
    pt = torch.empty(FB * 4, device=dev, dtype=torch.int32)
    po = torch.empty(FB, device=dev, dtype=torch.int32)
    su = torch.empty(FB, device=dev, dtype=torch.int32)
    return cute.compile(
        kernel,
        kv_splits,
        to_cute_tensor(q),
        to_cute_tensor(k),
        to_cute_tensor(v),
        None,  # q_sf
        None,  # k_sf
        None,  # v_sf
        to_cute_tensor(o),
        to_cute_tensor(bias),
        to_cute_tensor(op, assumed_align=4),
        to_cute_tensor(mp, assumed_align=4),
        to_cute_tensor(lp, assumed_align=4),
        1.0 / math.sqrt(head_dim),
        None,  # mCuSeqlensQ
        None,  # mCuSeqlensK
        None,  # mSeqUsedQ
        to_cute_tensor(su, assumed_align=4, leading_dim=0),
        None,  # max_seqlen_q
        None,  # max_seqlen_k
        to_cute_tensor(pt, assumed_align=4, leading_dim=0),
        to_cute_tensor(po, assumed_align=4, leading_dim=0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _ladder(B: int, cap: int = 64):
    b = 1
    while b < max(B, 1):
        b <<= 1
    sizes = {b}
    x = 1
    while x <= max(cap, b):
        sizes.add(x)
        x <<= 1
    return sorted(sizes)


def _alloc_workspaces(kv_splits, b, prediction, H, D, device):
    """Allocate the split-KV (O, M, L) partial-result workspace triple."""
    return (
        torch.empty(kv_splits, b, prediction, H, D, device=device, dtype=torch.float32),
        torch.empty(kv_splits, b, prediction, H, device=device, dtype=torch.float32),
        torch.empty(kv_splits, b, prediction, H, device=device, dtype=torch.float32),
    )


def _ensure_buffers(kv_splits, H, D, device, page_table, B, prediction=1):
    """Pre-allocate every per-B buffer OUTSIDE graph capture.

    Buffers first allocated INSIDE a capture come from the graph mempool,
    whose blocks are reused by other captures/replays — persistent state
    there gets clobbered (the bs>1 garbage root cause). The static table's
    base pointer is shared by every [:b] slice, so the whole pow2 ladder is
    pre-allocatable from any eager call."""
    P = prediction
    row_stride = page_table.stride(0)
    for b in _ladder(B):
        wkey = (kv_splits, H, D, device.index, b, P)
        if wkey not in _WORKSPACES:
            _WORKSPACES[wkey] = _alloc_workspaces(kv_splits, b, P, H, D, device)
        okey = (b, row_stride, device.index)
        if okey not in _OFFSETS:
            _OFFSETS[okey] = (
                torch.arange(b, device=device, dtype=torch.int32) * row_stride
            )


def v2_buffers_ready(q, window_left, page_table, prediction: int = 1) -> bool:
    """Capture-time gate: True iff every buffer this call needs already
    exists (so nothing would be allocated from the capture mempool)."""
    rows, H, D = q.shape
    B = rows // max(prediction, 1)
    kv_splits = 1 if window_left >= 0 else _V2_FULL_SPLITS
    if (kv_splits, H, D, q.device.index, B, prediction) not in _WORKSPACES:
        return False
    if (B, page_table.stride(0), q.device.index) not in _OFFSETS:
        return False
    return True


def is_compiled_for_v2(
    q, k_cache, rel_logits, window_left, enable_pdl: bool = True, prediction: int = 1
) -> bool:
    """Whether the config's kernel is already compiled (graph-capture guard)."""
    kv_splits = 1 if window_left >= 0 else _V2_FULL_SPLITS
    key = (
        q.shape[1],
        k_cache.shape[2],
        q.shape[-1],
        rel_logits.shape[-1],
        window_left if window_left >= 0 else -1,
        kv_splits,
        q.dtype,
        q.device.index,
        k_cache.shape[1],
        enable_pdl,
        prediction,
    )
    return key in _DECODE._cache


# Contents are a pure function of the key (write-once), so graphs may bake the addresses.
_OFFSETS: dict = {}


def _offsets_for(B: int, row_blocks: int, device) -> torch.Tensor:
    key = (B, row_blocks, device.index)
    off = _OFFSETS.get(key)
    if off is None:
        off = _OFFSETS[key] = (
            torch.arange(B, device=device, dtype=torch.int32) * row_blocks
        )
    return off


def rel_mha_decode_tsmha_v2(
    q: torch.Tensor,  # (B * prediction, num_q_heads, head_dim) bf16/fp16
    k_cache: torch.Tensor,  # (num_pages, page_tokens, num_kv_heads, head_dim)
    v_cache: torch.Tensor,
    page_table: torch.Tensor,  # (B, max_pages) int32, -1 holes
    cache_seqlens: torch.Tensor,  # (B,) int32
    rel_logits: torch.Tensor,  # (B * prediction, num_q_heads, extent)
    window_left: int,
    softmax_scale: float | None = None,
    enable_pdl: bool = True,
    prediction: int = 1,
) -> torch.Tensor:
    """Rel-bias decode on the standalone kernel (native multi-query).

    ``prediction`` is the kernel's native multi-token dimension: q carries
    ``prediction`` token rows per request (token-major), ``cache_seqlens``
    is per REQUEST (total length including all prediction tokens), and the
    kernel masks row t of request b to positions
    ``[.., seq_b - prediction + t]`` in-kernel — KV pages are read once
    per request, not once per row. 1 is plain decode; spec verify passes
    ``spec_num_tokens``.

    ``page_table`` is consumed at its NATIVE page granularity (64, 128, or
    256 tokens — the kernel's sub-page TMA boxes cover all three). It must
    be (a view of) the engine's address-stable per-step table with a dense
    innermost dim (column-sliced row strides are fine) and ``-1`` holes
    only outside the sliding window. Returns
    ``(B * prediction, num_q_heads * head_dim)`` in the query dtype.
    """
    P = max(prediction, 1)
    rows, H, D = q.shape
    B = rows // P
    assert rows == B * P, f"q rows {rows} not divisible by prediction {P}"
    assert cache_seqlens.shape[0] >= B and page_table.shape[0] >= B
    HK = k_cache.shape[2]
    page_tokens = k_cache.shape[1]
    assert page_tokens in (64, 128, 256), f"unsupported page size {page_tokens}"
    # page_table is a column-sliced view; fold row stride into offsets to consume it zero-copy.
    assert page_table.stride(-1) == 1, "page_table innermost dim must be dense"
    extent = rel_logits.shape[-1]
    kv_splits = 1 if window_left >= 0 else _V2_FULL_SPLITS
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)

    compiled = _DECODE.get(
        H,
        HK,
        D,
        extent,
        window_left if window_left >= 0 else -1,
        kv_splits,
        q.dtype,
        q.device.index,
        page_tokens,
        enable_pdl,
        P,
    )

    row_stride = page_table.stride(0)
    span = (B - 1) * row_stride + page_table.shape[1]
    flat_table = torch.as_strided(page_table, (span,), (1,))
    offsets = _offsets_for(B, row_stride, q.device)

    # Exact-B contiguous: sliced views mis-address the split reduction (dropped tail KV tiles).
    if not torch.cuda.is_current_stream_capturing():
        _ensure_buffers(kv_splits, H, D, q.device, page_table, B, P)
    key = (kv_splits, H, D, q.device.index, B, P)
    ws = _WORKSPACES.get(key)
    if ws is None:
        ws = _WORKSPACES[key] = _alloc_workspaces(kv_splits, B, P, H, D, q.device)
    op_w, mp_w, lp_w = ws
    o = torch.empty(B, P, H, D, device=q.device, dtype=q.dtype)

    compiled(
        kv_splits,
        q.view(B, P, H, D),
        k_cache,
        v_cache,
        None,
        None,
        None,
        o,
        rel_logits.view(B, P, H, extent),
        op_w,
        mp_w,
        lp_w,
        softmax_scale,
        None,
        None,
        None,
        cache_seqlens[:B],
        None,
        None,
        flat_table,
        offsets,
    )
    return o.view(B * P, H * D)
