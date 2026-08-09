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

"""Registration shim for gfx950 Gluon sigmoid top-k routing."""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

try:
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.routing import (
        invoke_sigmoid_bias_topk_route_gluon,
        invoke_sigmoid_bias_topk_route_prefill_gluon,
    )
except ImportError as exc:
    _IMPORT_ERROR = exc
    invoke_sigmoid_bias_topk_route_gluon = None
    invoke_sigmoid_bias_topk_route_prefill_gluon = None
else:
    _IMPORT_ERROR = None


if invoke_sigmoid_bias_topk_route_prefill_gluon is not None:

    @register_kernel(
        "moe",
        "sigmoid_bias_topk",
        name="gluon_sigmoid_bias_topk_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            "router_logits", "dense", {torch.float16, torch.bfloat16, torch.float32}
        ),
        priority=Priority.SPECIALIZED,
        tags={"prefill", "routing"},
    )
    def gluon_sigmoid_bias_topk_gfx950(
        *,
        router_logits: torch.Tensor,
        correction_bias: torch.Tensor,
        topk: int,
        routed_scaling_factor: float,
        normalize_topk_weights: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        route = (
            invoke_sigmoid_bias_topk_route_gluon
            if router_logits.shape[0] * topk <= 128
            else invoke_sigmoid_bias_topk_route_prefill_gluon
        )
        topk_ids, topk_weights = route(
            router_logits,
            correction_bias,
            topk,
            routed_scaling_factor=routed_scaling_factor,
            normalize_topk_weights=normalize_topk_weights,
        )
        return topk_weights, topk_ids

else:

    def gluon_sigmoid_bias_topk_gfx950(**kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        raise ImportError(
            "gluon_sigmoid_bias_topk_gfx950 requires tokenspeed-kernel-amd"
        ) from _IMPORT_ERROR


__all__ = ["gluon_sigmoid_bias_topk_gfx950"]
