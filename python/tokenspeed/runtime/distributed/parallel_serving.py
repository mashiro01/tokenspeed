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

"""Resolved serving topology for request- and context-parallel attention.

Legacy attention CP is a rank dimension in :class:`Mapping`. Decode context
parallelism (DCP) is deliberately different: it reuses ranks inside one TP
group, so it must never contribute to ``world_size`` or change model weight
sharding. Keeping the two concepts in one immutable plan prevents a backend
from accidentally treating the old ``ENABLE_CP`` skeleton as working DCP.
"""

from __future__ import annotations

from dataclasses import dataclass

from tokenspeed.runtime.distributed.mapping import Group, Mapping


@dataclass(frozen=True)
class ParallelServingPlan:
    """Validated topology consumed by schedulers, caches, and attention.

    Args:
        tensor_parallel_size: Model-compute TP width of one attention replica.
        decode_context_parallel_size: DCP width nested inside that TP group.
        cp_kv_cache_interleave_size: Consecutive tokens assigned to one DCP rank.
        legacy_attention_context_parallel_size: Independent legacy CP dimension.
    """

    tensor_parallel_size: int
    decode_context_parallel_size: int
    cp_kv_cache_interleave_size: int
    legacy_attention_context_parallel_size: int

    @classmethod
    def resolve(
        cls,
        mapping: Mapping,
        *,
        decode_context_parallel_size: int,
        cp_kv_cache_interleave_size: int,
    ) -> "ParallelServingPlan":
        dcp = _positive_int(
            "decode_context_parallel_size", decode_context_parallel_size
        )
        interleave = _positive_int(
            "cp_kv_cache_interleave_size", cp_kv_cache_interleave_size
        )
        tp = int(mapping.attn.tp_size)
        legacy_cp = int(mapping.attn.cp_size)
        if dcp > 1 and legacy_cp > 1:
            raise ValueError(
                "decode context parallelism cannot be combined with legacy "
                "attention context parallelism (ENABLE_CP); DCP is nested "
                "inside TP and does not use mapping.attn.cp"
            )
        if tp % dcp:
            raise ValueError(
                f"attn_tp_size ({tp}) must be divisible by "
                f"decode_context_parallel_size ({dcp})"
            )
        if dcp == 1 and interleave != 1:
            raise ValueError(
                "--cp-kv-cache-interleave-size only applies when DCP is enabled"
            )

        return cls(
            tensor_parallel_size=tp,
            decode_context_parallel_size=dcp,
            cp_kv_cache_interleave_size=interleave,
            legacy_attention_context_parallel_size=legacy_cp,
        )

    @property
    def has_dcp(self) -> bool:
        return self.decode_context_parallel_size > 1

    def dcp_group(self, mapping: Mapping) -> Group:
        """Return this rank's contiguous DCP subgroup inside its TP group."""

        tp_group = mapping.attn.tp_group
        if len(tp_group) != self.tensor_parallel_size:
            raise RuntimeError(
                "resolved DCP topology no longer matches attention TP group"
            )
        width = self.decode_context_parallel_size
        tp_index = tp_group.index(mapping.rank)
        start = tp_index // width * width
        return tuple(tp_group[start : start + width])

    def dcp_rank(self, mapping: Mapping) -> int:
        return self.dcp_group(mapping).index(mapping.rank)


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


__all__ = ["ParallelServingPlan"]
