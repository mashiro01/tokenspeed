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

"""Pipeline-stage-local startup warmup helpers.

Pipeline boot previously warmed one full forward through every stage. The
first stage compiled, then sent its activation to the next stage, which began
its own first compilation only after the prior one completed. These helpers
fabricate only the local input contract for non-first stages so every pipeline
stage can compile its model partition concurrently without exercising P2P or
the serving control plane.
"""

from __future__ import annotations

import torch

from tokenspeed.runtime.pipeline.contracts import ActivationSchema, StageActivation


DEFAULT_PIPELINE_LOCAL_WARMUP_MAX_TOKENS = 8192


def get_pipeline_local_warmup_token_sizes(
    *,
    chunked_prefill_size: int,
    max_tokens: int,
) -> tuple[int, ...]:
    """Return the bounded EXTEND signatures used to compile a PP stage locally."""

    if chunked_prefill_size <= 0 or max_tokens <= 0:
        return ()
    ceiling = min(int(chunked_prefill_size), int(max_tokens))
    return tuple(sorted({min(size, ceiling) for size in (1, 128, 2048, 8192)}))


def make_pipeline_local_warmup_activation(
    schema: ActivationSchema,
    *,
    num_tokens: int,
    device: torch.device | str,
) -> StageActivation:
    """Build a schema-valid synthetic activation for one local PP stage.

    The initial implementation intentionally supports token-major boundaries
    only. A different leading-rank contract needs a model-specific warmup
    adapter rather than guessing its dynamic dimensions.
    """

    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    values = []
    for field in schema.fields:
        if field.leading_rank != 1 or field.leading_shape_id not in (None, "tokens"):
            raise ValueError(
                "pipeline local warmup supports only token-major activation "
                f"fields, got {field.field_id!r} with leading_rank="
                f"{field.leading_rank} and leading_shape_id={field.leading_shape_id!r}"
            )
        dtype = getattr(torch, field.dtype.removeprefix("torch."), None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(
                f"pipeline local warmup does not recognize dtype {field.dtype!r} "
                f"for activation field {field.field_id!r}"
            )
        values.append(
            torch.zeros(
                (num_tokens, *field.trailing_shape),
                dtype=dtype,
                device=device,
            )
        )
    return schema.bind(values)
