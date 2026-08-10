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

"""Target-only breakable CUDA graphs for pipeline decode stages.

The generic decode graph wraps a whole executor step, which is invalid for
pipeline DSpark: control-plane collectives, P2P headers, PP7 sampling, the
PP7-to-PP0 context relay, and PP0 draft execution all have dynamic host-side
behavior. This runner instead captures only one rank's local target stage.

Adjacent activation transport remains eager and copies into a fixed stage-local
input before replay. Attention remains a breakable eager segment, so its live
cache metadata and ragged page tables are not frozen into the graph. The graph
therefore captures the dense K3/MoE compute around attention without capturing
any pipeline protocol operation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch

from tokenspeed.runtime.execution.breakable_cuda_graph import (
    BreakableCapture,
    active_forward,
)
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.cuda_graph_wrapper import (
    CudaGraphCaptureMode,
    get_batch_sizes_to_capture,
)
from tokenspeed.runtime.execution.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from tokenspeed.runtime.pipeline.contracts import StageActivation, StageOutput
from tokenspeed.runtime.pipeline.local_warmup import (
    make_pipeline_local_warmup_activation,
)
from tokenspeed.runtime.pipeline.model_runner_stage import PipelineForwardBatch
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.common import maybe_inference_mode

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.input_buffer import InputBuffers
    from tokenspeed.runtime.execution.model_executor import ModelExecutorConfig
    from tokenspeed.runtime.pipeline.contracts import StageModel


logger = get_colorful_logger(__name__)


@dataclass
class _CapturedStageDecode:
    """One exact-batch stage-local decode graph and its stable I/O."""

    ctx: ForwardContext
    batch: PipelineForwardBatch
    incoming: StageActivation | None
    capture: BreakableCapture
    output: StageOutput


class PipelineStageCudaGraphRunner:
    """Capture the target compute portion of one PP K3 DSpark stage.

    The runner intentionally supports only exact pure-decode buckets. PP
    padding would require synchronized dummy request rows and candidate blocks
    on every stage; keeping that separate avoids making the correctness path
    depend on the whole-step graph's padding machinery.
    """

    def __init__(
        self,
        *,
        stage: StageModel,
        attn_backend,
        token_to_kv_pool,
        input_buffers: InputBuffers,
        config: ModelExecutorConfig,
        prepare_capture_metadata: Callable[[int], None],
        num_warmup: int = 3,
    ) -> None:
        if stage.plan.stage_count < 2:
            raise ValueError("stage-local CUDA graphs require pipeline parallelism")
        if config.spec_algo != "DSPARK":
            raise ValueError("stage-local CUDA graphs currently require K3 DSpark")
        if config.output_length <= 0:
            raise ValueError("stage-local CUDA graphs require a positive decode width")
        if num_warmup < 1:
            raise ValueError("stage-local CUDA graph warmup count must be positive")

        self._stage = stage
        self._attn_backend = attn_backend
        self._token_to_kv_pool = token_to_kv_pool
        self._input_buffers = input_buffers
        self._config = config
        self._prepare_capture_metadata = prepare_capture_metadata
        self._num_warmup = num_warmup
        self._capture_batch_sizes = tuple(get_batch_sizes_to_capture(config))
        self._captures: dict[int, _CapturedStageDecode] = {}
        self._pool = None

    @property
    def capture_batch_sizes(self) -> tuple[int, ...]:
        return self._capture_batch_sizes

    def capture(self) -> None:
        """Capture every configured decode batch exactly once at startup.

        Capture failures are startup failures. Serving an eager-only PP DSpark
        candidate after graph capture was explicitly requested would make the
        performance envelope ambiguous and hide a graph-safety regression.
        """

        if not torch.cuda.is_available():
            raise RuntimeError("stage-local CUDA graph capture requires CUDA")
        if self._config.data_parallel_size != 1:
            raise RuntimeError(
                "stage-local K3 DSpark CUDA graphs currently require DP=1"
            )

        with maybe_inference_mode():
            for bs in self._capture_batch_sizes:
                self._captures[bs] = self._capture_one(bs)
        logger.info(
            "K3 PP stage-local decode graphs captured: stage=%d buckets=%s",
            self._stage.plan.stage_id,
            self._capture_batch_sizes,
        )

    def try_forward(
        self,
        context: object,
        batch: object,
        incoming: StageActivation | None,
    ) -> StageOutput | None:
        """Replay an exact decode graph, or return ``None`` for eager fallback."""

        if not isinstance(context, ForwardContext):
            return None
        if not isinstance(batch, PipelineForwardBatch):
            return None
        capture = self._captures.get(context.bs)
        if capture is None or not self._can_replay(context, batch, incoming, capture):
            return None

        if capture.incoming is not None:
            assert incoming is not None
            for destination, source in zip(
                capture.incoming.values,
                incoming.values,
                strict=True,
            ):
                destination.copy_(source)

        with active_forward(context):
            capture.capture.replay(valid_rows=context.input_num_tokens)
        return capture.output

    def _capture_one(self, bs: int) -> _CapturedStageDecode:
        ctx, batch, incoming = self._make_capture_inputs(bs)
        self._prepare_capture_metadata(bs)
        with active_forward(ctx):
            for _ in range(self._num_warmup):
                self._stage.forward_stage(ctx, batch, incoming)
            torch.cuda.synchronize()
            capture = BreakableCapture(
                pool=self._pool,
                capture_state_hook=CudaGraphCaptureMode(),
            )
            with capture:
                output = self._stage.forward_stage(ctx, batch, incoming)
            if self._pool is None:
                self._pool = capture.pool
            capture.replay(valid_rows=ctx.input_num_tokens)
        torch.cuda.synchronize()
        return _CapturedStageDecode(
            ctx=ctx,
            batch=batch,
            incoming=incoming,
            capture=capture,
            output=output,
        )

    def _make_capture_inputs(
        self,
        bs: int,
    ) -> tuple[ForwardContext, PipelineForwardBatch, StageActivation | None]:
        num_tokens = bs * self._config.output_length
        input_buffers = self._input_buffers
        input_buffers.input_ids_buf[:num_tokens].fill_(1)
        input_buffers.out_cache_loc_buf[:num_tokens].fill_(
            input_buffers.dummy_kv_slot
        )
        if self._config.model_is_mrope:
            input_buffers.mrope_positions_buf[:, :num_tokens].zero_()
            positions = input_buffers.mrope_positions_buf[:, :num_tokens]
        else:
            input_buffers.positions_buf[:num_tokens].zero_()
            positions = input_buffers.positions_buf[:num_tokens]
        input_buffers.req_pool_indices_buf[:bs].fill_(self._config.max_req_pool_size)
        input_buffers.seq_lens_buf[:bs].fill_(self._config.output_length)

        ctx = ForwardContext(
            attn_backend=self._attn_backend,
            token_to_kv_pool=self._token_to_kv_pool,
            bs=bs,
            num_extends=0,
            input_num_tokens=num_tokens,
            forward_mode=ForwardMode.DECODE,
            all_decode_or_idle=True,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )
        batch = PipelineForwardBatch(
            input_ids=input_buffers.input_ids_buf[:num_tokens],
            positions=positions,
            out_cache_loc=input_buffers.out_cache_loc_buf[:num_tokens],
            req_pool_indices=input_buffers.req_pool_indices_buf[:bs],
            seq_lens=input_buffers.seq_lens_buf[:bs],
        )
        incoming = (
            None
            if self._stage.plan.input_schema is None
            else make_pipeline_local_warmup_activation(
                self._stage.plan.input_schema,
                num_tokens=num_tokens,
                device=self._config.device,
            )
        )
        return ctx, batch, incoming

    def _can_replay(
        self,
        context: ForwardContext,
        batch: PipelineForwardBatch,
        incoming: StageActivation | None,
        capture: _CapturedStageDecode,
    ) -> bool:
        if (
            context.forward_mode is None
            or not context.forward_mode.is_decode()
            or context.num_extends != 0
            or context.capture_hidden_mode is not CaptureHiddenMode.FULL
            or context.input_num_tokens != context.bs * self._config.output_length
        ):
            return False
        if not self._same_static_tensor(batch.input_ids, capture.batch.input_ids):
            return False
        if not self._same_static_tensor(batch.positions, capture.batch.positions):
            return False
        if not self._same_static_tensor(
            batch.out_cache_loc,
            capture.batch.out_cache_loc,
        ):
            return False
        if capture.incoming is None:
            return incoming is None
        if incoming is None:
            return False
        try:
            self._stage.plan.input_schema.validate(incoming)
        except Exception:
            return False
        return all(
            source.shape == destination.shape and source.dtype == destination.dtype
            for source, destination in zip(
                incoming.values,
                capture.incoming.values,
                strict=True,
            )
        )

    @staticmethod
    def _same_static_tensor(live: object, captured: object) -> bool:
        return (
            isinstance(live, torch.Tensor)
            and isinstance(captured, torch.Tensor)
            and live.shape == captured.shape
            and live.dtype == captured.dtype
            and live.device == captured.device
            and live.data_ptr() == captured.data_ptr()
        )
