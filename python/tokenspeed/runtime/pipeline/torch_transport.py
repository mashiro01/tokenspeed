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

"""Torch distributed transport for static pipeline activations."""

from __future__ import annotations

import math

import torch
import torch.distributed as dist

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.distributed.qualification_events import (
    emit_pipeline_plan_consensus_success,
)
from tokenspeed.runtime.pipeline.contracts import (
    ActivationSchema,
    PipelineForwardMode,
    PipelinePlan,
    PipelineProtocolError,
    StageActivation,
)
from tokenspeed.runtime.pipeline.groups import PIPELINE_RESULT_GROUP_ROLE
from tokenspeed.runtime.pipeline.torch_control import (
    PipelineStepLease,
    TorchPipelineControlPlane,
)
from tokenspeed.runtime.pipeline.wire import (
    ACTIVATION_HEADER_WORDS,
    RESULT_HEADER_WORDS,
    ActivationWireHeader,
    ResultWireHeader,
)

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}
if hasattr(torch, "float8_e4m3fn"):
    _DTYPES["float8_e4m3fn"] = torch.float8_e4m3fn


def validate_pipeline_plan_consensus(plan: PipelinePlan, mapping: Mapping) -> None:
    """Fail startup unless every global rank constructed the same plan."""

    if mapping.world_size > 1:
        local = torch.tensor(list(bytes.fromhex(plan.digest)), dtype=torch.uint8)
        gathered = [torch.empty_like(local) for _ in range(mapping.world_size)]
        group = pg_manager.get_process_group("gloo", mapping.world_group)
        dist.all_gather(gathered, local, group=group)
        divergent = [
            rank
            for rank, digest in enumerate(gathered)
            if not torch.equal(digest, local)
        ]
        if divergent:
            raise PipelineProtocolError(
                f"pipeline plan digest differs on global ranks {divergent}"
            )

    emit_pipeline_plan_consensus_success(
        mapping,
        pipeline_plan_digest=plan.digest,
    )


class TorchPipelineTransport:
    """Send a fixed-schema payload along one rank's pipeline lane.

    Runtime leading dimensions travel in a fixed-length Gloo header. Tensor
    bytes travel directly over the lane's NCCL process group; no Python object
    serialization or variable field list is accepted.
    """

    def __init__(
        self,
        mapping: Mapping,
        *,
        device: torch.device | str,
        control: TorchPipelineControlPlane,
        max_leading_dimension: int = 1 << 20,
        max_tensor_elements: int = 1 << 31,
        max_decode_receive_cache_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        if mapping.pipeline.stage_count < 2:
            raise ValueError("TorchPipelineTransport requires at least two stages")
        if max_leading_dimension < 1 or max_tensor_elements < 1:
            raise ValueError("pipeline activation limits must be positive")
        if max_decode_receive_cache_bytes < 0:
            raise ValueError("decode receive cache limit must not be negative")
        self._mapping = mapping
        self._device = torch.device(device)
        self._control = control
        self._max_leading_dimension = max_leading_dimension
        self._max_tensor_elements = max_tensor_elements
        self._max_decode_receive_cache_bytes = max_decode_receive_cache_bytes
        self._decode_receive_buffers: dict[
            tuple[str, tuple[int, ...]], tuple[torch.Tensor, ...]
        ] = {}
        self._decode_receive_buffer_bytes = 0
        self._receive_header = torch.empty(ACTIVATION_HEADER_WORDS, dtype=torch.int64)
        group = mapping.pipeline.pipeline_group
        self._cpu_group = pg_manager.get_process_group("gloo", group)
        self._device_group = pg_manager.get_process_group("nccl", group)

    def send(
        self,
        step: PipelineStepLease,
        schema: ActivationSchema,
        activation: StageActivation,
    ) -> None:
        destination = self._mapping.pipeline.next_rank
        if destination is None:
            raise PipelineProtocolError(
                "the final pipeline stage cannot send activation"
            )
        schema.validate(activation)
        leading_dimensions = []
        for field, value in zip(schema.fields, activation.values):
            if not isinstance(value, torch.Tensor):
                raise PipelineProtocolError(
                    f"field {field.field_id} must be a torch.Tensor"
                )
            if value.device != self._device:
                raise PipelineProtocolError(
                    f"field {field.field_id} is on {value.device}, expected {self._device}"
                )
            leading = tuple(value.shape[: field.leading_rank])
            if (
                field.leading_shape_id == "tokens"
                and leading
                and leading[0] != step.descriptor.input_num_tokens
            ):
                raise PipelineProtocolError(
                    f"field {field.field_id} token dimension disagrees with the step"
                )
            leading_dimensions.extend(leading)
        self._validate_leading_dimensions(leading_dimensions)
        wire_header = ActivationWireHeader.for_activation(
            step=step.descriptor,
            source_stage=self._mapping.pipeline.stage_index,
            destination_stage=self._mapping.pipeline.stage_index + 1,
            schema=schema,
            activation=activation,
        )
        header = torch.tensor(wire_header.pack(), dtype=torch.int64)
        header_work = dist.isend(header, dst=destination, group=self._cpu_group)
        self._control.wait_work(header_work, step, "activation-header-send")
        payloads = [
            (value.contiguous(), field.field_id)
            for field, value in zip(schema.fields, activation.values)
        ]
        # Eager NCCL serializes independent P2P calls on one process group.
        # Submit every field of this activation as one batch before waiting.
        works = (
            dist.batch_isend_irecv(
                [
                    dist.P2POp(
                        dist.isend,
                        payload,
                        destination,
                        group=self._device_group,
                    )
                    for payload, _field_id in payloads
                ]
            )
            if payloads
            else []
        )
        self._wait_payload_works(
            works,
            step,
            direction="send",
            field_ids=[field_id for _payload, field_id in payloads],
        )

    def receive(
        self,
        step: PipelineStepLease,
        schema: ActivationSchema,
    ) -> StageActivation:
        source = self._mapping.pipeline.prev_rank
        if source is None:
            raise PipelineProtocolError(
                "the first pipeline stage cannot receive activation"
            )
        header_work = dist.irecv(
            self._receive_header,
            src=source,
            group=self._cpu_group,
        )
        self._control.wait_work(header_work, step, "activation-header-receive")
        wire_header = ActivationWireHeader.validate_and_unpack(
            self._receive_header.tolist(),
            expected_step=step.descriptor,
            expected_schema=schema,
            expected_source_stage=self._mapping.pipeline.stage_index - 1,
            expected_destination_stage=self._mapping.pipeline.stage_index,
        )
        leading_dimensions = wire_header.leading_dimensions
        expected_leading_count = sum(field.leading_rank for field in schema.fields)
        if len(leading_dimensions) != expected_leading_count:
            raise PipelineProtocolError(
                "activation header leading dimensions do not match schema"
            )
        self._validate_leading_dimensions(leading_dimensions)

        fields = []
        cursor = 0
        total_elements = 0
        leading_shapes: dict[str, tuple[int, ...]] = {}
        for field in schema.fields:
            leading = tuple(leading_dimensions[cursor : cursor + field.leading_rank])
            cursor += field.leading_rank
            if (
                field.leading_shape_id == "tokens"
                and leading
                and leading[0] != step.descriptor.input_num_tokens
            ):
                raise PipelineProtocolError(
                    f"field {field.field_id} token dimension disagrees with the step"
                )
            if field.leading_shape_id is not None:
                previous = leading_shapes.setdefault(field.leading_shape_id, leading)
                if previous != leading:
                    raise PipelineProtocolError(
                        f"field {field.field_id} has inconsistent leading shape"
                    )
            shape = (*leading, *field.trailing_shape)
            if math.prod(shape) > self._max_tensor_elements:
                raise PipelineProtocolError(
                    f"field {field.field_id} exceeds the activation element limit"
                )
            dtype = _DTYPES.get(field.dtype)
            if dtype is None:
                raise PipelineProtocolError(
                    f"field {field.field_id} uses unsupported dtype {field.dtype!r}"
                )
            total_elements += math.prod(shape)
            fields.append((field, shape, dtype))
        if total_elements != wire_header.total_elements:
            raise PipelineProtocolError(
                "activation header total element count does not match payload"
            )
        values = self._receive_payload_buffers(
            step=step,
            schema=schema,
            leading_dimensions=leading_dimensions,
            fields=fields,
        )

        payloads = list(zip(values, schema.fields, strict=True))
        works = (
            dist.batch_isend_irecv(
                [
                    dist.P2POp(
                        dist.irecv,
                        value,
                        source,
                        group=self._device_group,
                    )
                    for value, _field in payloads
                ]
            )
            if payloads
            else []
        )
        self._wait_payload_works(
            works,
            step,
            direction="receive",
            field_ids=[field.field_id for _value, field in payloads],
        )
        return schema.bind(values)

    def _receive_payload_buffers(
        self,
        *,
        step: PipelineStepLease,
        schema: ActivationSchema,
        leading_dimensions: tuple[int, ...],
        fields: list[tuple[object, tuple[int, ...], torch.dtype]],
    ) -> list[torch.Tensor]:
        """Allocate or reuse exact-shape payload buffers for steady decode.

        Decode repeatedly traverses a small set of batch buckets. Keeping the
        exact tensors alive makes the NCCL receive addresses stable and removes
        per-step CUDA allocation from the pipeline critical path. Prefill is
        deliberately excluded because its token dimensions can be very large
        and highly variable.
        """

        cache_key = (schema.digest, leading_dimensions)
        use_cache = (
            step.descriptor.forward_mode is PipelineForwardMode.DECODE
            and self._max_decode_receive_cache_bytes > 0
        )
        if use_cache:
            cached = self._decode_receive_buffers.get(cache_key)
            if cached is not None:
                return list(cached)

        payload_bytes = sum(
            math.prod(shape) * torch.empty((), dtype=dtype).element_size()
            for _field, shape, dtype in fields
        )
        values = [
            torch.empty(shape, dtype=dtype, device=self._device)
            for _field, shape, dtype in fields
        ]
        if (
            use_cache
            and payload_bytes <= self._max_decode_receive_cache_bytes
            and self._decode_receive_buffer_bytes + payload_bytes
            <= self._max_decode_receive_cache_bytes
        ):
            self._decode_receive_buffers[cache_key] = tuple(values)
            self._decode_receive_buffer_bytes += payload_bytes
        return values

    def _wait_payload_works(
        self,
        works,
        step: PipelineStepLease,
        *,
        direction: str,
        field_ids: list[str],
    ) -> None:
        if not field_ids:
            if works:
                raise PipelineProtocolError(
                    f"activation payload {direction} returned work for no fields"
                )
            return

        # NCCL coalescing returns one aggregate Work for a whole P2P batch,
        # whereas non-coalescing backends return one Work per P2POp.
        if len(works) == 1:
            self._control.wait_work(
                works[0],
                step,
                f"activation-payload-{direction}-batch",
            )
            return
        if len(works) != len(field_ids):
            raise PipelineProtocolError(
                f"activation payload {direction} returned {len(works)} work items "
                f"for {len(field_ids)} fields"
            )
        for work, field_id in zip(works, field_ids, strict=True):
            self._control.wait_work(
                work,
                step,
                f"activation-payload-{direction}:{field_id}",
            )

    def _validate_leading_dimensions(self, dimensions) -> None:
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value > self._max_leading_dimension
            for value in dimensions
        ):
            raise PipelineProtocolError(
                "pipeline activation leading dimension is outside the configured limit"
            )

    def close(self) -> None:
        self._decode_receive_buffers.clear()
        self._decode_receive_buffer_bytes = 0
        return None


class TorchPipelineResultSynchronizer:
    """Broadcast target sampling results back along every PP lane.

    ``accept_lengths`` and NaN flags are one value per request. ``output_tokens``
    can instead hold a full speculative verify window, so its explicit count
    is part of the caller contract rather than inferred from ``batch_size``.
    """

    _REQUEST_FIELDS = 2

    def __init__(
        self,
        mapping: Mapping,
        *,
        device: torch.device | str,
        control: TorchPipelineControlPlane,
    ) -> None:
        if mapping.pipeline.stage_count < 2:
            raise ValueError(
                "TorchPipelineResultSynchronizer requires at least two stages"
            )
        self._mapping = mapping
        self._device = torch.device(device)
        self._control = control
        self._group = pg_manager.get_process_group(
            "nccl",
            mapping.pipeline.pipeline_group,
            role=PIPELINE_RESULT_GROUP_ROLE,
        )
        self._source = mapping.pipeline.pipeline_group[-1]

    def synchronize(
        self,
        *,
        step: PipelineStepLease,
        batch_size: int,
        output_tokens: torch.Tensor | None = None,
        accept_lengths: torch.Tensor | None = None,
        nan_flags: torch.Tensor | None = None,
        output_token_count: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise ValueError("pipeline result batch_size must be an integer")
        if batch_size < 0:
            raise ValueError("pipeline result batch_size must be non-negative")
        if step.descriptor.batch_size != batch_size:
            raise PipelineProtocolError(
                "pipeline result batch_size disagrees with the active step"
            )
        if output_token_count is None:
            output_token_count = batch_size
        if (
            isinstance(output_token_count, bool)
            or not isinstance(output_token_count, int)
            or output_token_count < 0
        ):
            raise ValueError("pipeline output_token_count must be non-negative")
        packet = torch.empty(
            RESULT_HEADER_WORDS
            + output_token_count
            + self._REQUEST_FIELDS * batch_size,
            dtype=torch.int64,
            device=self._device,
        )
        payload = packet[RESULT_HEADER_WORDS:]
        if self._mapping.pipeline.is_last_stage:
            if output_tokens is None or accept_lengths is None:
                raise PipelineProtocolError(
                    "the final pipeline stage must publish sampling tensors"
                )
            tensors = (output_tokens, accept_lengths)
            if (
                output_tokens.numel() != output_token_count
                or accept_lengths.numel() != batch_size
            ):
                raise PipelineProtocolError(
                    "pipeline sampling tensors disagree with their declared output "
                    "token and request counts"
                )
            packet[:RESULT_HEADER_WORDS].copy_(
                torch.tensor(
                    ResultWireHeader(
                        step.descriptor,
                        source_stage=self._mapping.pipeline.stage_count - 1,
                    ).pack(),
                    dtype=torch.int64,
                    device=self._device,
                )
            )
            payload[:output_token_count].copy_(output_tokens.reshape(-1).to(torch.int64))
            payload[
                output_token_count : output_token_count + batch_size
            ].copy_(
                accept_lengths.reshape(-1).to(torch.int64)
            )
            if nan_flags is None:
                payload[output_token_count + batch_size :].zero_()
            else:
                if nan_flags.numel() != batch_size:
                    raise PipelineProtocolError(
                        "pipeline NaN flags must contain one value per request"
                    )
                payload[output_token_count + batch_size :].copy_(
                    nan_flags.reshape(-1).to(torch.int64)
                )
        work = dist.broadcast(
            packet,
            src=self._source,
            group=self._group,
            async_op=True,
        )
        self._control.wait_work(work, step, "result-broadcast")
        ResultWireHeader.validate(
            packet[:RESULT_HEADER_WORDS].to(device="cpu").tolist(),
            expected_step=step.descriptor,
            expected_source_stage=self._mapping.pipeline.stage_count - 1,
        )
        return (
            payload[:output_token_count].to(torch.int32),
            payload[
                output_token_count : output_token_count + batch_size
            ].to(torch.int32),
            payload[output_token_count + batch_size :].to(torch.int32),
        )


class TorchPipelineDSparkSynchronizer:
    """Move K3 DSpark's PP context and next candidates without replication.

    Target stages accumulate one FP32 context activation as it travels forward.
    PP7 owns verification and sends that context directly to PP0, the only
    rank that owns the draft. PP0 broadcasts the next candidate block after
    drafting so every target stage builds an identical next verify batch.
    """

    _CONTEXT_FLAG = 1
    _CANDIDATES_FLAG = 2
    _CANDIDATES_AND_WIDTHS_FLAG = 3
    _FIELD_COUNT = 1
    _CANDIDATES_AND_WIDTHS_FIELD_COUNT = 2

    def __init__(
        self,
        mapping: Mapping,
        *,
        device: torch.device | str,
        control: TorchPipelineControlPlane,
        context_hidden_size: int,
        candidate_width: int,
    ) -> None:
        if mapping.pipeline.stage_count < 2:
            raise ValueError("K3 DSpark synchronizer requires pipeline parallelism")
        if context_hidden_size < 1 or candidate_width < 1:
            raise ValueError("K3 DSpark context and candidate widths must be positive")
        self._mapping = mapping
        self._device = torch.device(device)
        self._control = control
        self._context_hidden_size = int(context_hidden_size)
        self._candidate_width = int(candidate_width)
        group = mapping.pipeline.pipeline_group
        self._cpu_group = pg_manager.get_process_group("gloo", group)
        self._device_group = pg_manager.get_process_group(
            "nccl",
            group,
            role=PIPELINE_RESULT_GROUP_ROLE,
        )
        self._draft_owner = group[0]
        self._verify_owner = group[-1]

    def relay_context(
        self,
        *,
        step: PipelineStepLease,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Send the final stage's FP32 context directly to the PP0 draft."""

        pipeline = self._mapping.pipeline
        if pipeline.is_last_stage:
            if context is None:
                raise PipelineProtocolError("PP7 must publish the DSpark context")
            expected_shape = (
                step.descriptor.input_num_tokens,
                self._context_hidden_size,
            )
            if (
                tuple(context.shape) != expected_shape
                or context.dtype != torch.float32
                or context.device != self._device
            ):
                raise PipelineProtocolError(
                    "K3 DSpark context must be a local FP32 tensor with shape "
                    f"{expected_shape}, got shape={tuple(context.shape)} "
                    f"dtype={context.dtype} device={context.device}"
                )
            header = torch.tensor(
                ResultWireHeader(
                    step.descriptor,
                    source_stage=pipeline.stage_count - 1,
                    field_count=self._FIELD_COUNT,
                    flags=self._CONTEXT_FLAG,
                ).pack(),
                dtype=torch.int64,
            )
            header_work = dist.isend(
                header,
                dst=self._draft_owner,
                group=self._cpu_group,
            )
            self._control.wait_work(header_work, step, "dspark-context-header-send")
            payload_work = dist.isend(
                context.contiguous(),
                dst=self._draft_owner,
                group=self._device_group,
            )
            self._control.wait_work(payload_work, step, "dspark-context-send")
            return None

        if not pipeline.is_first_stage:
            return None

        header = torch.empty(RESULT_HEADER_WORDS, dtype=torch.int64)
        header_work = dist.irecv(
            header,
            src=self._verify_owner,
            group=self._cpu_group,
        )
        self._control.wait_work(header_work, step, "dspark-context-header-receive")
        ResultWireHeader.validate(
            header.tolist(),
            expected_step=step.descriptor,
            expected_source_stage=pipeline.stage_count - 1,
            expected_field_count=self._FIELD_COUNT,
            expected_flags=self._CONTEXT_FLAG,
        )
        received = torch.empty(
            (step.descriptor.input_num_tokens, self._context_hidden_size),
            dtype=torch.float32,
            device=self._device,
        )
        payload_work = dist.irecv(
            received,
            src=self._verify_owner,
            group=self._device_group,
        )
        self._control.wait_work(payload_work, step, "dspark-context-receive")
        return received

    def broadcast_candidates(
        self,
        *,
        step: PipelineStepLease,
        candidates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Broadcast PP0's next verify candidates as a checked fixed packet."""

        batch_size = step.descriptor.batch_size
        packet = torch.empty(
            RESULT_HEADER_WORDS + batch_size * self._candidate_width,
            dtype=torch.int64,
            device=self._device,
        )
        if self._mapping.pipeline.is_first_stage:
            if candidates is None:
                raise PipelineProtocolError("PP0 must publish K3 DSpark candidates")
            expected_shape = (batch_size, self._candidate_width)
            if tuple(candidates.shape) != expected_shape:
                raise PipelineProtocolError(
                    "K3 DSpark candidates must have shape "
                    f"{expected_shape}, got {tuple(candidates.shape)}"
                )
            packet[:RESULT_HEADER_WORDS].copy_(
                torch.tensor(
                    ResultWireHeader(
                        step.descriptor,
                        source_stage=0,
                        field_count=self._FIELD_COUNT,
                        flags=self._CANDIDATES_FLAG,
                    ).pack(),
                    dtype=torch.int64,
                    device=self._device,
                )
            )
            packet[RESULT_HEADER_WORDS:].copy_(candidates.reshape(-1).to(torch.int64))
        work = dist.broadcast(
            packet,
            src=self._draft_owner,
            group=self._device_group,
            async_op=True,
        )
        self._control.wait_work(work, step, "dspark-candidate-broadcast")
        ResultWireHeader.validate(
            packet[:RESULT_HEADER_WORDS].to(device="cpu").tolist(),
            expected_step=step.descriptor,
            expected_source_stage=0,
            expected_field_count=self._FIELD_COUNT,
            expected_flags=self._CANDIDATES_FLAG,
        )
        return packet[RESULT_HEADER_WORDS:].to(torch.int32).view(
            batch_size,
            self._candidate_width,
        )

    def broadcast_candidates_and_widths(
        self,
        *,
        step: PipelineStepLease,
        candidates: torch.Tensor | None = None,
        verify_widths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Broadcast PP0's candidate rows and next verify widths atomically.

        Candidate ids and widths describe the same future target forward. They
        must travel in one packet so a delayed control message can never pair a
        fresh candidate block with a stale ragged layout on another PP stage.
        """

        batch_size = step.descriptor.batch_size
        candidate_elements = batch_size * self._candidate_width
        packet = torch.empty(
            RESULT_HEADER_WORDS + candidate_elements + batch_size,
            dtype=torch.int64,
            device=self._device,
        )
        if self._mapping.pipeline.is_first_stage:
            if candidates is None or verify_widths is None:
                raise PipelineProtocolError(
                    "PP0 must publish K3 DSpark candidates and verify widths"
                )
            expected_candidates = (batch_size, self._candidate_width)
            if tuple(candidates.shape) != expected_candidates:
                raise PipelineProtocolError(
                    "K3 DSpark candidates must have shape "
                    f"{expected_candidates}, got {tuple(candidates.shape)}"
                )
            if tuple(verify_widths.shape) != (batch_size,):
                raise PipelineProtocolError(
                    "K3 DSpark verify widths must have shape "
                    f"{(batch_size,)}, got {tuple(verify_widths.shape)}"
                )
            packet[:RESULT_HEADER_WORDS].copy_(
                torch.tensor(
                    ResultWireHeader(
                        step.descriptor,
                        source_stage=0,
                        field_count=self._CANDIDATES_AND_WIDTHS_FIELD_COUNT,
                        flags=self._CANDIDATES_AND_WIDTHS_FLAG,
                    ).pack(),
                    dtype=torch.int64,
                    device=self._device,
                )
            )
            payload = packet[RESULT_HEADER_WORDS:]
            payload[:candidate_elements].copy_(
                candidates.reshape(-1).to(torch.int64)
            )
            payload[candidate_elements:].copy_(verify_widths.to(torch.int64))
        work = dist.broadcast(
            packet,
            src=self._draft_owner,
            group=self._device_group,
            async_op=True,
        )
        self._control.wait_work(work, step, "dspark-candidate-width-broadcast")
        ResultWireHeader.validate(
            packet[:RESULT_HEADER_WORDS].to(device="cpu").tolist(),
            expected_step=step.descriptor,
            expected_source_stage=0,
            expected_field_count=self._CANDIDATES_AND_WIDTHS_FIELD_COUNT,
            expected_flags=self._CANDIDATES_AND_WIDTHS_FLAG,
        )
        payload = packet[RESULT_HEADER_WORDS:]
        candidate_rows = payload[:candidate_elements].to(torch.int32).view(
            batch_size,
            self._candidate_width,
        )
        widths = payload[candidate_elements:].to(torch.int32)
        return candidate_rows, widths
