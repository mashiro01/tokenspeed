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
    ) -> None:
        if mapping.pipeline.stage_count < 2:
            raise ValueError("TorchPipelineTransport requires at least two stages")
        if max_leading_dimension < 1 or max_tensor_elements < 1:
            raise ValueError("pipeline activation limits must be positive")
        self._mapping = mapping
        self._device = torch.device(device)
        self._control = control
        self._max_leading_dimension = max_leading_dimension
        self._max_tensor_elements = max_tensor_elements
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
        if len(works) != len(payloads):
            raise PipelineProtocolError(
                "activation payload send returned an unexpected work count"
            )
        for work, (_payload, field_id) in zip(works, payloads, strict=True):
            self._control.wait_work(
                work,
                step,
                f"activation-payload-send:{field_id}",
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
        header = torch.empty(ACTIVATION_HEADER_WORDS, dtype=torch.int64)
        header_work = dist.irecv(header, src=source, group=self._cpu_group)
        self._control.wait_work(header_work, step, "activation-header-receive")
        wire_header = ActivationWireHeader.validate_and_unpack(
            header.tolist(),
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

        values = []
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
            value = torch.empty(shape, dtype=dtype, device=self._device)
            total_elements += value.numel()
            values.append(value)
        if total_elements != wire_header.total_elements:
            raise PipelineProtocolError(
                "activation header total element count does not match payload"
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
        if len(works) != len(payloads):
            raise PipelineProtocolError(
                "activation payload receive returned an unexpected work count"
            )
        for work, (_value, field) in zip(works, payloads, strict=True):
            self._control.wait_work(
                work,
                step,
                f"activation-payload-receive:{field.field_id}",
            )
        return schema.bind(values)

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
        return None


class TorchPipelineResultSynchronizer:
    """Broadcast the target-only sampling result back along every PP lane."""

    _FIELDS_PER_REQUEST = 3

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise ValueError("pipeline result batch_size must be an integer")
        if batch_size < 0:
            raise ValueError("pipeline result batch_size must be non-negative")
        if step.descriptor.batch_size != batch_size:
            raise PipelineProtocolError(
                "pipeline result batch_size disagrees with the active step"
            )
        packet = torch.empty(
            RESULT_HEADER_WORDS + self._FIELDS_PER_REQUEST * batch_size,
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
            if any(tensor.numel() != batch_size for tensor in tensors):
                raise PipelineProtocolError(
                    "target-only pipeline sampling requires one result per request"
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
            payload[:batch_size].copy_(output_tokens.reshape(-1).to(torch.int64))
            payload[batch_size : 2 * batch_size].copy_(
                accept_lengths.reshape(-1).to(torch.int64)
            )
            if nan_flags is None:
                payload[2 * batch_size :].zero_()
            else:
                if nan_flags.numel() != batch_size:
                    raise PipelineProtocolError(
                        "pipeline NaN flags must contain one value per request"
                    )
                payload[2 * batch_size :].copy_(nan_flags.reshape(-1).to(torch.int64))
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
            payload[:batch_size].to(torch.int32),
            payload[batch_size : 2 * batch_size].to(torch.int32),
            payload[2 * batch_size :].to(torch.int32),
        )
