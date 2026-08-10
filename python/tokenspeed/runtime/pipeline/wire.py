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

"""Fixed-size, versioned wire headers for native pipeline messages."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence

from tokenspeed.runtime.pipeline.contracts import (
    ActivationSchema,
    PipelineProtocolError,
    PipelineStepDescriptor,
    digest_prefix,
)

WIRE_MAGIC = 0x5453504950455631  # "TSPIPEV1"
WIRE_PROTOCOL_VERSION = 1
ACTIVATION_HEADER_WORDS = 64
ACTIVATION_LEADING_DIMENSION_OFFSET = 20
MAX_ACTIVATION_LEADING_DIMENSIONS = (
    ACTIVATION_HEADER_WORDS - ACTIVATION_LEADING_DIMENSION_OFFSET
)
RESULT_HEADER_WORDS = 16


class PipelineMessageKind(IntEnum):
    ACTIVATION = 1
    RESULT = 2


def _words(value: Sequence[int], expected: int, label: str) -> tuple[int, ...]:
    try:
        result = tuple(int(word) for word in value)
    except (TypeError, ValueError) as exc:
        raise PipelineProtocolError(f"{label} contains a non-integer word") from exc
    if len(result) != expected:
        raise PipelineProtocolError(
            f"{label} requires {expected} words, got {len(result)}"
        )
    return result


@dataclass(frozen=True)
class ActivationWireHeader:
    step: PipelineStepDescriptor
    source_stage: int
    destination_stage: int
    schema_digest: str
    field_count: int
    tensor_count: int
    total_elements: int
    leading_dimensions: tuple[int, ...]
    flags: int = 0

    def pack(self) -> tuple[int, ...]:
        leading_dimensions = tuple(self.leading_dimensions)
        if len(leading_dimensions) > MAX_ACTIVATION_LEADING_DIMENSIONS:
            raise PipelineProtocolError(
                "pipeline activation exceeds the wire leading-dimension limit"
            )
        if any(value < 0 for value in leading_dimensions):
            raise PipelineProtocolError(
                "pipeline activation dimensions must be non-negative"
            )
        words = [0] * ACTIVATION_HEADER_WORDS
        words[:20] = (
            WIRE_MAGIC,
            WIRE_PROTOCOL_VERSION,
            int(PipelineMessageKind.ACTIVATION),
            self.step.epoch,
            self.step.step_id,
            int(self.step.forward_mode),
            self.step.batch_size,
            self.step.input_num_tokens,
            self.step.num_extends,
            self.source_stage,
            self.destination_stage,
            self.step.plan_digest_prefix,
            digest_prefix(self.schema_digest),
            self.step.batch_fingerprint,
            self.field_count,
            len(leading_dimensions),
            self.tensor_count,
            self.total_elements,
            self.flags,
            0,
        )
        words[
            ACTIVATION_LEADING_DIMENSION_OFFSET : ACTIVATION_LEADING_DIMENSION_OFFSET
            + len(leading_dimensions)
        ] = leading_dimensions
        return tuple(words)

    @classmethod
    def validate_and_unpack(
        cls,
        value: Sequence[int],
        *,
        expected_step: PipelineStepDescriptor,
        expected_schema: ActivationSchema,
        expected_source_stage: int,
        expected_destination_stage: int,
    ) -> "ActivationWireHeader":
        words = _words(value, ACTIVATION_HEADER_WORDS, "activation header")
        expected_prefix = (
            WIRE_MAGIC,
            WIRE_PROTOCOL_VERSION,
            int(PipelineMessageKind.ACTIVATION),
            expected_step.epoch,
            expected_step.step_id,
            int(expected_step.forward_mode),
            expected_step.batch_size,
            expected_step.input_num_tokens,
            expected_step.num_extends,
            expected_source_stage,
            expected_destination_stage,
            expected_step.plan_digest_prefix,
            digest_prefix(expected_schema.digest),
            expected_step.batch_fingerprint,
            len(expected_schema.fields),
        )
        if words[:15] != expected_prefix:
            raise PipelineProtocolError(
                "activation wire identity does not match this step"
            )
        leading_count = words[15]
        if not 0 <= leading_count <= MAX_ACTIVATION_LEADING_DIMENSIONS:
            raise PipelineProtocolError("activation leading-dimension count is invalid")
        if words[16] != len(expected_schema.fields):
            raise PipelineProtocolError("activation tensor count does not match schema")
        if words[17] < 0 or words[18] != 0 or words[19] != 0:
            raise PipelineProtocolError(
                "activation header metadata or flags are invalid"
            )
        if any(words[ACTIVATION_LEADING_DIMENSION_OFFSET + leading_count :]):
            raise PipelineProtocolError("activation header reserved words must be zero")
        leading_dimensions = words[
            ACTIVATION_LEADING_DIMENSION_OFFSET : ACTIVATION_LEADING_DIMENSION_OFFSET
            + leading_count
        ]
        return cls(
            step=expected_step,
            source_stage=expected_source_stage,
            destination_stage=expected_destination_stage,
            schema_digest=expected_schema.digest,
            field_count=words[14],
            tensor_count=words[16],
            total_elements=words[17],
            leading_dimensions=leading_dimensions,
            flags=words[18],
        )

    @classmethod
    def for_activation(
        cls,
        *,
        step: PipelineStepDescriptor,
        source_stage: int,
        destination_stage: int,
        schema: ActivationSchema,
        activation,
    ) -> "ActivationWireHeader":
        leading_dimensions = []
        total_elements = 0
        for field, tensor in zip(schema.fields, activation.values):
            shape = tuple(int(dimension) for dimension in tensor.shape)
            leading_dimensions.extend(shape[: field.leading_rank])
            total_elements += math.prod(shape)
        return cls(
            step=step,
            source_stage=source_stage,
            destination_stage=destination_stage,
            schema_digest=schema.digest,
            field_count=len(schema.fields),
            tensor_count=len(activation.values),
            total_elements=total_elements,
            leading_dimensions=tuple(leading_dimensions),
        )


@dataclass(frozen=True)
class ResultWireHeader:
    step: PipelineStepDescriptor
    source_stage: int
    field_count: int = 3
    flags: int = 0

    def pack(self) -> tuple[int, ...]:
        words = [0] * RESULT_HEADER_WORDS
        words[:13] = (
            WIRE_MAGIC,
            WIRE_PROTOCOL_VERSION,
            int(PipelineMessageKind.RESULT),
            self.step.epoch,
            self.step.step_id,
            int(self.step.forward_mode),
            self.step.batch_size,
            self.step.input_num_tokens,
            self.step.num_extends,
            self.source_stage,
            self.step.plan_digest_prefix,
            self.step.batch_fingerprint,
            self.field_count,
        )
        words[13] = self.flags
        return tuple(words)

    @classmethod
    def validate(
        cls,
        value: Sequence[int],
        *,
        expected_step: PipelineStepDescriptor,
        expected_source_stage: int,
        expected_field_count: int = 3,
        expected_flags: int = 0,
    ) -> None:
        words = _words(value, RESULT_HEADER_WORDS, "result header")
        expected = cls(
            expected_step,
            expected_source_stage,
            field_count=expected_field_count,
            flags=expected_flags,
        ).pack()
        if words != expected:
            raise PipelineProtocolError("result wire identity does not match this step")
