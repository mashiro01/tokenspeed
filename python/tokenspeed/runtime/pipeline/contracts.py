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

"""Model-independent contracts for pipeline stage execution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import IntEnum
from hashlib import sha256
from typing import Mapping, Protocol, Sequence

API_VERSION = "tokenspeed.pipeline/v1alpha1"


class PipelineProtocolError(RuntimeError):
    """A stage plan or activation violates the pipeline contract."""


class PipelineStepAborted(PipelineProtocolError):
    """The current step poisoned the pipeline instance."""


def canonical_digest(value: object) -> str:
    """Return a deterministic digest for a canonical JSON projection."""

    _validate_canonical_json(value)
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _validate_canonical_json(value: object) -> None:
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        raise ValueError("floating-point values are forbidden in canonical digests")
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_canonical_json(item)
        return
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("canonical object keys must be strings")
        for item in value.values():
            _validate_canonical_json(item)
        return
    raise TypeError(f"unsupported canonical value: {type(value).__name__}")


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _non_negative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return value


def digest_prefix(value: str) -> int:
    """Encode the first 60 bits of a lowercase SHA-256 digest for the wire."""

    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("wire digest must be a lowercase SHA-256 digest")
    return int(value[:15], 16)


class PipelineForwardMode(IntEnum):
    """Stable wire values independent of the runtime ForwardMode enum."""

    DECODE = 1
    EXTEND = 2
    MIXED = 3
    IDLE = 4

    @classmethod
    def from_runtime_name(cls, name: str) -> "PipelineForwardMode":
        try:
            return cls[name.upper()]
        except (AttributeError, KeyError) as exc:
            raise ValueError(f"unsupported pipeline forward mode {name!r}") from exc


@dataclass(frozen=True)
class PipelineStepDescriptor:
    """Rank-independent identity for one globally mirrored scheduler step."""

    epoch: int
    step_id: int
    forward_mode: PipelineForwardMode
    batch_size: int
    input_num_tokens: int
    num_extends: int
    plan_digest: str
    batch_fingerprint: int

    def __post_init__(self) -> None:
        _positive_int("pipeline epoch", self.epoch)
        _positive_int("pipeline step_id", self.step_id)
        if not isinstance(self.forward_mode, PipelineForwardMode):
            try:
                object.__setattr__(
                    self,
                    "forward_mode",
                    PipelineForwardMode(self.forward_mode),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("pipeline forward_mode is invalid") from exc
        _non_negative_int("pipeline batch_size", self.batch_size)
        _non_negative_int("pipeline input_num_tokens", self.input_num_tokens)
        _non_negative_int("pipeline num_extends", self.num_extends)
        if self.num_extends > self.batch_size:
            raise ValueError("pipeline num_extends cannot exceed batch_size")
        if self.forward_mode is PipelineForwardMode.IDLE:
            if self.batch_size != 0 or self.input_num_tokens != 0:
                raise ValueError("pipeline idle step must have an empty batch")
        elif self.batch_size == 0:
            raise ValueError("a non-idle pipeline step requires a non-empty batch")
        digest_prefix(self.plan_digest)
        _non_negative_int("pipeline batch_fingerprint", self.batch_fingerprint)
        if self.epoch >= 1 << 60 or self.batch_fingerprint >= 1 << 60:
            raise ValueError("pipeline wire identifiers must fit in 60 bits")

    @property
    def plan_digest_prefix(self) -> int:
        return digest_prefix(self.plan_digest)

    def wire_words(self) -> tuple[int, ...]:
        """Return the fixed descriptor compared before any NCCL data work."""

        return (
            1,
            self.epoch,
            self.step_id,
            int(self.forward_mode),
            self.batch_size,
            self.input_num_tokens,
            self.num_extends,
            self.plan_digest_prefix,
            self.batch_fingerprint,
        )


def batch_fingerprint(
    request_ids: Sequence[str],
    input_lengths: Sequence[int],
    extend_prefix_lengths: Sequence[int],
    *,
    request_pool_indices: Sequence[int] = (),
    prefill_lengths: Sequence[int] = (),
    input_token_ids: Sequence[int] = (),
    shifted_input_ids: Sequence[int] = (),
    decode_input_ids: Sequence[int] = (),
    sampling_fingerprint: int = 0,
    multimodal_fingerprint: int = 0,
    cache_table_digests: Sequence[tuple[str, str, tuple[int, ...], str]] = (),
) -> int:
    """Return a deterministic 60-bit identity without touching GPU tensors."""

    request_ids = tuple(request_ids)
    input_lengths = tuple(input_lengths)
    extend_prefix_lengths = tuple(extend_prefix_lengths)
    request_pool_indices = tuple(request_pool_indices)
    prefill_lengths = tuple(prefill_lengths)
    input_token_ids = tuple(input_token_ids)
    shifted_input_ids = tuple(shifted_input_ids)
    decode_input_ids = tuple(decode_input_ids)
    cache_table_digests = tuple(cache_table_digests)
    if len(request_ids) != len(input_lengths):
        raise ValueError("request ids and input lengths must have equal length")
    for name, values in (
        ("request_pool_indices", request_pool_indices),
        ("prefill_lengths", prefill_lengths),
    ):
        if values and len(values) != len(request_ids):
            raise ValueError(f"{name} must be empty or cover every request")
    if len(decode_input_ids) > len(request_ids):
        raise ValueError("decode_input_ids cannot exceed the request count")
    if any(not isinstance(request_id, str) for request_id in request_ids):
        raise TypeError("pipeline request ids must be strings")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (
            *input_lengths,
            *extend_prefix_lengths,
            *request_pool_indices,
            *prefill_lengths,
        )
    ):
        raise ValueError("pipeline batch lengths must be non-negative integers")
    for name, values in (
        ("input token ids", input_token_ids),
        ("shifted input ids", shifted_input_ids),
        ("decode input ids", decode_input_ids),
    ):
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            raise ValueError(f"pipeline {name} must be integers")
    for name, value in (
        ("sampling_fingerprint", sampling_fingerprint),
        ("multimodal_fingerprint", multimodal_fingerprint),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < 1 << 60
        ):
            raise ValueError(f"pipeline {name} must be a non-negative 60-bit integer")
    normalized_cache_tables = []
    for entry in cache_table_digests:
        if not isinstance(entry, (list, tuple)) or len(entry) != 4:
            raise ValueError("pipeline cache table digest entries require four fields")
        group_id, dtype, shape, digest = entry
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("pipeline cache group ids must be non-empty strings")
        if not isinstance(dtype, str) or not dtype:
            raise ValueError("pipeline cache table dtypes must be non-empty strings")
        shape = tuple(shape)
        if any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension < 0
            for dimension in shape
        ):
            raise ValueError(
                "pipeline cache table shapes must be non-negative integers"
            )
        digest_prefix(digest)
        normalized_cache_tables.append((group_id, dtype, shape, digest))
    return digest_prefix(
        canonical_digest(
            {
                "request_ids": request_ids,
                "input_lengths": input_lengths,
                "extend_prefix_lengths": extend_prefix_lengths,
                "request_pool_indices": request_pool_indices,
                "prefill_lengths": prefill_lengths,
                "input_token_ids": input_token_ids,
                "shifted_input_ids": shifted_input_ids,
                "decode_input_ids": decode_input_ids,
                "sampling_fingerprint": sampling_fingerprint,
                "multimodal_fingerprint": multimodal_fingerprint,
                "cache_table_digests": tuple(normalized_cache_tables),
            }
        )
    )


def cache_table_digests(
    block_tables: Mapping[object, object],
) -> tuple[tuple[str, str, tuple[int, ...], str], ...]:
    """Hash contiguous scheduler page tables without expanding page ids."""

    result = []
    for group_id, array in sorted(
        block_tables.items(),
        key=lambda item: str(item[0]),
    ):
        shape = tuple(int(dimension) for dimension in array.shape)
        if len(shape) != 2:
            raise ValueError(
                f"pipeline cache table {group_id!r} must be two-dimensional"
            )
        if not array.flags.c_contiguous:
            raise ValueError(f"pipeline cache table {group_id!r} must be C-contiguous")
        result.append(
            (
                str(group_id),
                str(array.dtype),
                shape,
                sha256(memoryview(array)).hexdigest(),
            )
        )
    return tuple(result)


def cache_table_fingerprint(
    digests: Sequence[tuple[str, str, tuple[int, ...], str]],
) -> int:
    """Return the stage-local identity of scheduler page-table state."""

    return batch_fingerprint((), (), (), cache_table_digests=digests)


def sampling_params_fingerprint(params: Sequence[object]) -> int:
    """Hash every SamplingParams field that can change token selection."""

    manifest = []
    for param in params:
        bias = getattr(param, "logit_bias", None)
        normalized_bias = (
            ()
            if not bias
            else tuple(
                sorted(
                    (str(token_id), float(value).hex())
                    for token_id, value in bias.items()
                )
            )
        )
        manifest.append(
            (
                float(getattr(param, "temperature")).hex(),
                float(getattr(param, "top_p")).hex(),
                int(getattr(param, "top_k")),
                float(getattr(param, "min_p")).hex(),
                float(getattr(param, "frequency_penalty")).hex(),
                float(getattr(param, "presence_penalty")).hex(),
                float(getattr(param, "repetition_penalty")).hex(),
                getattr(param, "seed"),
                normalized_bias,
            )
        )
    return digest_prefix(canonical_digest(manifest))


def multimodal_context_fingerprint(context: object | None) -> int:
    """Hash image identity and placement metadata without scanning feature bytes."""

    if context is None:
        return 0

    def tensor_geometry(value):
        if value is None:
            return None
        return (str(value.dtype).removeprefix("torch."), tuple(value.shape))

    requests = []
    for mm_input in context.mm_inputs:
        if mm_input is None:
            requests.append(None)
            continue
        items = []
        for item in mm_input.mm_items:
            modality = getattr(item.modality, "name", str(item.modality))
            model_specific = tuple(
                (key, tensor_geometry(value))
                for key, value in sorted(item.model_specific_data.items())
            )
            items.append(
                (
                    modality,
                    item.hash,
                    item.pad_value,
                    tuple(item.offsets or ()),
                    model_specific,
                )
            )
        requests.append(
            (
                tuple(items),
                mm_input.im_token_id,
                mm_input.video_token_id,
                tensor_geometry(mm_input.mrope_positions),
                tensor_geometry(mm_input.mrope_position_delta),
                mm_input.mrope_position_delta_scalar,
            )
        )
    return digest_prefix(
        canonical_digest(
            {
                "requests": tuple(requests),
                "extend_prefix_lens": tuple(context.extend_prefix_lens),
                "extend_seq_lens": tuple(context.extend_seq_lens),
            }
        )
    )


@dataclass(frozen=True)
class ActivationFieldSpec:
    """One tensor-like field crossing a pipeline boundary.

    Leading dimensions are runtime bucket dimensions. ``trailing_shape`` is
    the fixed model shape and must match exactly.
    """

    field_id: str
    dtype: str
    trailing_shape: tuple[int, ...]
    leading_rank: int = 1
    leading_shape_id: str | None = "tokens"

    def __post_init__(self) -> None:
        if not isinstance(self.field_id, str) or not self.field_id:
            raise ValueError("field_id must be non-empty")
        if not isinstance(self.dtype, str) or not self.dtype:
            raise ValueError("dtype must be non-empty")
        object.__setattr__(self, "trailing_shape", tuple(self.trailing_shape))
        _non_negative_int("leading_rank", self.leading_rank)
        if self.leading_shape_id is not None and (
            not isinstance(self.leading_shape_id, str) or not self.leading_shape_id
        ):
            raise ValueError("leading_shape_id must be a non-empty string or None")
        for dimension in self.trailing_shape:
            _positive_int("trailing_shape dimension", dimension)

    def projection(self) -> dict[str, object]:
        return {
            "field_id": self.field_id,
            "dtype": self.dtype,
            "trailing_shape": self.trailing_shape,
            "leading_rank": self.leading_rank,
            "leading_shape_id": self.leading_shape_id,
        }


@dataclass(frozen=True)
class ActivationSchema:
    """Static ordered schema for one adjacent-stage boundary."""

    boundary_id: str
    fields: tuple[ActivationFieldSpec, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.boundary_id, str) or not self.boundary_id:
            raise ValueError("boundary_id must be non-empty")
        object.__setattr__(self, "fields", tuple(self.fields))
        if not self.fields:
            raise ValueError("an activation schema must contain at least one field")
        if any(not isinstance(field, ActivationFieldSpec) for field in self.fields):
            raise TypeError("activation schema fields must be ActivationFieldSpec")
        field_ids = tuple(field.field_id for field in self.fields)
        if len(field_ids) != len(set(field_ids)):
            raise ValueError("activation field ids must be unique")

    @property
    def digest(self) -> str:
        return canonical_digest(
            {
                "api_version": API_VERSION,
                "kind": "activation_schema",
                "boundary_id": self.boundary_id,
                "fields": tuple(field.projection() for field in self.fields),
            }
        )

    def bind(self, values: Sequence[object]) -> "StageActivation":
        activation = StageActivation(tuple(values), self.digest)
        self.validate(activation)
        return activation

    def validate(self, activation: "StageActivation") -> None:
        if activation.schema_digest != self.digest:
            raise PipelineProtocolError(
                f"boundary {self.boundary_id} schema digest mismatch"
            )
        if len(activation.values) != len(self.fields):
            raise PipelineProtocolError(
                f"boundary {self.boundary_id} expected {len(self.fields)} fields, "
                f"got {len(activation.values)}"
            )
        leading_shapes: dict[str, tuple[int, ...]] = {}
        for field, value in zip(self.fields, activation.values):
            dtype = str(getattr(value, "dtype", "")).removeprefix("torch.")
            if dtype != field.dtype:
                raise PipelineProtocolError(
                    f"field {field.field_id} expected dtype {field.dtype}, got {dtype!r}"
                )
            raw_shape = getattr(value, "shape", None)
            if raw_shape is None:
                raise PipelineProtocolError(f"field {field.field_id} has no shape")
            try:
                shape = tuple(int(dimension) for dimension in raw_shape)
            except (TypeError, ValueError) as error:
                raise PipelineProtocolError(
                    f"field {field.field_id} has an invalid shape"
                ) from error
            trailing = field.trailing_shape
            expected_rank = field.leading_rank + len(trailing)
            if len(shape) != expected_rank:
                raise PipelineProtocolError(
                    f"field {field.field_id} expected rank {expected_rank}, "
                    f"got shape {shape}"
                )
            if trailing and shape[-len(trailing) :] != trailing:
                raise PipelineProtocolError(
                    f"field {field.field_id} expected trailing shape {trailing}, "
                    f"got {shape}"
                )
            if field.leading_shape_id is not None:
                leading = shape[: field.leading_rank]
                previous = leading_shapes.setdefault(field.leading_shape_id, leading)
                if previous != leading:
                    raise PipelineProtocolError(
                        f"field {field.field_id} leading shape {leading} disagrees "
                        f"with {field.leading_shape_id!r} shape {previous}"
                    )


@dataclass(frozen=True)
class StagePlan:
    """Static ownership and adjacent payload contract for one stage."""

    stage_id: int
    stage_count: int
    first_layer: int
    end_layer: int
    owns_embedding: bool
    owns_head: bool
    input_schema: ActivationSchema | None = None
    output_schema: ActivationSchema | None = None

    def __post_init__(self) -> None:
        _positive_int("stage_count", self.stage_count)
        _non_negative_int("stage_id", self.stage_id)
        if self.stage_id >= self.stage_count:
            raise ValueError("stage_id must be smaller than stage_count")
        _non_negative_int("first_layer", self.first_layer)
        _positive_int("end_layer", self.end_layer)
        if self.end_layer <= self.first_layer:
            raise ValueError("end_layer must be greater than first_layer")
        if not isinstance(self.owns_embedding, bool) or not isinstance(
            self.owns_head, bool
        ):
            raise ValueError("stage ownership flags must be boolean")
        if self.stage_id == 0 and self.input_schema is not None:
            raise ValueError("the first stage cannot have an activation input schema")
        if self.stage_id > 0 and self.input_schema is None:
            raise ValueError("a non-first stage requires an activation input schema")
        if self.stage_id == self.stage_count - 1 and self.output_schema is not None:
            raise ValueError("the final stage cannot have an activation output schema")
        if self.stage_id < self.stage_count - 1 and self.output_schema is None:
            raise ValueError("a non-final stage requires an activation output schema")

    def projection(self) -> dict[str, object]:
        return {
            "stage_id": self.stage_id,
            "stage_count": self.stage_count,
            "first_layer": self.first_layer,
            "end_layer": self.end_layer,
            "owns_embedding": self.owns_embedding,
            "owns_head": self.owns_head,
            "input_schema_digest": (
                None if self.input_schema is None else self.input_schema.digest
            ),
            "output_schema_digest": (
                None if self.output_schema is None else self.output_schema.digest
            ),
        }


@dataclass(frozen=True)
class PipelinePlan:
    """Validated ordered collection of stage plans."""

    stages: tuple[StagePlan, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "stages", tuple(self.stages))
        if not self.stages:
            raise ValueError("a pipeline plan requires at least one stage")
        if any(not isinstance(stage, StagePlan) for stage in self.stages):
            raise TypeError("pipeline stages must be StagePlan values")
        expected_count = len(self.stages)
        if tuple(stage.stage_id for stage in self.stages) != tuple(
            range(expected_count)
        ):
            raise ValueError("pipeline stages must be ordered and gap-free")
        if any(stage.stage_count != expected_count for stage in self.stages):
            raise ValueError("every stage must agree on stage_count")
        if self.stages[0].first_layer != 0:
            raise ValueError("a pipeline plan must start at logical layer zero")
        if tuple(stage.owns_embedding for stage in self.stages) != (
            True,
            *(False for _ in range(expected_count - 1)),
        ):
            raise ValueError("only the first stage must own the embedding")
        if tuple(stage.owns_head for stage in self.stages) != (
            *(False for _ in range(expected_count - 1)),
            True,
        ):
            raise ValueError("only the final stage must own the head")
        for left, right in zip(self.stages, self.stages[1:]):
            if left.end_layer != right.first_layer:
                raise ValueError("pipeline layer ranges must be contiguous")
            if left.output_schema is None or right.input_schema is None:
                raise ValueError("adjacent stages require a boundary schema")
            if left.output_schema.digest != right.input_schema.digest:
                raise ValueError("adjacent stages disagree on boundary schema")

    @classmethod
    def single(cls, num_layers: int) -> "PipelinePlan":
        _positive_int("num_layers", num_layers)
        return cls(
            (
                StagePlan(
                    stage_id=0,
                    stage_count=1,
                    first_layer=0,
                    end_layer=num_layers,
                    owns_embedding=True,
                    owns_head=True,
                ),
            )
        )

    @property
    def digest(self) -> str:
        return canonical_digest(
            {
                "api_version": API_VERSION,
                "kind": "pipeline_plan",
                "stages": tuple(stage.projection() for stage in self.stages),
            }
        )


@dataclass(frozen=True)
class StageActivation:
    """Opaque tensor-like values bound to a static activation schema."""

    values: tuple[object, ...]
    schema_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(self.values))
        if not isinstance(self.schema_digest, str) or not self.schema_digest:
            raise ValueError("schema_digest must be a non-empty string")


@dataclass(frozen=True)
class StageOutput:
    """Exactly one non-final activation or final model output."""

    activation: StageActivation | None = None
    final_output: object | None = None


class StageModel(Protocol):
    """Model adapter that owns exactly one stage's modules."""

    @property
    def plan(self) -> StagePlan: ...

    def forward_stage(
        self,
        context: object,
        batch: object,
        incoming: StageActivation | None,
    ) -> StageOutput: ...

    def idle(self) -> None: ...

    def close(self) -> None: ...
