from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from tokenspeed.runtime.pipeline.contracts import (
    ActivationFieldSpec,
    ActivationSchema,
    PipelineForwardMode,
    PipelinePlan,
    PipelineProtocolError,
    PipelineStepDescriptor,
    batch_fingerprint,
    cache_table_digests,
    cache_table_fingerprint,
    multimodal_context_fingerprint,
    sampling_params_fingerprint,
)
from tokenspeed.runtime.pipeline.wire import (
    ACTIVATION_HEADER_WORDS,
    RESULT_HEADER_WORDS,
    ActivationWireHeader,
    ResultWireHeader,
)


@dataclass(frozen=True)
class FakeTensor:
    dtype: str
    shape: tuple[int, ...]


def _step(**overrides) -> PipelineStepDescriptor:
    values = {
        "epoch": 41,
        "step_id": 7,
        "forward_mode": PipelineForwardMode.EXTEND,
        "batch_size": 2,
        "input_num_tokens": 5,
        "num_extends": 2,
        "plan_digest": PipelinePlan.single(1).digest,
        "batch_fingerprint": 97,
    }
    values.update(overrides)
    return PipelineStepDescriptor(**values)


def _schema() -> ActivationSchema:
    return ActivationSchema(
        "stage-0-to-1",
        (
            ActivationFieldSpec("hidden", "bfloat16", (8,)),
            ActivationFieldSpec("residual", "bfloat16", (8,)),
        ),
    )


def test_step_descriptor_and_batch_fingerprint_are_stable():
    first = batch_fingerprint(("r1", "r2"), (3, 2), (0, 1))
    second = batch_fingerprint(("r1", "r2"), (3, 2), (0, 1))

    assert first == second
    assert first != batch_fingerprint(("r2", "r1"), (3, 2), (0, 1))
    richer = batch_fingerprint(
        ("r1", "r2"),
        (3, 2),
        (0, 1),
        request_pool_indices=(7, 9),
        prefill_lengths=(3, 5),
        decode_input_ids=(-1,),
        cache_table_digests=(("history", "int32", (2, 4), "a" * 64),),
    )
    assert richer != first
    assert richer != batch_fingerprint(
        ("r1", "r2"),
        (3, 2),
        (0, 1),
        request_pool_indices=(7, 9),
        prefill_lengths=(3, 5),
        input_token_ids=(1, 2, 4),
        shifted_input_ids=(2, 3, 5),
        decode_input_ids=(-1,),
        cache_table_digests=(("history", "int32", (2, 4), "a" * 64),),
    )
    assert richer != batch_fingerprint(
        ("r1", "r2"),
        (3, 2),
        (0, 1),
        request_pool_indices=(7, 8),
        prefill_lengths=(3, 5),
        decode_input_ids=(-1,),
        cache_table_digests=(("history", "int32", (2, 4), "a" * 64),),
    )
    assert _step().wire_words() == _step().wire_words()
    with pytest.raises(ValueError, match="equal length"):
        batch_fingerprint(("r1",), (1, 2), ())
    with pytest.raises(ValueError, match="non-idle.*non-empty"):
        _step(batch_size=0, input_num_tokens=0, num_extends=0)


def test_cache_table_digests_cover_raw_page_ids_and_geometry():
    first = np.arange(12, dtype=np.int32).reshape(3, 4)
    second = first.copy()

    original = cache_table_digests({"history": first})
    assert original == cache_table_digests({"history": second})
    second[2, 3] += 1
    assert original != cache_table_digests({"history": second})
    assert cache_table_fingerprint(original) == cache_table_fingerprint(original)
    assert cache_table_fingerprint(original) != cache_table_fingerprint(
        cache_table_digests({"history": second})
    )
    with pytest.raises(ValueError, match="C-contiguous"):
        cache_table_digests({"history": first[:, ::2]})


def test_sampling_and_multimodal_fingerprints_cover_execution_metadata():
    sampling = SimpleNamespace(
        temperature=0.7,
        top_p=0.9,
        top_k=32,
        min_p=0.0,
        frequency_penalty=0.1,
        presence_penalty=0.2,
        repetition_penalty=1.05,
        seed=17,
        logit_bias={"4": -1.0},
    )
    baseline = sampling_params_fingerprint((sampling,))
    sampling.top_k = 16
    assert baseline != sampling_params_fingerprint((sampling,))

    item = SimpleNamespace(
        modality=SimpleNamespace(name="IMAGE"),
        hash=41,
        pad_value=1_000_041,
        offsets=[(3, 9)],
        model_specific_data={},
    )
    mm_input = SimpleNamespace(
        mm_items=[item],
        im_token_id=7,
        video_token_id=None,
        mrope_positions=None,
        mrope_position_delta=None,
        mrope_position_delta_scalar=None,
    )
    context = SimpleNamespace(
        mm_inputs=[mm_input],
        extend_prefix_lens=[0],
        extend_seq_lens=[12],
    )
    baseline = multimodal_context_fingerprint(context)
    item.offsets = [(4, 10)]
    assert baseline != multimodal_context_fingerprint(context)


def test_activation_header_round_trip_covers_step_schema_and_geometry():
    schema = _schema()
    activation = schema.bind(
        (
            FakeTensor("bfloat16", (5, 8)),
            FakeTensor("bfloat16", (5, 8)),
        )
    )
    header = ActivationWireHeader.for_activation(
        step=_step(),
        source_stage=0,
        destination_stage=1,
        schema=schema,
        activation=activation,
    )
    packed = header.pack()

    assert len(packed) == ACTIVATION_HEADER_WORDS
    assert packed[15] == 2
    assert packed[17] == 80
    assert packed[20:22] == (5, 5)
    unpacked = ActivationWireHeader.validate_and_unpack(
        packed,
        expected_step=_step(),
        expected_schema=schema,
        expected_source_stage=0,
        expected_destination_stage=1,
    )
    assert unpacked.leading_dimensions == (5, 5)
    assert unpacked.total_elements == 80


@pytest.mark.parametrize("slot", range(3, 15))
def test_activation_header_rejects_step_or_schema_identity_tampering(slot: int):
    schema = _schema()
    activation = schema.bind(
        (
            FakeTensor("bfloat16", (2, 8)),
            FakeTensor("bfloat16", (2, 8)),
        )
    )
    packed = list(
        ActivationWireHeader.for_activation(
            step=_step(),
            source_stage=0,
            destination_stage=1,
            schema=schema,
            activation=activation,
        ).pack()
    )
    packed[slot] += 1

    with pytest.raises(PipelineProtocolError, match="identity"):
        ActivationWireHeader.validate_and_unpack(
            packed,
            expected_step=_step(),
            expected_schema=schema,
            expected_source_stage=0,
            expected_destination_stage=1,
        )


def test_result_header_round_trip_and_stale_step_rejection():
    packed = ResultWireHeader(_step(), source_stage=7).pack()

    assert len(packed) == RESULT_HEADER_WORDS
    ResultWireHeader.validate(
        packed,
        expected_step=_step(),
        expected_source_stage=7,
    )
    with pytest.raises(PipelineProtocolError, match="identity"):
        ResultWireHeader.validate(
            packed,
            expected_step=_step(step_id=8),
            expected_source_stage=7,
        )
