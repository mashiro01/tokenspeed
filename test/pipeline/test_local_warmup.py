from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tokenspeed.runtime.pipeline.contracts import ActivationFieldSpec, ActivationSchema
from tokenspeed.runtime.pipeline.local_warmup import (
    get_pipeline_local_warmup_token_sizes,
    make_pipeline_local_warmup_activation,
)


def _schema(*, leading_rank: int = 1) -> ActivationSchema:
    return ActivationSchema(
        boundary_id="test/stage-0-to-1",
        fields=(
            ActivationFieldSpec(
                field_id="prefix_sum",
                dtype="bfloat16",
                trailing_shape=(8,),
                leading_rank=leading_rank,
            ),
            ActivationFieldSpec(
                field_id="block_residual.0",
                dtype="float32",
                trailing_shape=(8,),
                leading_rank=leading_rank,
            ),
        ),
    )


def test_local_warmup_token_sizes_are_bounded_and_deduplicated() -> None:
    assert get_pipeline_local_warmup_token_sizes(
        chunked_prefill_size=8192, max_tokens=8192
    ) == (1, 128, 2048, 8192)
    assert get_pipeline_local_warmup_token_sizes(
        chunked_prefill_size=512, max_tokens=8192
    ) == (1, 128, 512)
    assert get_pipeline_local_warmup_token_sizes(
        chunked_prefill_size=8192, max_tokens=2048
    ) == (1, 128, 2048)
    assert get_pipeline_local_warmup_token_sizes(
        chunked_prefill_size=0, max_tokens=8192
    ) == ()


def test_local_warmup_activation_honors_the_pipeline_schema() -> None:
    schema = _schema()

    activation = make_pipeline_local_warmup_activation(
        schema, num_tokens=16, device="cpu"
    )

    schema.validate(activation)
    assert [value.shape for value in activation.values] == [(16, 8), (16, 8)]
    assert [value.dtype for value in activation.values] == [
        torch.bfloat16,
        torch.float32,
    ]


def test_local_warmup_rejects_unknown_dynamic_activation_shape() -> None:
    with pytest.raises(ValueError, match="token-major"):
        make_pipeline_local_warmup_activation(
            _schema(leading_rank=2), num_tokens=16, device="cpu"
        )
