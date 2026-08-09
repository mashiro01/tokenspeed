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

import pytest

from tokenspeed.runtime.pipeline.capabilities import (
    require_single_stage_control,
    validate_pipeline_capability,
)


def test_kimi_k3_bf16_pp8_is_qualified():
    capability = validate_pipeline_capability(
        architecture="KimiK3ForConditionalGeneration",
        activation_dtype="bfloat16",
        stage_count=8,
    )

    assert capability is not None
    assert capability.stage_counts == (8,)


@pytest.mark.parametrize(
    ("architecture", "activation_dtype", "stage_count", "message"),
    (
        ("UnknownForCausalLM", "bfloat16", 8, "no qualified implementation"),
        (
            "KimiK3ForConditionalGeneration",
            "float16",
            8,
            "requires activation dtype",
        ),
        (
            "KimiK3ForConditionalGeneration",
            "bfloat16",
            4,
            "requires stage count",
        ),
    ),
)
def test_unqualified_pipeline_configuration_fails_early(
    architecture: str,
    activation_dtype: str,
    stage_count: int,
    message: str,
):
    with pytest.raises(ValueError, match=message):
        validate_pipeline_capability(
            architecture=architecture,
            activation_dtype=activation_dtype,
            stage_count=stage_count,
        )


def test_pp1_does_not_require_a_pipeline_capability():
    assert (
        validate_pipeline_capability(
            architecture="UnknownForCausalLM",
            activation_dtype="float16",
            stage_count=1,
        )
        is None
    )


def test_non_transactional_control_operations_are_rejected_for_pp():
    require_single_stage_control(stage_count=1, operation="runtime profiling")
    with pytest.raises(RuntimeError, match="prepare/commit/rollback"):
        require_single_stage_control(stage_count=8, operation="runtime profiling")
