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

from __future__ import annotations

import copy

import pytest
from tokenspeed_kernel.ops.moe.flashinfer.cutlass_mxfp4_tactics import (
    _validate_manifest,
    cutlass_mxfp4_tactic_filename,
    select_cutlass_mxfp4_tactics,
)


def _expected() -> dict:
    return {
        "gpu": "NVIDIA RTX 6000D",
        "compute_capability": "12.0",
        "flashinfer_version": "0.6.16",
        "cutlass_tactic_abi_version": "0.1",
        "torch_version": "2.13.0+cu130",
        "cuda_version": "13.0",
        "hidden_size": 4096,
        "intermediate_size": 512,
        "num_experts": 256,
        "top_k": 6,
        "tp_size": 4,
        "ep_size": 1,
        "activation_type": "swiglu_bias",
        "activation_dtype": "mxfp8_e4m3",
        "weight_dtype": "mxfp4_e2m1_e8m0",
        "output_dtype": "bfloat16",
        "fused_finalize": True,
        "enable_pdl": False,
    }


def _manifest() -> dict:
    expected = _expected()
    return {
        "schema_version": 1,
        "backend": "flashinfer_cutlass_mxfp8_mxfp4",
        "metadata": {
            key: expected[key]
            for key in (
                "gpu",
                "compute_capability",
                "flashinfer_version",
                "cutlass_tactic_abi_version",
                "torch_version",
                "cuda_version",
            )
        },
        "shape": {
            key: expected[key]
            for key in (
                "hidden_size",
                "intermediate_size",
                "num_experts",
                "top_k",
                "tp_size",
                "ep_size",
                "activation_type",
                "activation_dtype",
                "weight_dtype",
                "output_dtype",
                "fused_finalize",
                "enable_pdl",
            )
        },
        "candidate_counts": {"gemm1": 20, "gemm2": 40},
        "tactics": {"1": [16, 46], "8": [18, 48]},
    }


def test_cutlass_mxfp4_tactic_filename_is_environment_specific() -> None:
    assert cutlass_mxfp4_tactic_filename(
        hidden_size=4096,
        intermediate_size=512,
        num_experts=256,
        top_k=6,
        tp_size=4,
        ep_size=1,
        device_name="NVIDIA RTX 6000D",
        flashinfer_version="0.6.16",
        cutlass_tactic_abi_version="0.1",
        enable_pdl=True,
    ) == (
        "cutlass-mxfp4,h=4096,i=512,e=256,k=6,tp=4,ep=1,"
        "device_name=NVIDIA_RTX_6000D,flashinfer=0.6.16,abi=0.1,pdl=1.json"
    )


def test_validate_manifest_accepts_exact_environment_and_shape() -> None:
    _validate_manifest(_manifest(), _expected())


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("metadata", "gpu", "NVIDIA B300"),
        ("metadata", "cutlass_tactic_abi_version", "0.2"),
        ("shape", "tp_size", 8),
        ("shape", "enable_pdl", True),
    ],
)
def test_validate_manifest_rejects_mismatch(section, key, value) -> None:
    manifest = copy.deepcopy(_manifest())
    manifest[section][key] = value
    with pytest.raises(RuntimeError, match="mismatch"):
        _validate_manifest(manifest, _expected())


@pytest.mark.parametrize("pair", [[20, 46], [16, 19], [16], [16, 60]])
def test_validate_manifest_rejects_invalid_profile_ids(pair) -> None:
    manifest = copy.deepcopy(_manifest())
    manifest["tactics"]["1"] = pair
    with pytest.raises(RuntimeError):
        _validate_manifest(manifest, _expected())


def test_select_cutlass_mxfp4_tactics_uses_floor_bucket() -> None:
    table = {1: [16, 46], 8: [18, 48], 16: [16, 48]}
    assert select_cutlass_mxfp4_tactics(table, 1) == [16, 46]
    assert select_cutlass_mxfp4_tactics(table, 15) == [18, 48]
    assert select_cutlass_mxfp4_tactics(table, 32) == [16, 48]
