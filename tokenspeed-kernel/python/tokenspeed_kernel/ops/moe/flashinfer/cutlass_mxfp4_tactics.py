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

import functools
import json
import logging
import os
from importlib.metadata import version
from typing import Any

import torch

_BACKEND = "flashinfer_cutlass_mxfp8_mxfp4"
logger = logging.getLogger(__name__)


def cutlass_mxfp4_tactic_filename(
    *,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    tp_size: int,
    ep_size: int,
    device_name: str,
    flashinfer_version: str,
    cutlass_tactic_abi_version: str,
    enable_pdl: bool,
) -> str:
    """Build the exact-environment filename for a CUTLASS MXFP4 tactic table."""
    device = device_name.replace(" ", "_")
    return (
        f"cutlass-mxfp4,h={hidden_size},i={intermediate_size},e={num_experts},"
        f"k={top_k},tp={tp_size},ep={ep_size},device_name={device},"
        f"flashinfer={flashinfer_version},abi={cutlass_tactic_abi_version},"
        f"pdl={int(enable_pdl)}.json"
    )


def _validate_manifest(payload: dict[str, Any], expected: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1 or payload.get("backend") != _BACKEND:
        raise RuntimeError("unsupported CUTLASS MXFP4 tactic-manifest schema")
    metadata = payload.get("metadata")
    shape = payload.get("shape")
    if not isinstance(metadata, dict) or not isinstance(shape, dict):
        raise RuntimeError("CUTLASS MXFP4 tactic manifest lacks metadata or shape")
    for key in (
        "gpu",
        "compute_capability",
        "flashinfer_version",
        "cutlass_tactic_abi_version",
        "torch_version",
        "cuda_version",
    ):
        if metadata.get(key) != expected[key]:
            raise RuntimeError(
                f"CUTLASS MXFP4 tactic manifest {key} mismatch: "
                f"expected {expected[key]!r}, got {metadata.get(key)!r}"
            )
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
    ):
        if shape.get(key) != expected[key]:
            raise RuntimeError(
                f"CUTLASS MXFP4 tactic manifest {key} mismatch: "
                f"expected {expected[key]!r}, got {shape.get(key)!r}"
            )
    counts = payload.get("candidate_counts")
    if not isinstance(counts, dict):
        raise RuntimeError("CUTLASS MXFP4 tactic manifest lacks candidate counts")
    gemm1_count = int(counts.get("gemm1", 0))
    gemm2_count = int(counts.get("gemm2", 0))
    if gemm1_count <= 0 or gemm2_count <= 0:
        raise RuntimeError("CUTLASS MXFP4 tactic manifest has invalid candidate counts")
    tactics = payload.get("tactics")
    if not isinstance(tactics, dict) or not tactics:
        raise RuntimeError("CUTLASS MXFP4 tactic manifest has no tactics")
    for raw_bucket, raw_pair in tactics.items():
        bucket = int(raw_bucket)
        if bucket <= 0 or not isinstance(raw_pair, list) or len(raw_pair) != 2:
            raise RuntimeError(
                f"invalid CUTLASS MXFP4 tactic entry {raw_bucket!r}: {raw_pair!r}"
            )
        gemm1, gemm2 = map(int, raw_pair)
        if not (0 <= gemm1 < gemm1_count):
            raise RuntimeError(f"GEMM1 tactic {gemm1} is out of range")
        if not (gemm1_count <= gemm2 < gemm1_count + gemm2_count):
            raise RuntimeError(f"GEMM2 tactic {gemm2} is out of range")


@functools.cache
def load_cutlass_mxfp4_tactics(
    *,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    tp_size: int,
    ep_size: int,
    activation_type: str,
    enable_pdl: bool,
    device_index: int,
) -> dict[int, list[int]] | None:
    """Load an exact-match packaged tactic table or return ``None`` on a miss.

    A table hit is fail-closed: malformed data and every environment or shape
    mismatch raise at model load instead of silently running stale profile IDs.
    Absence is a normal miss for model shapes that have never been swept.
    """
    from flashinfer.autotuner.autotuner import _nvfp4_cutlass_version

    flashinfer_version = version("flashinfer-python")
    abi_version = str(_nvfp4_cutlass_version)
    device = torch.device("cuda", device_index)
    device_name = torch.cuda.get_device_name(device)
    major, minor = torch.cuda.get_device_capability(device)
    expected = {
        "gpu": device_name,
        "compute_capability": f"{major}.{minor}",
        "flashinfer_version": flashinfer_version,
        "cutlass_tactic_abi_version": abi_version,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_experts": num_experts,
        "top_k": top_k,
        "tp_size": tp_size,
        "ep_size": ep_size,
        "activation_type": activation_type,
        "activation_dtype": "mxfp8_e4m3",
        "weight_dtype": "mxfp4_e2m1_e8m0",
        "output_dtype": "bfloat16",
        "fused_finalize": True,
        "enable_pdl": enable_pdl,
    }
    filename = cutlass_mxfp4_tactic_filename(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        tp_size=tp_size,
        ep_size=ep_size,
        device_name=device_name,
        flashinfer_version=flashinfer_version,
        cutlass_tactic_abi_version=abi_version,
        enable_pdl=enable_pdl,
    )
    path = os.path.join(os.path.dirname(__file__), "tactics", filename)
    if not os.path.exists(path):
        return None
    with open(path) as file:
        payload = json.load(file)
    _validate_manifest(payload, expected)
    table = {
        int(bucket): [int(profile_ids[0]), int(profile_ids[1])]
        for bucket, profile_ids in payload["tactics"].items()
    }
    logger.info(
        "Loaded %d exact CUTLASS MXFP4 tactic buckets from %s",
        len(table),
        filename,
    )
    return table


def select_cutlass_mxfp4_tactics(
    table: dict[int, list[int]], num_tokens: int
) -> list[int]:
    """Select the same floor bucket used by FlashInfer's hybrid mapper."""
    buckets = sorted(table)
    selected = buckets[0]
    for bucket in buckets:
        if bucket > num_tokens:
            break
        selected = bucket
    return table[selected]
