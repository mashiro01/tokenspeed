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

"""Bounded PP-complete decode timing trace for K3 DSpark SPS calibration."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

import torch


class DSparkTargetSPSTrace:
    """Write exact, host-observed target-step timing records on rank zero."""

    def __init__(
        self,
        path: str | Path,
        *,
        max_verify_width: int,
        max_records: int,
        pipeline_stage_count: int,
        benchmark_verify_widths: Sequence[int] | None,
    ) -> None:
        if max_verify_width < 1:
            raise ValueError("max_verify_width must be positive")
        if max_records < 1:
            raise ValueError("max_records must be positive")
        if pipeline_stage_count < 1:
            raise ValueError("pipeline_stage_count must be positive")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._handle = self.path.open("x", encoding="utf-8")
        except FileExistsError as exc:
            raise ValueError(
                f"DSpark target SPS trace already exists: {self.path}"
            ) from exc
        self.max_verify_width = max_verify_width
        self.max_records = max_records
        self._records_written = 0
        self._write(
            {
                "schema_version": 1,
                "kind": "dspark_target_sps_trace_header",
                "max_verify_width": max_verify_width,
                "max_records": max_records,
                "pipeline_stage_count": pipeline_stage_count,
                "benchmark_verify_widths": (
                    None
                    if benchmark_verify_widths is None
                    else list(benchmark_verify_widths)
                ),
            }
        )
        self._handle.flush()

    @property
    def records_written(self) -> int:
        return self._records_written

    def flush(self) -> None:
        self._handle.flush()

    def record(
        self,
        *,
        input_lengths: Sequence[int],
        num_extends: int,
        accept_lengths: torch.Tensor | None,
        elapsed_ms: float,
    ) -> None:
        """Record one synchronized pure-decode target forward, when enabled."""

        if self._records_written >= self.max_records or num_extends != 0:
            return
        if not math.isfinite(elapsed_ms) or elapsed_ms <= 0.0:
            raise ValueError("DSpark target SPS elapsed_ms must be finite and positive")
        widths = [int(width) for width in input_lengths]
        if not widths:
            return
        if any(width < 1 or width > self.max_verify_width for width in widths):
            raise ValueError(
                "DSpark target SPS widths must lie within the configured verify range"
            )
        accepted_tokens = None
        if accept_lengths is not None:
            if accept_lengths.device.type != "cpu" or accept_lengths.numel() != len(
                widths
            ):
                raise ValueError(
                    "DSpark target SPS accept lengths must be CPU-aligned with decode rows"
                )
            accepted_tokens = [int(value) for value in accept_lengths.tolist()]
        self._write(
            {
                "schema_version": 1,
                "kind": "dspark_target_sps_trace_record",
                "batch_size": len(widths),
                "verify_widths": widths,
                "target_tokens": sum(widths),
                "elapsed_ms": elapsed_ms,
                "accept_lengths": accepted_tokens,
            }
        )
        self._records_written += 1
        if self._records_written % 128 == 0:
            self._handle.flush()

    def _write(self, record: dict[str, object]) -> None:
        self._handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
