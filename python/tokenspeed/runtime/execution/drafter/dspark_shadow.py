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

"""Bounded host-side trace for calibrating K3 DSpark confidence."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import torch


class DSparkShadowTrace:
    """Pair a draft block's confidence with its next exact target verification.

    A confidence block is emitted while drafting the *following* verify window,
    whereas its labels arrive on that window's next target step. Pending rows
    are therefore retained by request ID in host memory. The trace deliberately
    stores only logits and accepted-prefix length, never prompt or output text.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        candidate_count: int,
        max_records: int,
    ) -> None:
        if candidate_count < 1:
            raise ValueError("candidate_count must be positive")
        if max_records < 1:
            raise ValueError("max_records must be positive")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._handle = self.path.open("x", encoding="utf-8")
        except FileExistsError as exc:
            raise ValueError(
                f"DSpark shadow trace already exists: {self.path}"
            ) from exc
        self.candidate_count = candidate_count
        self.max_records = max_records
        self._pending: dict[str, list[float]] = {}
        self._records_written = 0
        self._write(
            {
                "schema_version": 1,
                "kind": "dspark_shadow_trace_header",
                "candidate_count": candidate_count,
                "max_records": max_records,
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
        request_ids: Sequence[str],
        num_extends: int,
        accept_lengths: torch.Tensor,
        next_confidence_logits: torch.Tensor | None,
    ) -> None:
        """Record labels for the current decode rows and stage next logits."""

        batch_size = len(request_ids)
        if num_extends < 0 or num_extends > batch_size:
            raise ValueError("num_extends must lie within the active batch")
        if accept_lengths.device.type != "cpu" or accept_lengths.numel() != batch_size:
            raise ValueError(
                "DSpark shadow accept_lengths must be a CPU tensor aligned with the batch"
            )
        if next_confidence_logits is not None:
            if next_confidence_logits.device.type != "cpu" or tuple(
                next_confidence_logits.shape
            ) != (batch_size, self.candidate_count):
                raise ValueError(
                    "DSpark shadow confidence logits must be a CPU tensor with "
                    "shape [batch, candidate_count]"
                )

        accept_values = [int(value) for value in accept_lengths.tolist()]
        for index in range(num_extends, batch_size):
            confidence = self._pending.pop(request_ids[index], None)
            if confidence is None or self._records_written >= self.max_records:
                continue
            accepted_draft_tokens = min(
                self.candidate_count,
                max(0, accept_values[index] - 1),
            )
            self._write(
                {
                    "schema_version": 1,
                    "kind": "dspark_shadow_trace_record",
                    "confidence_logits": confidence,
                    "accepted_draft_tokens": accepted_draft_tokens,
                }
            )
            self._records_written += 1

        if next_confidence_logits is None or self._records_written >= self.max_records:
            return
        for index, request_id in enumerate(request_ids):
            self._pending[request_id] = [
                float(value) for value in next_confidence_logits[index].tolist()
            ]
        if self._records_written % 128 == 0:
            self._handle.flush()

    def _write(self, record: dict[str, object]) -> None:
        self._handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
