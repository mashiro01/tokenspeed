from __future__ import annotations

import json

import pytest
import torch

from tokenspeed.runtime.execution.drafter.dspark_shadow import DSparkShadowTrace


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_shadow_trace_pairs_next_draft_confidence_with_next_acceptance(tmp_path):
    path = tmp_path / "shadow.jsonl"
    trace = DSparkShadowTrace(path, candidate_count=3, max_records=8)

    trace.record(
        request_ids=["request-a"],
        num_extends=1,
        accept_lengths=torch.tensor([1], dtype=torch.int32),
        next_confidence_logits=torch.tensor([[0.2, -0.3, 0.8]]),
    )
    trace.record(
        request_ids=["request-a"],
        num_extends=0,
        accept_lengths=torch.tensor([3], dtype=torch.int32),
        next_confidence_logits=torch.tensor([[0.1, 0.4, -0.7]]),
    )
    trace.flush()

    header, record = _records(path)
    assert header == {
        "candidate_count": 3,
        "kind": "dspark_shadow_trace_header",
        "max_records": 8,
        "schema_version": 1,
    }
    assert record["accepted_draft_tokens"] == 2
    assert record["kind"] == "dspark_shadow_trace_record"
    assert record["schema_version"] == 1
    assert record["confidence_logits"] == pytest.approx([0.2, -0.3, 0.8])


def test_shadow_trace_stops_after_the_configured_record_limit(tmp_path):
    trace = DSparkShadowTrace(
        tmp_path / "bounded.jsonl", candidate_count=2, max_records=1
    )
    logits = torch.tensor([[0.0, 0.0]])

    trace.record(
        request_ids=["a"],
        num_extends=1,
        accept_lengths=torch.tensor([1], dtype=torch.int32),
        next_confidence_logits=logits,
    )
    trace.record(
        request_ids=["a"],
        num_extends=0,
        accept_lengths=torch.tensor([2], dtype=torch.int32),
        next_confidence_logits=logits,
    )
    trace.record(
        request_ids=["a"],
        num_extends=0,
        accept_lengths=torch.tensor([2], dtype=torch.int32),
        next_confidence_logits=logits,
    )

    assert trace.records_written == 1


def test_shadow_trace_rejects_misaligned_gpu_or_shape_inputs(tmp_path):
    trace = DSparkShadowTrace(
        tmp_path / "invalid.jsonl", candidate_count=2, max_records=1
    )

    with pytest.raises(ValueError, match="shape"):
        trace.record(
            request_ids=["a"],
            num_extends=0,
            accept_lengths=torch.tensor([1], dtype=torch.int32),
            next_confidence_logits=torch.zeros((1, 3)),
        )
