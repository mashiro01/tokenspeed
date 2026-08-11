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

"""Coverage for bounded K3 DSpark target-forward timing traces."""

from __future__ import annotations

import json
from types import SimpleNamespace

import torch

from tokenspeed.runtime.engine.event_loop import EventLoop
from tokenspeed.runtime.execution.drafter.dspark_sps_trace import DSparkTargetSPSTrace


def test_target_sps_trace_records_synchronized_pure_decode_only(tmp_path) -> None:
    path = tmp_path / "target-sps.jsonl"
    trace = DSparkTargetSPSTrace(
        path,
        max_verify_width=4,
        max_records=2,
        pipeline_stage_count=8,
        benchmark_verify_widths=(1, 2, 3, 4),
    )

    trace.record(
        input_lengths=[4],
        num_extends=1,
        accept_lengths=torch.tensor([1], dtype=torch.int32),
        elapsed_ms=1.0,
    )
    trace.record(
        input_lengths=[3, 1],
        num_extends=0,
        accept_lengths=torch.tensor([2, 1], dtype=torch.int32),
        elapsed_ms=12.5,
    )
    trace.flush()

    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert trace.records_written == 1
    assert records[0] == {
        "benchmark_verify_widths": [1, 2, 3, 4],
        "kind": "dspark_target_sps_trace_header",
        "max_records": 2,
        "max_verify_width": 4,
        "pipeline_stage_count": 8,
        "schema_version": 1,
    }
    assert records[1] == {
        "accept_lengths": [2, 1],
        "batch_size": 2,
        "elapsed_ms": 12.5,
        "kind": "dspark_target_sps_trace_record",
        "schema_version": 1,
        "target_tokens": 4,
        "verify_widths": [3, 1],
    }


def test_event_loop_records_target_sps_after_post_process_sync() -> None:
    events: list[str] = []
    recorded: list[dict[str, float]] = []
    loop = EventLoop.__new__(EventLoop)
    loop.request_handler = SimpleNamespace(
        forward_ct=0,
        _profile_batch_predicate=lambda _forward_mode: events.append("profile"),
    )
    loop.kv_transfer = None
    loop.output_processor = SimpleNamespace(
        post_process_forward_op=lambda *_args, **_kwargs: events.append("post") or []
    )
    loop.model_executor = SimpleNamespace(
        record_dspark_shadow_step=lambda *_args: events.append("shadow"),
        record_dspark_target_sps_step=lambda *_args, **kwargs: recorded.append(kwargs),
        accumulate_decode_stats=lambda *_args: events.append("stats"),
    )
    forward_op = SimpleNamespace(
        request_ids=["request-1"],
        input_lengths=[2],
        num_extends=lambda: 0,
    )
    results = SimpleNamespace(output_lengths=torch.tensor([1], dtype=torch.int32))

    request_changes = EventLoop._commit_forward_results(
        loop,
        forward_op,
        results,
        forward_started_at=0.0,
    )

    assert request_changes == []
    assert events == ["profile", "post", "shadow", "stats"]
    assert len(recorded) == 1
    assert recorded[0]["elapsed_ms"] > 0.0
