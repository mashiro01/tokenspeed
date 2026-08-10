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

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tokenspeed.runtime.distributed import qualification_events
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.attention.kv_cache.recipes import kimi_k3
from tokenspeed.runtime.pipeline import torch_transport
from tokenspeed.runtime.pipeline.contracts import PipelineProtocolError


class _CollectingLogger:
    def __init__(self):
        self.messages = []

    def info(self, message, *args):
        self.messages.append(message % args if args else message)


def _mapping(rank: int, *, world_size: int = 64) -> Mapping:
    return Mapping(
        rank=rank,
        world_size=world_size,
        attn_tp_size=8 if world_size == 64 else 1,
        dense_tp_size=8 if world_size == 64 else 1,
        moe_tp_size=8 if world_size == 64 else 1,
        vision_tp_size=1,
        vision_dp_size=8 if world_size == 64 else 1,
        pipeline_parallel_size=8 if world_size == 64 else world_size,
        nprocs_per_node=8 if world_size == 64 else 1,
        nnodes=8 if world_size == 64 else world_size,
    )


def _payload(message: str) -> dict:
    assert "\n" not in message
    payload = json.loads(message)
    assert payload["schema"] == qualification_events.QUALIFICATION_EVENT_SCHEMA
    assert payload["schema_version"] == 1
    assert payload["status"] == "success"
    return payload


def test_topology_event_is_parseable_and_complete_for_every_rank(monkeypatch):
    logger = _CollectingLogger()
    monkeypatch.setattr(qualification_events, "_event_logger", logger)

    for rank in range(64):
        qualification_events.emit_distributed_topology_success(
            _mapping(rank),
            cuda_device=f"cuda:{rank % 8}",
            process_group_backend="nccl",
        )

    assert len(logger.messages) == 64
    for rank, message in enumerate(logger.messages):
        payload = _payload(message)
        stage = rank // 8
        stage_group = list(range(stage * 8, stage * 8 + 8))
        assert payload == {
            "cuda_device": f"cuda:{rank % 8}",
            "event": "distributed_topology",
            "global_rank": rank,
            "local_rank": rank % 8,
            "node_rank": rank // 8,
            "pp_stage": stage,
            "pp_stage_count": 8,
            "process_group_backend": "nccl",
            "schema": qualification_events.QUALIFICATION_EVENT_SCHEMA,
            "schema_version": 1,
            "stage_local_rank": rank % 8,
            "stage_world_size": 8,
            "status": "success",
            "tp_group": stage_group,
            "tp_rank": rank % 8,
            "vision_dp_group": stage_group,
            "vision_dp_rank": rank % 8,
            "vision_tp_group": [rank],
            "vision_tp_rank": 0,
            "world_size": 64,
        }


def test_pipeline_plan_event_follows_successful_consensus(monkeypatch):
    logger = _CollectingLogger()
    mapping = _mapping(1, world_size=2)
    digest = "a" * 64
    monkeypatch.setattr(qualification_events, "_event_logger", logger)
    monkeypatch.setattr(
        torch_transport.pg_manager,
        "get_process_group",
        lambda *_args, **_kwargs: "world-gloo",
    )

    def all_gather(gathered, local, *, group):
        assert group == "world-gloo"
        for candidate in gathered:
            candidate.copy_(local)

    monkeypatch.setattr(torch_transport.dist, "all_gather", all_gather)

    torch_transport.validate_pipeline_plan_consensus(
        SimpleNamespace(digest=digest), mapping
    )

    assert _payload(logger.messages[0]) == {
        "event": "pipeline_plan_consensus",
        "global_rank": 1,
        "pipeline_plan_digest": digest,
        "pp_stage": 1,
        "schema": qualification_events.QUALIFICATION_EVENT_SCHEMA,
        "schema_version": 1,
        "stage_local_rank": 0,
        "status": "success",
        "world_size": 2,
    }


def test_pipeline_plan_failure_does_not_emit_success(monkeypatch):
    logger = _CollectingLogger()
    mapping = _mapping(0, world_size=2)
    monkeypatch.setattr(qualification_events, "_event_logger", logger)
    monkeypatch.setattr(
        torch_transport.pg_manager,
        "get_process_group",
        lambda *_args, **_kwargs: "world-gloo",
    )

    def all_gather(gathered, local, *, group):
        del group
        for candidate in gathered:
            candidate.copy_(local)
        gathered[1][0] ^= 1

    monkeypatch.setattr(torch_transport.dist, "all_gather", all_gather)

    with pytest.raises(PipelineProtocolError, match=r"global ranks \[1\]"):
        torch_transport.validate_pipeline_plan_consensus(
            SimpleNamespace(digest="b" * 64), mapping
        )

    assert logger.messages == []


@pytest.mark.parametrize("phase", ["layout", "runtime"])
def test_kimi_k3_cache_event_contains_complete_consensus_digests(monkeypatch, phase):
    logger = _CollectingLogger()
    mapping = _mapping(1, world_size=2)
    global_digest = "c" * 64
    stage_digest = "d" * 64
    monkeypatch.setattr(qualification_events, "_event_logger", logger)
    monkeypatch.setattr(
        kimi_k3.pg_manager,
        "get_process_group",
        lambda *_args, **_kwargs: "world-gloo",
    )

    def all_gather(gathered, local, *, group):
        assert group == "world-gloo"
        for candidate in gathered:
            candidate.copy_(local)

    monkeypatch.setattr(kimi_k3.dist, "all_gather", all_gather)

    kimi_k3._validate_pipeline_cache_digest_consensus(
        global_digest,
        stage_digest,
        mapping,
        consensus_phase=phase,
    )

    assert _payload(logger.messages[0]) == {
        "consensus_phase": phase,
        "event": "kimi_k3_cache_abi_consensus",
        "global_abi_digest": global_digest,
        "global_rank": 1,
        "pp_stage": 1,
        "schema": qualification_events.QUALIFICATION_EVENT_SCHEMA,
        "schema_version": 1,
        "stage_abi_digest": stage_digest,
        "stage_local_rank": 0,
        "status": "success",
        "world_size": 2,
    }


def test_kimi_k3_cache_failure_does_not_emit_success(monkeypatch):
    logger = _CollectingLogger()
    mapping = _mapping(0, world_size=2)
    monkeypatch.setattr(qualification_events, "_event_logger", logger)
    monkeypatch.setattr(
        kimi_k3.pg_manager,
        "get_process_group",
        lambda *_args, **_kwargs: "world-gloo",
    )

    def all_gather(gathered, local, *, group):
        del group
        for candidate in gathered:
            candidate.copy_(local)
        gathered[1][0] ^= 1

    monkeypatch.setattr(kimi_k3.dist, "all_gather", all_gather)

    with pytest.raises(RuntimeError, match=r"global ABI ranks=\[1\]"):
        kimi_k3._validate_pipeline_cache_digest_consensus(
            "e" * 64,
            "f" * 64,
            mapping,
            consensus_phase="runtime",
        )

    assert logger.messages == []
