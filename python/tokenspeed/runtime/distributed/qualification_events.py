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

"""Stable, content-free startup events for distributed qualification."""

from __future__ import annotations

import json
import logging
from typing import Any

from tokenspeed.runtime.distributed.mapping import Mapping

QUALIFICATION_EVENT_SCHEMA = "tokenspeed.qualification"
QUALIFICATION_EVENT_SCHEMA_VERSION = 1
_event_logger = logging.getLogger("tokenspeed.qualification")
_event_logger.propagate = False
_event_logger.setLevel(logging.INFO)
if not _event_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _event_logger.addHandler(_handler)


def _emit_success_event(
    event: str,
    **fields: Any,
) -> None:
    payload = {
        "event": event,
        "schema": QUALIFICATION_EVENT_SCHEMA,
        "schema_version": QUALIFICATION_EVENT_SCHEMA_VERSION,
        "status": "success",
        **fields,
    }
    _event_logger.info(
        "%s",
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True),
    )


def emit_distributed_topology_success(
    mapping: Mapping,
    *,
    cuda_device: str,
    process_group_backend: str,
) -> None:
    """Emit the rank-local topology after distributed initialization succeeds."""

    _emit_success_event(
        "distributed_topology",
        cuda_device=cuda_device,
        global_rank=mapping.rank,
        local_rank=mapping.local_rank,
        node_rank=mapping.node_rank,
        pp_stage=mapping.pipeline.stage_index,
        pp_stage_count=mapping.pipeline.stage_count,
        process_group_backend=process_group_backend,
        stage_local_rank=mapping.pipeline.stage_local_rank,
        stage_world_size=mapping.pipeline.stage_world_size,
        tp_group=list(mapping.attn.tp_group),
        tp_rank=mapping.attn.tp_rank,
        vision_dp_group=list(mapping.vision.dp_group),
        vision_dp_rank=mapping.vision.dp_rank,
        vision_tp_group=list(mapping.vision.tp_group),
        vision_tp_rank=mapping.vision.tp_rank,
        world_size=mapping.world_size,
    )


def emit_pipeline_plan_consensus_success(
    mapping: Mapping,
    *,
    pipeline_plan_digest: str,
) -> None:
    """Emit the complete plan digest after global consensus succeeds."""

    _emit_success_event(
        "pipeline_plan_consensus",
        global_rank=mapping.rank,
        pipeline_plan_digest=pipeline_plan_digest,
        pp_stage=mapping.pipeline.stage_index,
        stage_local_rank=mapping.pipeline.stage_local_rank,
        world_size=mapping.world_size,
    )


def emit_kimi_k3_cache_consensus_success(
    mapping: Mapping,
    *,
    consensus_phase: str,
    global_abi_digest: str,
    stage_abi_digest: str,
) -> None:
    """Emit complete global and stage Kimi-K3 cache ABI digests."""

    _emit_success_event(
        "kimi_k3_cache_abi_consensus",
        consensus_phase=consensus_phase,
        global_abi_digest=global_abi_digest,
        global_rank=mapping.rank,
        pp_stage=mapping.pipeline.stage_index,
        stage_abi_digest=stage_abi_digest,
        stage_local_rank=mapping.pipeline.stage_local_rank,
        world_size=mapping.world_size,
    )
