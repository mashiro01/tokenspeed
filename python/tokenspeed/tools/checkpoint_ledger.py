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

"""Build a CPU-free, value-free byte ledger from safetensors metadata.

The tool reads only the safetensors JSON header and an optional Hugging Face
index. A declarative plan supplies module ownership and TP/EP routing. It never
imports torch or materializes tensor payloads.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import stat
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Pattern, Sequence

SCHEMA_VERSION = 1
MAX_HEADER_BYTES = 100_000_000
MAX_INDEX_BYTES = 128 * 1024 * 1024
MAX_PLAN_BYTES = 16 * 1024 * 1024
MAX_JSON_DEPTH = 128
MAX_PATTERN_LENGTH = 4096
MAX_TENSOR_NAME_LENGTH = 16 * 1024
MAX_SHARD_PATH_LENGTH = 4096
MAX_TENSOR_DIMENSIONS = 64
MAX_TENSOR_DIMENSION_SIZE = (1 << 63) - 1
MAX_TENSORS = 1_000_000
MAX_SHARDS = 100_000
MAX_STAGES = 4096
MAX_RANKS = 65_536
MAX_RULES = 100_000
MAX_PLAN_STRING_LENGTH = 16 * 1024
CLASSIFICATIONS = ("exact", "replicated", "sharded", "unknown")
PARALLEL_FIELDS = {"tp": "tp_rank", "ep": "ep_rank"}

# Safetensors stores fixed-width values. Unknown future dtypes remain readable,
# but they cannot be partitioned with a byte-exact proof.
DTYPE_BITS = {
    "BOOL": 8,
    "U8": 8,
    "I8": 8,
    "F8_E4M3": 8,
    "F8_E4M3FN": 8,
    "F8_E4M3FNUZ": 8,
    "F8_E5M2": 8,
    "F8_E5M2FNUZ": 8,
    "F8_E8M0": 8,
    "F8_E8M0FNU": 8,
    "U16": 16,
    "I16": 16,
    "F16": 16,
    "BF16": 16,
    "U32": 32,
    "I32": 32,
    "F32": 32,
    "U64": 64,
    "I64": 64,
    "F64": 64,
    "C64": 64,
    "C128": 128,
    "F4_E2M1": 4,
    "F6_E2M3": 6,
    "F6_E3M2": 6,
}


class LedgerError(ValueError):
    """Base error for deterministic CLI diagnostics."""


class PlanValidationError(LedgerError):
    """Raised when a ledger plan is structurally invalid."""


class CheckpointFormatError(LedgerError):
    """Raised when a safetensors file or index is structurally invalid."""


class _DuplicateKeyError(ValueError):
    pass


@dataclass(frozen=True)
class TensorInfo:
    """Metadata for one tensor payload without its values."""

    name: str
    file: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    dtype_bits: int | None
    proof_error: str | None


@dataclass(frozen=True)
class RankSpec:
    rank: int
    tp_rank: int
    ep_rank: int


@dataclass(frozen=True)
class StageSpec:
    id: str
    ranks: tuple[RankSpec, ...]

    def parallel_size(self, by: str) -> int:
        field = PARALLEL_FIELDS[by]
        return max(getattr(rank, field) for rank in self.ranks) + 1


@dataclass(frozen=True)
class OwnershipRule:
    pattern_text: str
    pattern: Pattern[str]
    stage: str
    ranks: tuple[int, ...] | None


@dataclass(frozen=True)
class PartitionSpec:
    axis: int
    by: str
    parts: int | None
    start: int
    length: int | None


@dataclass(frozen=True)
class OwnerSpec:
    by: str
    capture: str
    count: int


@dataclass(frozen=True)
class RouteRule:
    pattern_text: str
    pattern: Pattern[str]
    kind: str
    owner: OwnerSpec | None
    partitions: tuple[PartitionSpec, ...]
    reason: str | None


@dataclass(frozen=True)
class LedgerPlan:
    version: int
    stages: tuple[StageSpec, ...]
    ownership: tuple[OwnershipRule, ...]
    routes: tuple[RouteRule, ...]


@dataclass(frozen=True)
class _RankTarget:
    stage: StageSpec
    rank: RankSpec


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _decode_json(text: str, context: str, error_type: type[LedgerError]) -> Any:
    depth = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > MAX_JSON_DEPTH:
                raise error_type(
                    f"{context}: JSON nesting exceeds {MAX_JSON_DEPTH} levels"
                )
        elif character in "]}":
            depth -= 1

    try:
        return json.loads(text, object_pairs_hook=_object_without_duplicate_keys)
    except (ValueError, RecursionError, OverflowError) as error:
        raise error_type(f"{context}: invalid JSON: {error}") from error


def _read_utf8_file(
    path: Path,
    *,
    kind: str,
    maximum_bytes: int,
    error_type: type[LedgerError],
) -> str:
    try:
        with path.open("rb") as file:
            file_stat = os.fstat(file.fileno())
            if not stat.S_ISREG(file_stat.st_mode):
                raise error_type(f"{path}: {kind} must be a regular file")
            if file_stat.st_size > maximum_bytes:
                raise error_type(
                    f"{path}: {kind} size {file_stat.st_size} exceeds "
                    f"limit {maximum_bytes}"
                )
            raw = file.read(maximum_bytes + 1)
            if len(raw) != file_stat.st_size:
                raise error_type(f"{path}: {kind} changed while being read")
            final_stat = os.fstat(file.fileno())
            if (
                final_stat.st_dev != file_stat.st_dev
                or final_stat.st_ino != file_stat.st_ino
                or final_stat.st_size != file_stat.st_size
                or final_stat.st_mtime_ns != file_stat.st_mtime_ns
            ):
                raise error_type(f"{path}: {kind} changed while being read")
    except LedgerError:
        raise
    except (OSError, ValueError) as error:
        raise error_type(f"cannot read {kind} {path}: {error}") from error

    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise error_type(f"{path}: {kind} is not UTF-8") from error


def _require_object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PlanValidationError(f"{context} must be an object")
    return value


def _require_list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise PlanValidationError(f"{context} must be an array")
    return value


def _require_string(
    value: Any,
    context: str,
    maximum_length: int = MAX_PLAN_STRING_LENGTH,
) -> str:
    if not isinstance(value, str) or not value:
        raise PlanValidationError(f"{context} must be a non-empty string")
    if len(value) > maximum_length:
        raise PlanValidationError(f"{context} exceeds maximum length {maximum_length}")
    return value


def _require_int(value: Any, context: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PlanValidationError(f"{context} must be an integer >= {minimum}")
    return value


def _validate_keys(
    value: dict[str, Any],
    context: str,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    missing = sorted(required - value.keys())
    if missing:
        raise PlanValidationError(f"{context} is missing keys: {missing}")
    unexpected = sorted(value.keys() - required - optional)
    if unexpected:
        raise PlanValidationError(f"{context} has unexpected keys: {unexpected}")


def _repeat_end(pattern_text: str, index: int) -> int | None:
    if index >= len(pattern_text):
        return None
    if pattern_text[index] in "*+?":
        return index + 1
    if pattern_text[index] != "{":
        return None

    cursor = index + 1
    digit_start = cursor
    while cursor < len(pattern_text) and pattern_text[cursor].isdigit():
        cursor += 1
    if cursor == digit_start:
        return None
    if cursor < len(pattern_text) and pattern_text[cursor] == ",":
        cursor += 1
        while cursor < len(pattern_text) and pattern_text[cursor].isdigit():
            cursor += 1
    if cursor >= len(pattern_text) or pattern_text[cursor] != "}":
        return None
    return cursor + 1


def _has_high_risk_group_quantifier(pattern_text: str) -> bool:
    group_has_repeat = [False]
    group_has_alternation = [False]
    index = 0
    while index < len(pattern_text):
        character = pattern_text[index]
        if character == "\\":
            index += 2
            continue
        if character == "[":
            index += 1
            while index < len(pattern_text):
                if pattern_text[index] == "\\":
                    index += 2
                elif pattern_text[index] == "]":
                    index += 1
                    break
                else:
                    index += 1
            continue
        if character == "(":
            group_has_repeat.append(False)
            group_has_alternation.append(False)
            index += 2 if pattern_text[index : index + 2] == "(?" else 1
            continue
        if character == ")" and len(group_has_repeat) > 1:
            child_has_repeat = group_has_repeat.pop()
            child_has_alternation = group_has_alternation.pop()
            repeat_end = _repeat_end(pattern_text, index + 1)
            if repeat_end is not None:
                if child_has_repeat or child_has_alternation:
                    return True
                group_has_repeat[-1] = True
                index = repeat_end
                if index < len(pattern_text) and pattern_text[index] in "?+":
                    index += 1
                continue
            if child_has_repeat:
                group_has_repeat[-1] = True
            if child_has_alternation:
                group_has_alternation[-1] = True
            index += 1
            continue
        if character == "|":
            group_has_alternation[-1] = True
            index += 1
            continue

        repeat_end = _repeat_end(pattern_text, index)
        if repeat_end is not None:
            group_has_repeat[-1] = True
            index = repeat_end
            if index < len(pattern_text) and pattern_text[index] in "?+":
                index += 1
            continue
        index += 1
    return False


def _enables_verbose_regex(pattern_text: str) -> bool:
    index = 0
    while index < len(pattern_text):
        if pattern_text[index] == "\\":
            index += 2
            continue
        if pattern_text[index] == "[":
            index += 1
            while index < len(pattern_text):
                if pattern_text[index] == "\\":
                    index += 2
                elif pattern_text[index] == "]":
                    index += 1
                    break
                else:
                    index += 1
            continue
        if pattern_text[index : index + 2] == "(?":
            cursor = index + 2
            enabled_flags: list[str] = []
            while cursor < len(pattern_text) and pattern_text[cursor] in "aiLmsux":
                enabled_flags.append(pattern_text[cursor])
                cursor += 1
            if "x" in enabled_flags:
                return True
        index += 1
    return False


def _compile_pattern(pattern_text: str, context: str) -> Pattern[str]:
    if len(pattern_text) > MAX_PATTERN_LENGTH:
        raise PlanValidationError(
            f"{context}.pattern exceeds maximum length {MAX_PATTERN_LENGTH}"
        )
    if _enables_verbose_regex(pattern_text):
        raise PlanValidationError(f"{context}.pattern cannot enable verbose regex mode")
    if _has_high_risk_group_quantifier(pattern_text):
        raise PlanValidationError(
            f"{context}.pattern contains a high-risk quantified group"
        )
    try:
        return re.compile(pattern_text)
    except (re.error, RecursionError, OverflowError) as error:
        raise PlanValidationError(f"{context} has invalid regex: {error}") from error


def _parse_stages(value: Any) -> tuple[StageSpec, ...]:
    raw_stages = _require_list(value, "plan.stages")
    if not raw_stages:
        raise PlanValidationError("plan.stages must not be empty")
    if len(raw_stages) > MAX_STAGES:
        raise PlanValidationError(f"plan.stages contains more than {MAX_STAGES} stages")

    stages: list[StageSpec] = []
    stage_ids: set[str] = set()
    global_ranks: set[int] = set()
    for stage_index, raw_stage in enumerate(raw_stages):
        context = f"plan.stages[{stage_index}]"
        stage_data = _require_object(raw_stage, context)
        _validate_keys(stage_data, context, {"id", "ranks"})
        stage_id = _require_string(stage_data["id"], f"{context}.id")
        if stage_id in stage_ids:
            raise PlanValidationError(f"duplicate stage id {stage_id!r}")
        stage_ids.add(stage_id)

        raw_ranks = _require_list(stage_data["ranks"], f"{context}.ranks")
        if not raw_ranks:
            raise PlanValidationError(f"{context}.ranks must not be empty")
        if len(raw_ranks) > MAX_RANKS:
            raise PlanValidationError(
                f"{context}.ranks contains more than {MAX_RANKS} ranks"
            )
        if len(global_ranks) + len(raw_ranks) > MAX_RANKS:
            raise PlanValidationError(
                f"plan contains more than {MAX_RANKS} global ranks"
            )
        ranks: list[RankSpec] = []
        coordinates: set[tuple[int, int]] = set()
        for rank_index, raw_rank in enumerate(raw_ranks):
            rank_context = f"{context}.ranks[{rank_index}]"
            rank_data = _require_object(raw_rank, rank_context)
            _validate_keys(rank_data, rank_context, {"rank", "tp_rank", "ep_rank"})
            rank = RankSpec(
                rank=_require_int(rank_data["rank"], f"{rank_context}.rank"),
                tp_rank=_require_int(rank_data["tp_rank"], f"{rank_context}.tp_rank"),
                ep_rank=_require_int(rank_data["ep_rank"], f"{rank_context}.ep_rank"),
            )
            if rank.rank in global_ranks:
                raise PlanValidationError(f"duplicate global rank {rank.rank}")
            global_ranks.add(rank.rank)
            coordinate = (rank.tp_rank, rank.ep_rank)
            if coordinate in coordinates:
                raise PlanValidationError(
                    f"{context} has duplicate TP/EP coordinate {coordinate}"
                )
            coordinates.add(coordinate)
            ranks.append(rank)

        tp_values = {rank.tp_rank for rank in ranks}
        ep_values = {rank.ep_rank for rank in ranks}
        if (
            max(tp_values) != len(tp_values) - 1
            or max(ep_values) != len(ep_values) - 1
            or len(coordinates) != len(tp_values) * len(ep_values)
        ):
            raise PlanValidationError(
                f"{context}.ranks must form a complete TP x EP grid"
            )
        stages.append(StageSpec(stage_id, tuple(sorted(ranks, key=lambda x: x.rank))))

    return tuple(sorted(stages, key=lambda stage: stage.id))


def _parse_ownership(
    value: Any, stages: tuple[StageSpec, ...]
) -> tuple[OwnershipRule, ...]:
    raw_rules = _require_list(value, "plan.ownership")
    if len(raw_rules) > MAX_RULES:
        raise PlanValidationError(
            f"plan.ownership contains more than {MAX_RULES} rules"
        )
    stage_map = {stage.id: stage for stage in stages}
    rules: list[OwnershipRule] = []
    for index, raw_rule in enumerate(raw_rules):
        context = f"plan.ownership[{index}]"
        rule_data = _require_object(raw_rule, context)
        _validate_keys(rule_data, context, {"pattern", "stage"}, {"ranks"})
        pattern_text = _require_string(rule_data["pattern"], f"{context}.pattern")
        pattern = _compile_pattern(pattern_text, context)
        stage_id = _require_string(rule_data["stage"], f"{context}.stage")
        if stage_id not in stage_map:
            raise PlanValidationError(
                f"{context}.stage references unknown stage {stage_id!r}"
            )

        ranks: tuple[int, ...] | None = None
        if "ranks" in rule_data:
            raw_ranks = _require_list(rule_data["ranks"], f"{context}.ranks")
            if not raw_ranks:
                raise PlanValidationError(f"{context}.ranks must not be empty")
            parsed_ranks = tuple(
                _require_int(rank, f"{context}.ranks[{rank_index}]")
                for rank_index, rank in enumerate(raw_ranks)
            )
            if len(set(parsed_ranks)) != len(parsed_ranks):
                raise PlanValidationError(f"{context}.ranks contains duplicates")
            stage_ranks = {rank.rank for rank in stage_map[stage_id].ranks}
            outside = sorted(set(parsed_ranks) - stage_ranks)
            if outside:
                raise PlanValidationError(
                    f"{context}.ranks are outside stage {stage_id!r}: {outside}"
                )
            ranks = tuple(sorted(parsed_ranks))
        rules.append(OwnershipRule(pattern_text, pattern, stage_id, ranks))
    return tuple(rules)


def _parse_owner(value: Any, route: Pattern[str], context: str) -> OwnerSpec:
    owner_data = _require_object(value, context)
    _validate_keys(owner_data, context, {"by", "capture", "count"})
    by = _require_string(owner_data["by"], f"{context}.by")
    if by not in PARALLEL_FIELDS:
        raise PlanValidationError(f"{context}.by must be 'tp' or 'ep'")
    capture = _require_string(owner_data["capture"], f"{context}.capture")
    if capture not in route.groupindex:
        raise PlanValidationError(
            f"{context}.capture must name a named capture in the route regex"
        )
    count = _require_int(owner_data["count"], f"{context}.count", minimum=1)
    return OwnerSpec(by, capture, count)


def _parse_partitions(value: Any, context: str) -> tuple[PartitionSpec, ...]:
    raw_partitions = _require_list(value, context)
    if not raw_partitions:
        raise PlanValidationError(f"{context} must not be empty")
    partitions: list[PartitionSpec] = []
    axes: set[int] = set()
    dimensions: set[str] = set()
    for index, raw_partition in enumerate(raw_partitions):
        item_context = f"{context}[{index}]"
        partition_data = _require_object(raw_partition, item_context)
        _validate_keys(
            partition_data,
            item_context,
            {"axis", "by"},
            {"parts", "start", "length"},
        )
        axis = _require_int(partition_data["axis"], f"{item_context}.axis")
        if axis >= MAX_TENSOR_DIMENSIONS:
            raise PlanValidationError(
                f"{item_context}.axis must be less than {MAX_TENSOR_DIMENSIONS}"
            )
        by = _require_string(partition_data["by"], f"{item_context}.by")
        if by not in PARALLEL_FIELDS:
            raise PlanValidationError(f"{item_context}.by must be 'tp' or 'ep'")
        if axis in axes:
            raise PlanValidationError(f"{context} contains duplicate axis {axis}")
        if by in dimensions:
            raise PlanValidationError(
                f"{context} partitions parallel dimension {by!r} more than once"
            )
        axes.add(axis)
        dimensions.add(by)
        parts = None
        if "parts" in partition_data:
            parts = _require_int(
                partition_data["parts"], f"{item_context}.parts", minimum=1
            )
        start = _require_int(partition_data.get("start", 0), f"{item_context}.start")
        length = None
        if "length" in partition_data:
            length = _require_int(
                partition_data["length"], f"{item_context}.length", minimum=1
            )
        partitions.append(PartitionSpec(axis, by, parts, start, length))
    return tuple(sorted(partitions, key=lambda partition: partition.axis))


def _parse_routes(value: Any) -> tuple[RouteRule, ...]:
    raw_routes = _require_list(value, "plan.routes")
    if len(raw_routes) > MAX_RULES:
        raise PlanValidationError(f"plan.routes contains more than {MAX_RULES} rules")
    routes: list[RouteRule] = []
    for index, raw_route in enumerate(raw_routes):
        context = f"plan.routes[{index}]"
        route_data = _require_object(raw_route, context)
        _validate_keys(
            route_data,
            context,
            {"pattern", "kind"},
            {"owner", "partitions", "reason"},
        )
        pattern_text = _require_string(route_data["pattern"], f"{context}.pattern")
        pattern = _compile_pattern(pattern_text, context)
        kind = _require_string(route_data["kind"], f"{context}.kind")
        if kind not in CLASSIFICATIONS:
            raise PlanValidationError(
                f"{context}.kind must be one of {list(CLASSIFICATIONS)}"
            )

        owner = None
        if "owner" in route_data:
            owner = _parse_owner(route_data["owner"], pattern, f"{context}.owner")
        partitions: tuple[PartitionSpec, ...] = ()
        if "partitions" in route_data:
            partitions = _parse_partitions(
                route_data["partitions"], f"{context}.partitions"
            )
        reason = None
        if "reason" in route_data:
            reason = _require_string(route_data["reason"], f"{context}.reason")

        if kind == "sharded":
            if owner is None and not partitions:
                raise PlanValidationError(
                    f"{context} kind 'sharded' requires owner or partitions"
                )
            if owner is not None and any(
                partition.by == owner.by for partition in partitions
            ):
                raise PlanValidationError(
                    f"{context} cannot own and partition by {owner.by!r}"
                )
        elif owner is not None or partitions:
            raise PlanValidationError(
                f"{context} kind {kind!r} does not accept owner or partitions"
            )
        if kind != "unknown" and reason is not None:
            raise PlanValidationError(
                f"{context}.reason is only valid for kind 'unknown'"
            )
        routes.append(RouteRule(pattern_text, pattern, kind, owner, partitions, reason))
    return tuple(routes)


def parse_plan(data: Any) -> LedgerPlan:
    """Validate a decoded plan and return its immutable representation.

    Args:
        data: JSON-compatible plan object.

    Returns:
        A validated ledger plan.

    Raises:
        PlanValidationError: If any schema or topology invariant is invalid.
    """
    plan_data = _require_object(data, "plan")
    _validate_keys(plan_data, "plan", {"version", "stages", "ownership", "routes"})
    version = _require_int(plan_data["version"], "plan.version", minimum=1)
    if version != SCHEMA_VERSION:
        raise PlanValidationError(
            f"plan.version must be {SCHEMA_VERSION}, got {version}"
        )
    stages = _parse_stages(plan_data["stages"])
    ownership = _parse_ownership(plan_data["ownership"], stages)
    routes = _parse_routes(plan_data["routes"])
    return LedgerPlan(version, stages, ownership, routes)


def load_plan(path: str | Path) -> LedgerPlan:
    """Load and validate a JSON ledger plan.

    Args:
        path: Path to the plan JSON file.

    Returns:
        A validated ledger plan.
    """
    plan_path = Path(path)
    text = _read_utf8_file(
        plan_path,
        kind="plan",
        maximum_bytes=MAX_PLAN_BYTES,
        error_type=PlanValidationError,
    )
    return parse_plan(_decode_json(text, str(plan_path), PlanValidationError))


def _checkpoint_json(text: str, context: str) -> dict[str, Any]:
    value = _decode_json(text, context, CheckpointFormatError)
    if not isinstance(value, dict):
        raise CheckpointFormatError(f"{context}: JSON root must be an object")
    return value


def _tensor_proof(
    dtype: str, shape: tuple[int, ...], nbytes: int
) -> tuple[int | None, str | None]:
    bits = DTYPE_BITS.get(dtype)
    if bits is None:
        return None, f"unknown safetensors dtype {dtype!r}"

    if 0 in shape:
        elements = 0
    else:
        maximum_elements = nbytes * 8 // bits
        elements = 1
        for size in shape:
            if elements > maximum_elements // size:
                return (
                    bits,
                    f"payload bytes {nbytes} are smaller than dtype/shape requires",
                )
            elements *= size

    expected_bytes = (elements * bits + 7) // 8
    if expected_bytes != nbytes:
        return (
            bits,
            f"payload bytes {nbytes} do not match dtype/shape bytes {expected_bytes}",
        )
    return bits, None


def _read_safetensors(path: Path, label: str) -> list[TensorInfo]:
    try:
        with path.open("rb") as file:
            initial_stat = os.fstat(file.fileno())
            if not stat.S_ISREG(initial_stat.st_mode):
                raise CheckpointFormatError(
                    f"{label}: safetensors input must be a regular file"
                )
            file_size = initial_stat.st_size
            prefix = file.read(8)
            if len(prefix) != 8:
                raise CheckpointFormatError(
                    f"{label}: file is too short for a safetensors header"
                )
            header_length = struct.unpack("<Q", prefix)[0]
            if header_length == 0 or header_length > MAX_HEADER_BYTES:
                raise CheckpointFormatError(
                    f"{label}: invalid header length {header_length}"
                )
            if 8 + header_length > file_size:
                raise CheckpointFormatError(
                    f"{label}: header length {header_length} exceeds file size"
                )
            header_bytes = file.read(header_length)
            if len(header_bytes) != header_length:
                raise CheckpointFormatError(
                    f"{label}: safetensors header ended after {len(header_bytes)} "
                    f"of {header_length} bytes"
                )
            final_stat = os.fstat(file.fileno())
            if (
                final_stat.st_dev != initial_stat.st_dev
                or final_stat.st_ino != initial_stat.st_ino
                or final_stat.st_size != initial_stat.st_size
                or final_stat.st_mtime_ns != initial_stat.st_mtime_ns
            ):
                raise CheckpointFormatError(
                    f"{label}: safetensors file changed while being read"
                )
    except CheckpointFormatError:
        raise
    except (OSError, ValueError) as error:
        raise CheckpointFormatError(
            f"cannot read safetensors file {path}: {error}"
        ) from error

    try:
        header_text = header_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CheckpointFormatError(f"{label}: header is not UTF-8") from error
    header = _checkpoint_json(header_text, f"{label} header")
    payload_bytes = file_size - 8 - header_length

    if "__metadata__" in header:
        metadata = header["__metadata__"]
        if not isinstance(metadata, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in metadata.items()
        ):
            raise CheckpointFormatError(
                f"{label}: __metadata__ must map strings to strings"
            )

    tensor_names = [key for key in header if key != "__metadata__"]
    if len(tensor_names) > MAX_TENSORS:
        raise CheckpointFormatError(
            f"{label}: header contains more than {MAX_TENSORS} tensors"
        )
    for name in tensor_names:
        if not name:
            raise CheckpointFormatError(f"{label}: tensor names must be non-empty")
        if len(name) > MAX_TENSOR_NAME_LENGTH:
            raise CheckpointFormatError(
                f"{label}: tensor name exceeds maximum length "
                f"{MAX_TENSOR_NAME_LENGTH}"
            )

    tensors: list[TensorInfo] = []
    ranges: list[tuple[int, int, str]] = []
    for name in sorted(tensor_names):
        raw_tensor = header[name]
        if not isinstance(raw_tensor, dict):
            raise CheckpointFormatError(f"{label}: tensor {name!r} must be an object")
        expected_keys = {"dtype", "shape", "data_offsets"}
        if set(raw_tensor) != expected_keys:
            raise CheckpointFormatError(
                f"{label}: tensor {name!r} must have exactly {sorted(expected_keys)}"
            )
        dtype = raw_tensor["dtype"]
        if not isinstance(dtype, str) or not dtype:
            raise CheckpointFormatError(f"{label}: tensor {name!r} has invalid dtype")
        raw_shape = raw_tensor["shape"]
        if not isinstance(raw_shape, list) or any(
            type(size) is not int or size < 0 or size > MAX_TENSOR_DIMENSION_SIZE
            for size in raw_shape
        ):
            raise CheckpointFormatError(f"{label}: tensor {name!r} has invalid shape")
        if len(raw_shape) > MAX_TENSOR_DIMENSIONS:
            raise CheckpointFormatError(
                f"{label}: tensor {name!r} shape has more than "
                f"{MAX_TENSOR_DIMENSIONS} dimensions"
            )
        raw_offsets = raw_tensor["data_offsets"]
        if (
            not isinstance(raw_offsets, list)
            or len(raw_offsets) != 2
            or any(type(offset) is not int for offset in raw_offsets)
        ):
            raise CheckpointFormatError(
                f"{label}: tensor {name!r} has invalid data_offsets"
            )
        start, stop = raw_offsets
        if start < 0 or stop < start or stop > payload_bytes:
            raise CheckpointFormatError(
                f"{label}: tensor {name!r} data_offsets are outside payload"
            )
        nbytes = stop - start
        shape = tuple(raw_shape)
        dtype_bits, proof_error = _tensor_proof(dtype, shape, nbytes)
        tensors.append(
            TensorInfo(
                name=name,
                file=label,
                dtype=dtype,
                shape=shape,
                nbytes=nbytes,
                dtype_bits=dtype_bits,
                proof_error=proof_error,
            )
        )
        if nbytes:
            ranges.append((start, stop, name))

    cursor = 0
    for start, stop, name in sorted(ranges):
        if start != cursor:
            relation = "overlaps prior tensor" if start < cursor else "leaves a gap"
            raise CheckpointFormatError(
                f"{label}: tensor {name!r} {relation} in the payload"
            )
        cursor = stop
    if cursor != payload_bytes:
        raise CheckpointFormatError(
            f"{label}: tensor ranges cover {cursor} of {payload_bytes} payload bytes"
        )
    return tensors


def _safe_relative_file(
    root: Path,
    relative_name: str,
    context: str,
    kind: str,
) -> tuple[Path, str]:
    if "\x00" in relative_name or len(relative_name) > MAX_SHARD_PATH_LENGTH:
        raise CheckpointFormatError(f"{context}: {kind} path is invalid or too long")
    try:
        relative = Path(relative_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise CheckpointFormatError(
                f"{context}: {kind} path must stay inside the checkpoint directory"
            )
        root_resolved = root.resolve(strict=True)
        resolved = (root_resolved / relative).resolve(strict=True)
        try:
            label = resolved.relative_to(root_resolved).as_posix()
        except ValueError as error:
            raise CheckpointFormatError(
                f"{context}: {kind} path must stay inside the checkpoint directory"
            ) from error
        if not resolved.is_file():
            raise CheckpointFormatError(
                f"{context}: {kind} {relative_name!r} is not a regular file"
            )
    except CheckpointFormatError:
        raise
    except FileNotFoundError as error:
        raise CheckpointFormatError(
            f"{context}: {kind} {relative_name!r} does not exist"
        ) from error
    except (OSError, RuntimeError, ValueError) as error:
        raise CheckpointFormatError(
            f"{context}: invalid {kind} path {relative_name!r}: {error}"
        ) from error
    return resolved, label


def _safe_index_shard(root: Path, shard_name: str, context: str) -> tuple[Path, str]:
    return _safe_relative_file(root, shard_name, context, "shard")


def _load_index(index_path: Path) -> list[TensorInfo]:
    text = _read_utf8_file(
        index_path,
        kind="index",
        maximum_bytes=MAX_INDEX_BYTES,
        error_type=CheckpointFormatError,
    )
    index = _checkpoint_json(text, str(index_path))
    unexpected = sorted(set(index) - {"metadata", "weight_map"})
    if unexpected:
        raise CheckpointFormatError(
            f"{index_path}: index has unexpected keys: {unexpected}"
        )
    if "weight_map" not in index or not isinstance(index["weight_map"], dict):
        raise CheckpointFormatError(f"{index_path}: weight_map must be an object")
    metadata: dict[str, Any] | None = None
    if "metadata" in index:
        raw_metadata = index["metadata"]
        if not isinstance(raw_metadata, dict):
            raise CheckpointFormatError(f"{index_path}: metadata must be an object")
        metadata = raw_metadata
    declared_total_size: int | None = None
    if metadata is not None and "total_size" in metadata:
        raw_total_size = metadata["total_size"]
        if type(raw_total_size) is not int or raw_total_size < 0:
            raise CheckpointFormatError(
                f"{index_path}: metadata.total_size must be a non-negative integer"
            )
        declared_total_size = raw_total_size

    weight_map = index["weight_map"]
    if not weight_map:
        raise CheckpointFormatError(f"{index_path}: weight_map must not be empty")
    if len(weight_map) > MAX_TENSORS:
        raise CheckpointFormatError(
            f"{index_path}: weight_map contains more than {MAX_TENSORS} tensors"
        )
    for name, shard_name in weight_map.items():
        if not isinstance(name, str) or not name:
            raise CheckpointFormatError(
                f"{index_path}: weight_map names must be non-empty strings"
            )
        if len(name) > MAX_TENSOR_NAME_LENGTH:
            raise CheckpointFormatError(
                f"{index_path}: tensor name exceeds maximum length "
                f"{MAX_TENSOR_NAME_LENGTH}"
            )
        if not isinstance(shard_name, str) or not shard_name:
            raise CheckpointFormatError(
                f"{index_path}: weight_map shard names must be non-empty strings"
            )

    root = index_path.parent
    shard_names = sorted(set(weight_map.values()))
    if len(shard_names) > MAX_SHARDS:
        raise CheckpointFormatError(
            f"{index_path}: weight_map references more than {MAX_SHARDS} shards"
        )
    shard_by_label: dict[str, Path] = {}
    label_by_shard_name: dict[str, str] = {}
    for shard_name in shard_names:
        shard, label = _safe_index_shard(root, shard_name, str(index_path))
        shard_by_label[label] = shard
        label_by_shard_name[shard_name] = label

    tensors: list[TensorInfo] = []
    locations: dict[str, str] = {}
    for label in sorted(shard_by_label):
        for tensor in _read_safetensors(shard_by_label[label], label):
            if tensor.name in locations:
                raise CheckpointFormatError(
                    f"tensor {tensor.name!r} appears in both "
                    f"{locations[tensor.name]!r} and {label!r}"
                )
            locations[tensor.name] = label
            tensors.append(tensor)
            if len(tensors) > MAX_TENSORS:
                raise CheckpointFormatError(
                    f"{index_path}: checkpoint contains more than "
                    f"{MAX_TENSORS} tensors"
                )

    for name, shard_name in sorted(weight_map.items()):
        expected_label = label_by_shard_name[shard_name]
        actual_label = locations.get(name)
        if actual_label is None:
            raise CheckpointFormatError(
                f"{index_path}: tensor {name!r} is not present in {expected_label!r}"
            )
        if actual_label != expected_label:
            raise CheckpointFormatError(
                f"{index_path}: tensor {name!r} maps to {expected_label!r} "
                f"but is present in {actual_label!r}"
            )
    extra_names = sorted(set(locations) - set(weight_map))
    if extra_names:
        name = extra_names[0]
        raise CheckpointFormatError(
            f"{index_path}: tensor {name!r} in {locations[name]!r} is absent "
            "from weight_map"
        )

    actual_total_size = sum(tensor.nbytes for tensor in tensors)
    if declared_total_size is not None and declared_total_size != actual_total_size:
        raise CheckpointFormatError(
            f"{index_path}: metadata.total_size is {declared_total_size}, "
            f"expected {actual_total_size}"
        )
    return sorted(tensors, key=lambda tensor: (tensor.name, tensor.file))


def load_checkpoint(path: str | Path) -> list[TensorInfo]:
    """Read tensor metadata from a safetensors file, index, or directory.

    Args:
        path: A ``.safetensors`` file, an index JSON file, or a checkpoint
            directory. A directory may contain one index or plain shards.

    Returns:
        Tensor metadata sorted by tensor name and shard label.

    Raises:
        CheckpointFormatError: If the checkpoint cannot be proven structurally
            valid.
    """
    checkpoint = Path(path)
    try:
        is_directory = checkpoint.is_dir()
        is_file = checkpoint.is_file()
    except (OSError, ValueError) as error:
        raise CheckpointFormatError(
            f"cannot inspect checkpoint path {checkpoint}: {error}"
        ) from error

    if is_directory:
        try:
            indexes = sorted(checkpoint.glob("*.safetensors.index.json"))
        except (OSError, ValueError) as error:
            raise CheckpointFormatError(
                f"cannot inspect checkpoint directory {checkpoint}: {error}"
            ) from error
        if len(indexes) > 1:
            raise CheckpointFormatError(
                f"{checkpoint}: multiple safetensors indexes found; pass one explicitly"
            )
        if indexes:
            index, _ = _safe_relative_file(
                checkpoint,
                indexes[0].name,
                str(checkpoint),
                "index",
            )
            return _load_index(index)
        try:
            shard_candidates = sorted(checkpoint.glob("*.safetensors"))
        except (OSError, ValueError) as error:
            raise CheckpointFormatError(
                f"cannot inspect checkpoint directory {checkpoint}: {error}"
            ) from error
        if not shard_candidates:
            raise CheckpointFormatError(f"{checkpoint}: no safetensors files found")
        if len(shard_candidates) > MAX_SHARDS:
            raise CheckpointFormatError(
                f"{checkpoint}: directory contains more than {MAX_SHARDS} shards"
            )
        shard_by_label: dict[str, Path] = {}
        for candidate in shard_candidates:
            shard, label = _safe_relative_file(
                checkpoint,
                candidate.name,
                str(checkpoint),
                "shard",
            )
            if label in shard_by_label:
                raise CheckpointFormatError(
                    f"{checkpoint}: multiple shard paths resolve to {label!r}"
                )
            shard_by_label[label] = shard
        tensors: list[TensorInfo] = []
        seen: dict[str, str] = {}
        for label in sorted(shard_by_label):
            for tensor in _read_safetensors(shard_by_label[label], label):
                if tensor.name in seen:
                    raise CheckpointFormatError(
                        f"tensor {tensor.name!r} appears in both "
                        f"{seen[tensor.name]!r} and {label!r}"
                    )
                seen[tensor.name] = label
                tensors.append(tensor)
                if len(tensors) > MAX_TENSORS:
                    raise CheckpointFormatError(
                        f"{checkpoint}: checkpoint contains more than "
                        f"{MAX_TENSORS} tensors"
                    )
        return sorted(tensors, key=lambda tensor: (tensor.name, tensor.file))
    if not is_file:
        raise CheckpointFormatError(f"checkpoint path does not exist: {checkpoint}")
    if checkpoint.name.endswith(".safetensors.index.json"):
        return _load_index(checkpoint)
    if checkpoint.suffix == ".safetensors":
        return _read_safetensors(checkpoint, checkpoint.name)
    raise CheckpointFormatError(
        f"unsupported checkpoint path {checkpoint}; expected safetensors or index JSON"
    )


def _targets_for_ownership(
    ownership: OwnershipRule, stage_map: dict[str, StageSpec]
) -> list[_RankTarget]:
    stage = stage_map[ownership.stage]
    allowed = set(ownership.ranks) if ownership.ranks is not None else None
    return [
        _RankTarget(stage, rank)
        for rank in stage.ranks
        if allowed is None or rank.rank in allowed
    ]


def _unknown_entries(
    tensor: TensorInfo,
    reason: str,
    targets: Sequence[_RankTarget],
) -> list[dict[str, Any]]:
    if not targets:
        return [
            _entry(
                tensor=tensor,
                classification="unknown",
                byte_count=tensor.nbytes,
                reason=reason,
            )
        ]
    return [
        _entry(
            tensor=tensor,
            classification="unknown",
            byte_count=tensor.nbytes,
            target=target,
            reason=reason,
        )
        for target in sorted(targets, key=lambda item: item.rank.rank)
    ]


def _entry(
    tensor: TensorInfo,
    classification: str,
    byte_count: int,
    target: _RankTarget | None = None,
    slices: Sequence[dict[str, int]] = (),
    reason: str | None = None,
) -> dict[str, Any]:
    rank = target.rank if target is not None else None
    return {
        "tensor": tensor.name,
        "file": tensor.file,
        "dtype": tensor.dtype,
        "shape": list(tensor.shape),
        "stage": target.stage.id if target is not None else None,
        "rank": rank.rank if rank is not None else None,
        "tp_rank": rank.tp_rank if rank is not None else None,
        "ep_rank": rank.ep_rank if rank is not None else None,
        "classification": classification,
        "bytes": byte_count,
        "slices": list(slices),
        "reason": reason,
    }


def _owner_targets(
    route: RouteRule,
    match: re.Match[str],
    targets: list[_RankTarget],
) -> tuple[list[_RankTarget] | None, str | None]:
    if route.owner is None:
        return targets, None
    owner = route.owner
    try:
        owner_id = int(match.group(owner.capture))
    except (IndexError, TypeError, ValueError):
        return None, f"capture {owner.capture!r} is not an integer"
    if owner_id < 0 or owner_id >= owner.count:
        return None, f"captured owner id {owner_id} is outside [0, {owner.count})"

    stage = targets[0].stage
    parallel_size = stage.parallel_size(owner.by)
    if owner.count % parallel_size:
        return (
            None,
            f"owner count {owner.count} is not divisible by {owner.by} size "
            f"{parallel_size}",
        )
    local_count = owner.count // parallel_size
    owner_coordinate = owner_id // local_count
    field = PARALLEL_FIELDS[owner.by]
    selected = [
        target for target in targets if getattr(target.rank, field) == owner_coordinate
    ]
    if not selected:
        return None, f"ownership contains no {owner.by} rank {owner_coordinate}"
    return selected, None


def _partition_entries(
    tensor: TensorInfo,
    route: RouteRule,
    targets: list[_RankTarget],
) -> tuple[list[dict[str, Any]] | None, str | None]:
    stage = targets[0].stage
    partition_details: list[tuple[PartitionSpec, int, int, int]] = []
    for partition in route.partitions:
        if partition.axis >= len(tensor.shape):
            return (
                None,
                f"partition axis {partition.axis} is outside shape rank "
                f"{len(tensor.shape)}",
            )
        parallel_size = stage.parallel_size(partition.by)
        parts = partition.parts or parallel_size
        if parts > parallel_size or parallel_size % parts:
            return (
                None,
                f"{partition.by} size {parallel_size} is not divisible by "
                f"partition parts {parts}",
            )
        source_axis_size = tensor.shape[partition.axis]
        if partition.start >= source_axis_size:
            return (
                None,
                f"partition start {partition.start} is outside axis "
                f"{partition.axis} size {source_axis_size}",
            )
        axis_size = (
            partition.length
            if partition.length is not None
            else source_axis_size - partition.start
        )
        if partition.start + axis_size > source_axis_size:
            return (
                None,
                f"partition range [{partition.start}, "
                f"{partition.start + axis_size}) exceeds axis "
                f"{partition.axis} size {source_axis_size}",
            )
        if axis_size % parts:
            return (
                None,
                f"partition axis {partition.axis} length {axis_size} is not divisible "
                f"by partition parts {parts}",
            )
        partition_details.append(
            (partition, parts, partition.start, axis_size // parts)
        )

    rows: list[tuple[_RankTarget, list[dict[str, int]], tuple[int, ...]]] = []
    observed_regions: set[tuple[int, ...]] = set()
    for target in targets:
        slices: list[dict[str, int]] = []
        region: list[int] = []
        for partition, parts, source_start, shard_size in partition_details:
            field = PARALLEL_FIELDS[partition.by]
            coordinate = getattr(target.rank, field)
            parallel_size = stage.parallel_size(partition.by)
            part_index = coordinate * parts // parallel_size
            start = source_start + part_index * shard_size
            slices.append(
                {"axis": partition.axis, "start": start, "stop": start + shard_size}
            )
            region.append(part_index)
        region_tuple = tuple(region)
        observed_regions.add(region_tuple)
        rows.append((target, slices, region_tuple))

    expected_regions = set(
        itertools.product(*(range(parts) for _, parts, _, _ in partition_details))
    )
    if observed_regions != expected_regions:
        return None, "owned ranks do not cover every declared tensor partition"

    entries: list[dict[str, Any]] = []
    for target, slices, _ in rows:
        selected_shape = list(tensor.shape)
        for item in slices:
            selected_shape[item["axis"]] = item["stop"] - item["start"]
        assert tensor.dtype_bits is not None
        if 0 in selected_shape:
            selected_elements = 0
        else:
            maximum_elements = tensor.nbytes * 8 // tensor.dtype_bits
            selected_elements = 1
            for size in selected_shape:
                if selected_elements > maximum_elements // size:
                    return None, "selected tensor partition exceeds source storage"
                selected_elements *= size
        selected_bits = selected_elements * tensor.dtype_bits
        if selected_bits % 8:
            return None, "selected tensor partition is not byte aligned"
        entries.append(
            _entry(
                tensor=tensor,
                classification="sharded",
                byte_count=selected_bits // 8,
                target=target,
                slices=slices,
            )
        )
    return entries, None


def _ledger_entries(tensor: TensorInfo, plan: LedgerPlan) -> list[dict[str, Any]]:
    stage_map = {stage.id: stage for stage in plan.stages}
    ownership_matches = [
        ownership
        for ownership in plan.ownership
        if ownership.pattern.search(tensor.name)
    ]
    ownership_targets = [
        target
        for ownership in ownership_matches
        for target in _targets_for_ownership(ownership, stage_map)
    ]
    unique_targets = {
        (target.stage.id, target.rank.rank): target for target in ownership_targets
    }
    candidate_targets = list(unique_targets.values())
    if not ownership_matches:
        return _unknown_entries(tensor, "no ownership matched tensor", ())
    if len(ownership_matches) != 1:
        return _unknown_entries(
            tensor,
            f"{len(ownership_matches)} ownership rules matched tensor",
            candidate_targets,
        )

    targets = _targets_for_ownership(ownership_matches[0], stage_map)
    route_matches = [
        (route, match)
        for route in plan.routes
        if (match := route.pattern.search(tensor.name)) is not None
    ]
    if not route_matches:
        return _unknown_entries(tensor, "no route matched tensor", targets)
    if len(route_matches) != 1:
        return _unknown_entries(
            tensor, f"{len(route_matches)} route rules matched tensor", targets
        )
    route, match = route_matches[0]

    if tensor.proof_error is not None:
        return _unknown_entries(tensor, tensor.proof_error, targets)
    if route.kind == "unknown":
        return _unknown_entries(
            tensor, route.reason or "route explicitly marked unknown", targets
        )
    if route.kind == "exact":
        if len(targets) != 1:
            return _unknown_entries(
                tensor,
                f"exact route requires one owned rank, found {len(targets)}",
                targets,
            )
        return [_entry(tensor, "exact", tensor.nbytes, target=targets[0])]
    if route.kind == "replicated":
        classification = "exact" if len(targets) == 1 else "replicated"
        return [
            _entry(tensor, classification, tensor.nbytes, target=target)
            for target in targets
        ]

    selected_targets, owner_error = _owner_targets(route, match, targets)
    if owner_error is not None or selected_targets is None:
        return _unknown_entries(tensor, owner_error or "invalid owner", targets)
    entries, partition_error = _partition_entries(tensor, route, selected_targets)
    if partition_error is not None or entries is None:
        return _unknown_entries(tensor, partition_error or "invalid partition", targets)
    return entries


def _entry_sort_key(entry: dict[str, Any]) -> tuple[Any, ...]:
    rank = entry["rank"]
    return (
        entry["tensor"],
        entry["file"],
        entry["stage"] is None,
        entry["stage"] or "",
        rank is None,
        rank if rank is not None else 0,
    )


def _empty_summary() -> dict[str, int]:
    return {f"{classification}_bytes": 0 for classification in CLASSIFICATIONS}


def build_ledger(tensors: Sequence[TensorInfo], plan: LedgerPlan) -> dict[str, Any]:
    """Route checkpoint tensors through a validated stage and rank plan.

    Args:
        tensors: Tensor metadata returned by :func:`load_checkpoint`.
        plan: Validated module ownership and TP/EP routing plan.

    Returns:
        A deterministic JSON-compatible ledger with tensor entries, stage and
        rank summaries, and source/rank totals.
    """
    ordered_tensors = sorted(tensors, key=lambda tensor: (tensor.name, tensor.file))
    entries = sorted(
        [
            entry
            for tensor in ordered_tensors
            for entry in _ledger_entries(tensor, plan)
        ],
        key=_entry_sort_key,
    )

    rank_summary_by_rank: dict[int, dict[str, Any]] = {}
    stage_summary_by_id: dict[str, dict[str, Any]] = {}
    for stage in plan.stages:
        stage_summary_by_id[stage.id] = {"stage": stage.id, **_empty_summary()}
        for rank in stage.ranks:
            rank_summary_by_rank[rank.rank] = {
                "stage": stage.id,
                "rank": rank.rank,
                "tp_rank": rank.tp_rank,
                "ep_rank": rank.ep_rank,
                **_empty_summary(),
            }

    classification_totals = _empty_summary()
    for entry in entries:
        field = f"{entry['classification']}_bytes"
        classification_totals[field] += entry["bytes"]
        if entry["stage"] is not None:
            stage_summary_by_id[entry["stage"]][field] += entry["bytes"]
        if entry["rank"] is not None:
            rank_summary_by_rank[entry["rank"]][field] += entry["bytes"]

    rank_summaries = [
        rank_summary_by_rank[rank] for rank in sorted(rank_summary_by_rank)
    ]
    stage_summaries = [
        stage_summary_by_id[stage_id] for stage_id in sorted(stage_summary_by_id)
    ]
    for summary in [*rank_summaries, *stage_summaries]:
        summary["total_bytes"] = sum(
            summary[f"{classification}_bytes"] for classification in CLASSIFICATIONS
        )
    unknown_tensors = {
        entry["tensor"] for entry in entries if entry["classification"] == "unknown"
    }
    tensor_by_name = {tensor.name: tensor for tensor in ordered_tensors}
    totals = {
        "checkpoint_source_bytes": sum(tensor.nbytes for tensor in ordered_tensors),
        "tensor_count": len(ordered_tensors),
        "entry_count": len(entries),
        **classification_totals,
        "known_rank_bytes": sum(
            classification_totals[f"{classification}_bytes"]
            for classification in ("exact", "replicated", "sharded")
        ),
        "rank_attributed_bytes": sum(
            entry["bytes"] for entry in entries if entry["rank"] is not None
        ),
        "unknown_source_bytes": sum(
            tensor_by_name[name].nbytes for name in unknown_tensors
        ),
        "unassigned_unknown_bytes": sum(
            entry["bytes"]
            for entry in entries
            if entry["classification"] == "unknown" and entry["rank"] is None
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "entries": entries,
        "stage_summaries": stage_summaries,
        "rank_summaries": rank_summaries,
        "totals": totals,
    }


def _json_line(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def render_json(ledger: dict[str, Any]) -> str:
    """Serialize a ledger as deterministic compact JSON with a final newline."""
    return _json_line(ledger) + "\n"


def render_jsonl(ledger: dict[str, Any]) -> str:
    """Serialize tensor and summary records as deterministic JSON Lines."""
    records: list[dict[str, Any]] = []
    records.extend({"record_type": "tensor", **entry} for entry in ledger["entries"])
    records.extend(
        {"record_type": "stage_summary", **summary}
        for summary in ledger["stage_summaries"]
    )
    records.extend(
        {"record_type": "rank_summary", **summary}
        for summary in ledger["rank_summaries"]
    )
    records.append({"record_type": "total_summary", **ledger["totals"]})
    return "".join(_json_line(record) + "\n" for record in records)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an offline TP/EP byte ledger from safetensors headers."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Safetensors file, index JSON, or checkpoint directory.",
    )
    parser.add_argument("--plan", required=True, help="Validated ledger plan JSON.")
    parser.add_argument(
        "--format", choices=("json", "jsonl"), default="json", dest="output_format"
    )
    parser.add_argument(
        "--output", help="Output path. Omit or use '-' to write to stdout."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the checkpoint ledger CLI.

    Args:
        argv: Optional argument sequence. Defaults to ``sys.argv[1:]``.

    Returns:
        Zero after writing deterministic JSON or JSONL output.
    """
    parser = _argument_parser()
    arguments = parser.parse_args(argv)
    try:
        plan = load_plan(arguments.plan)
        tensors = load_checkpoint(arguments.checkpoint)
        ledger = build_ledger(tensors, plan)
    except LedgerError as error:
        parser.error(str(error))

    output = render_json(ledger)
    if arguments.output_format == "jsonl":
        output = render_jsonl(ledger)
    if arguments.output in (None, "-"):
        try:
            sys.stdout.write(output)
            sys.stdout.flush()
        except OSError as error:
            parser.error(f"cannot write stdout: {error}")
    else:
        try:
            Path(arguments.output).write_text(output, encoding="utf-8")
        except (OSError, ValueError) as error:
            parser.error(f"cannot write output {arguments.output}: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
