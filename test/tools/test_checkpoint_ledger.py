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
import json
import struct
from pathlib import Path

import pytest

from tokenspeed.tools.checkpoint_ledger import (
    MAX_INDEX_BYTES,
    MAX_JSON_DEPTH,
    MAX_PATTERN_LENGTH,
    MAX_TENSOR_DIMENSION_SIZE,
    MAX_TENSOR_DIMENSIONS,
    MAX_TENSOR_NAME_LENGTH,
    CheckpointFormatError,
    PlanValidationError,
    build_ledger,
    load_checkpoint,
    main,
    parse_plan,
    render_json,
    render_jsonl,
)


def _write_safetensors(
    path: Path,
    tensors: dict[str, tuple[str, list[int], bytes]],
) -> None:
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    payload = bytearray()
    for name in sorted(tensors):
        dtype, shape, data = tensors[name]
        start = len(payload)
        payload.extend(data)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, len(payload)],
        }

    encoded = json.dumps(
        header, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def _write_index(
    path: Path,
    weight_map: dict[str, str],
    total_size: int,
) -> None:
    path.write_text(
        json.dumps(
            {"metadata": {"total_size": total_size}, "weight_map": weight_map},
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _rank_grid() -> list[dict[str, int]]:
    return [
        {"rank": 0, "tp_rank": 0, "ep_rank": 0},
        {"rank": 1, "tp_rank": 1, "ep_rank": 0},
        {"rank": 2, "tp_rank": 0, "ep_rank": 1},
        {"rank": 3, "tp_rank": 1, "ep_rank": 1},
    ]


def _plan_data() -> dict[str, object]:
    return {
        "version": 1,
        "stages": [{"id": "decoder", "ranks": _rank_grid()}],
        "ownership": [
            {
                "pattern": r"^model\.embed_tokens\.weight$",
                "stage": "decoder",
                "ranks": [0],
            },
            {"pattern": r"^model\.layers\.", "stage": "decoder"},
        ],
        "routes": [
            {"pattern": r"embed_tokens\.weight$", "kind": "exact"},
            {
                "pattern": r"q_proj\.weight$",
                "kind": "sharded",
                "partitions": [{"axis": 0, "by": "tp"}],
            },
            {"pattern": r"norm\.weight$", "kind": "replicated"},
            {
                "pattern": r"experts\.(?P<expert>\d+)\.up_proj\.weight$",
                "kind": "sharded",
                "owner": {"by": "ep", "capture": "expert", "count": 2},
                "partitions": [{"axis": 0, "by": "tp"}],
            },
        ],
    }


def _build_checkpoint(tmp_path: Path) -> Path:
    shard_a = tmp_path / "model-00001-of-00002.safetensors"
    shard_b = tmp_path / "model-00002-of-00002.safetensors"
    tensors_a = {
        "model.embed_tokens.weight": ("F32", [1], bytes(4)),
        "model.layers.0.mystery": ("U8", [8], bytes(8)),
        "model.layers.0.norm.weight": ("F32", [4], bytes(16)),
        "model.layers.0.q_proj.weight": ("F32", [8, 4], bytes(128)),
    }
    tensors_b = {
        "model.layers.0.experts.0.up_proj.weight": (
            "F32",
            [8, 4],
            bytes(128),
        ),
        "model.layers.0.experts.1.up_proj.weight": (
            "F32",
            [8, 4],
            bytes(128),
        ),
    }
    _write_safetensors(shard_a, tensors_a)
    _write_safetensors(shard_b, tensors_b)

    weight_map = {name: shard_a.name for name in tensors_a} | {
        name: shard_b.name for name in tensors_b
    }
    index = tmp_path / "model.safetensors.index.json"
    _write_index(
        index,
        weight_map,
        total_size=sum(
            len(data)
            for tensors in (tensors_a, tensors_b)
            for _, _, data in tensors.values()
        ),
    )
    return index


def test_index_to_stage_and_rank_ledger(tmp_path: Path) -> None:
    checkpoint = _build_checkpoint(tmp_path)
    tensors = load_checkpoint(checkpoint)
    ledger = build_ledger(tensors, parse_plan(_plan_data()))

    assert [tensor.name for tensor in tensors] == sorted(
        tensor.name for tensor in tensors
    )
    assert ledger["totals"] == {
        "checkpoint_source_bytes": 412,
        "entry_count": 17,
        "exact_bytes": 4,
        "known_rank_bytes": 580,
        "rank_attributed_bytes": 612,
        "replicated_bytes": 64,
        "sharded_bytes": 512,
        "tensor_count": 6,
        "unassigned_unknown_bytes": 0,
        "unknown_bytes": 32,
        "unknown_source_bytes": 8,
    }

    summaries = {item["rank"]: item for item in ledger["rank_summaries"]}
    assert summaries[0] == {
        "stage": "decoder",
        "rank": 0,
        "tp_rank": 0,
        "ep_rank": 0,
        "exact_bytes": 4,
        "replicated_bytes": 16,
        "sharded_bytes": 128,
        "unknown_bytes": 8,
        "total_bytes": 156,
    }
    assert summaries[1]["total_bytes"] == 152
    assert summaries[2]["total_bytes"] == 152
    assert summaries[3]["total_bytes"] == 152

    q_entries = [
        entry
        for entry in ledger["entries"]
        if entry["tensor"] == "model.layers.0.q_proj.weight"
    ]
    assert [entry["bytes"] for entry in q_entries] == [64, 64, 64, 64]
    assert q_entries[0]["slices"] == [{"axis": 0, "start": 0, "stop": 4}]
    assert q_entries[1]["slices"] == [{"axis": 0, "start": 4, "stop": 8}]
    assert q_entries[2]["slices"] == [{"axis": 0, "start": 0, "stop": 4}]

    expert_0_ranks = [
        entry["rank"]
        for entry in ledger["entries"]
        if entry["tensor"] == "model.layers.0.experts.0.up_proj.weight"
    ]
    expert_1_ranks = [
        entry["rank"]
        for entry in ledger["entries"]
        if entry["tensor"] == "model.layers.0.experts.1.up_proj.weight"
    ]
    assert expert_0_ranks == [0, 1]
    assert expert_1_ranks == [2, 3]
    expert_entries = [
        entry for entry in ledger["entries"] if ".experts." in entry["tensor"]
    ]
    assert [entry["bytes"] for entry in expert_entries] == [64, 64, 64, 64]
    assert [entry["slices"] for entry in expert_entries] == [
        [{"axis": 0, "start": 0, "stop": 4}],
        [{"axis": 0, "start": 4, "stop": 8}],
        [{"axis": 0, "start": 0, "stop": 4}],
        [{"axis": 0, "start": 4, "stop": 8}],
    ]

    unknown = [
        entry
        for entry in ledger["entries"]
        if entry["tensor"] == "model.layers.0.mystery"
    ]
    assert [entry["rank"] for entry in unknown] == [0, 1, 2, 3]
    assert {entry["classification"] for entry in unknown} == {"unknown"}
    assert {entry["reason"] for entry in unknown} == {"no route matched tensor"}


def test_missing_ownership_is_unassigned_unknown(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.safetensors"
    _write_safetensors(checkpoint, {"orphan": ("BF16", [4], bytes(8))})
    plan_data = {
        "version": 1,
        "stages": [
            {
                "id": "only",
                "ranks": [{"rank": 7, "tp_rank": 0, "ep_rank": 0}],
            }
        ],
        "ownership": [],
        "routes": [],
    }

    ledger = build_ledger(load_checkpoint(checkpoint), parse_plan(plan_data))

    assert ledger["entries"] == [
        {
            "bytes": 8,
            "classification": "unknown",
            "dtype": "BF16",
            "ep_rank": None,
            "file": "model.safetensors",
            "rank": None,
            "reason": "no ownership matched tensor",
            "shape": [4],
            "slices": [],
            "stage": None,
            "tensor": "orphan",
            "tp_rank": None,
        }
    ]
    assert ledger["rank_summaries"][0]["total_bytes"] == 0
    assert ledger["totals"]["unknown_source_bytes"] == 8
    assert ledger["totals"]["unassigned_unknown_bytes"] == 8


def test_unprovable_storage_and_partition_default_to_unknown(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.safetensors"
    _write_safetensors(
        checkpoint,
        {
            "bad_storage": ("F32", [4], bytes(8)),
            "odd_axis": ("F32", [5], bytes(20)),
        },
    )
    plan_data = {
        "version": 1,
        "stages": [
            {
                "id": "stage",
                "ranks": [
                    {"rank": 0, "tp_rank": 0, "ep_rank": 0},
                    {"rank": 1, "tp_rank": 1, "ep_rank": 0},
                ],
            }
        ],
        "ownership": [{"pattern": ".*", "stage": "stage"}],
        "routes": [
            {
                "pattern": ".*",
                "kind": "sharded",
                "partitions": [{"axis": 0, "by": "tp"}],
            }
        ],
    }

    ledger = build_ledger(load_checkpoint(checkpoint), parse_plan(plan_data))
    by_tensor = {
        name: [entry for entry in ledger["entries"] if entry["tensor"] == name]
        for name in ("bad_storage", "odd_axis")
    }

    assert {entry["classification"] for entry in by_tensor["bad_storage"]} == {
        "unknown"
    }
    assert "payload bytes" in by_tensor["bad_storage"][0]["reason"]
    assert {entry["classification"] for entry in by_tensor["odd_axis"]} == {"unknown"}
    assert "not divisible" in by_tensor["odd_axis"][0]["reason"]


def test_fused_tensor_can_partition_across_ep_and_tp(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.safetensors"
    _write_safetensors(
        checkpoint,
        {"fused_experts": ("F32", [4, 8, 2], bytes(256))},
    )
    plan_data = {
        "version": 1,
        "stages": [{"id": "moe", "ranks": _rank_grid()}],
        "ownership": [{"pattern": "^fused_experts$", "stage": "moe"}],
        "routes": [
            {
                "pattern": "^fused_experts$",
                "kind": "sharded",
                "partitions": [
                    {"axis": 0, "by": "ep"},
                    {"axis": 1, "by": "tp"},
                ],
            }
        ],
    }

    ledger = build_ledger(load_checkpoint(checkpoint), parse_plan(plan_data))

    assert [entry["bytes"] for entry in ledger["entries"]] == [64, 64, 64, 64]
    assert ledger["entries"][0]["slices"] == [
        {"axis": 0, "start": 0, "stop": 2},
        {"axis": 1, "start": 0, "stop": 4},
    ]
    assert ledger["entries"][3]["slices"] == [
        {"axis": 0, "start": 2, "stop": 4},
        {"axis": 1, "start": 4, "stop": 8},
    ]


def test_partition_parts_can_replicate_qkv_slices(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.safetensors"
    _write_safetensors(checkpoint, {"kv_proj": ("F32", [4], bytes(16))})
    ranks = [{"rank": rank, "tp_rank": rank, "ep_rank": 0} for rank in range(4)]
    plan_data = {
        "version": 1,
        "stages": [{"id": "attention", "ranks": ranks}],
        "ownership": [{"pattern": "^kv_proj$", "stage": "attention"}],
        "routes": [
            {
                "pattern": "^kv_proj$",
                "kind": "sharded",
                "partitions": [{"axis": 0, "by": "tp", "parts": 2}],
            }
        ],
    }

    ledger = build_ledger(load_checkpoint(checkpoint), parse_plan(plan_data))

    assert [entry["bytes"] for entry in ledger["entries"]] == [8, 8, 8, 8]
    assert [entry["slices"] for entry in ledger["entries"]] == [
        [{"axis": 0, "start": 0, "stop": 2}],
        [{"axis": 0, "start": 0, "stop": 2}],
        [{"axis": 0, "start": 2, "stop": 4}],
        [{"axis": 0, "start": 2, "stop": 4}],
    ]


def test_partition_can_exclude_checkpoint_padding(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.safetensors"
    _write_safetensors(checkpoint, {"padded": ("F32", [16], bytes(64))})
    plan_data = {
        "version": 1,
        "stages": [
            {
                "id": "attention",
                "ranks": [
                    {"rank": 0, "tp_rank": 0, "ep_rank": 0},
                    {"rank": 1, "tp_rank": 1, "ep_rank": 0},
                ],
            }
        ],
        "ownership": [{"pattern": "^padded$", "stage": "attention"}],
        "routes": [
            {
                "pattern": "^padded$",
                "kind": "sharded",
                "partitions": [{"axis": 0, "by": "tp", "start": 2, "length": 12}],
            }
        ],
    }

    ledger = build_ledger(load_checkpoint(checkpoint), parse_plan(plan_data))

    assert [entry["bytes"] for entry in ledger["entries"]] == [24, 24]
    assert [entry["slices"] for entry in ledger["entries"]] == [
        [{"axis": 0, "start": 2, "stop": 8}],
        [{"axis": 0, "start": 8, "stop": 14}],
    ]


def test_plan_validation_is_strict() -> None:
    invalid_plans: list[tuple[dict[str, object], str]] = []

    with_extra_key = _plan_data()
    with_extra_key["extra"] = True
    invalid_plans.append((with_extra_key, "unexpected keys"))

    duplicate_rank = copy.deepcopy(_plan_data())
    duplicate_rank["stages"][0]["ranks"][1]["rank"] = 0
    invalid_plans.append((duplicate_rank, "duplicate global rank"))

    incomplete_grid = copy.deepcopy(_plan_data())
    incomplete_grid["stages"][0]["ranks"].pop()
    invalid_plans.append((incomplete_grid, "complete TP x EP grid"))

    invalid_regex = copy.deepcopy(_plan_data())
    invalid_regex["routes"][0]["pattern"] = "["
    invalid_plans.append((invalid_regex, "invalid regex"))

    missing_capture = copy.deepcopy(_plan_data())
    missing_capture["routes"][3]["owner"]["capture"] = "missing"
    invalid_plans.append((missing_capture, "named capture"))

    empty_shard = copy.deepcopy(_plan_data())
    empty_shard["routes"][1].pop("partitions")
    invalid_plans.append((empty_shard, "requires owner or partitions"))

    sparse_coordinate = copy.deepcopy(_plan_data())
    sparse_coordinate["stages"][0]["ranks"][3]["ep_rank"] = 10**100
    invalid_plans.append((sparse_coordinate, "complete TP x EP grid"))

    nested_quantifier = copy.deepcopy(_plan_data())
    nested_quantifier["routes"][0]["pattern"] = r"^(a+)+$"
    invalid_plans.append((nested_quantifier, "high-risk quantified group"))

    ambiguous_alternation = copy.deepcopy(_plan_data())
    ambiguous_alternation["routes"][0]["pattern"] = r"^(a|aa)+$"
    invalid_plans.append((ambiguous_alternation, "high-risk quantified group"))

    verbose_nested_quantifier = copy.deepcopy(_plan_data())
    verbose_nested_quantifier["routes"][0]["pattern"] = r"(?x)^(a+) +$"
    invalid_plans.append((verbose_nested_quantifier, "verbose regex mode"))

    long_pattern = copy.deepcopy(_plan_data())
    long_pattern["routes"][0]["pattern"] = "a" * (MAX_PATTERN_LENGTH + 1)
    invalid_plans.append((long_pattern, "maximum length"))

    for data, message in invalid_plans:
        with pytest.raises(PlanValidationError, match=message):
            parse_plan(data)

    safe_group_quantifier = copy.deepcopy(_plan_data())
    safe_group_quantifier["routes"][0]["pattern"] = r"^(?:ab)+$"
    parse_plan(safe_group_quantifier)


def test_checkpoint_index_and_header_validation(tmp_path: Path) -> None:
    shard = tmp_path / "model.safetensors"
    _write_safetensors(shard, {"present": ("F32", [1], bytes(4))})
    index = tmp_path / "model.safetensors.index.json"
    _write_index(index, {"missing": shard.name}, total_size=4)

    with pytest.raises(CheckpointFormatError, match="missing.*not present"):
        load_checkpoint(index)

    malformed = tmp_path / "malformed.safetensors"
    malformed.write_bytes(struct.pack("<Q", 1024) + b"{}")
    with pytest.raises(CheckpointFormatError, match="header length"):
        load_checkpoint(malformed)


def test_index_requires_exact_tensor_set_and_total_size(tmp_path: Path) -> None:
    shard = tmp_path / "model.safetensors"
    _write_safetensors(
        shard,
        {
            "extra": ("U8", [1], b"x"),
            "mapped": ("U8", [1], b"y"),
        },
    )
    index = tmp_path / "model.safetensors.index.json"

    _write_index(index, {"mapped": shard.name}, total_size=2)
    with pytest.raises(CheckpointFormatError, match="extra.*absent from weight_map"):
        load_checkpoint(index)

    weight_map = {"extra": shard.name, "mapped": shard.name}
    _write_index(index, weight_map, total_size=1)
    with pytest.raises(CheckpointFormatError, match="total_size is 1, expected 2"):
        load_checkpoint(index)

    _write_index(index, weight_map, total_size=2)
    assert [tensor.name for tensor in load_checkpoint(index)] == [
        "extra",
        "mapped",
    ]


def test_checkpoint_paths_stay_inside_directory(tmp_path: Path) -> None:
    outside = tmp_path / "outside.safetensors"
    _write_safetensors(outside, {"outside": ("U8", [1], b"x")})

    indexed = tmp_path / "indexed"
    indexed.mkdir()
    index = indexed / "model.safetensors.index.json"
    _write_index(index, {"outside": "../outside.safetensors"}, total_size=1)
    with pytest.raises(CheckpointFormatError, match="stay inside"):
        load_checkpoint(index)

    _write_index(index, {"outside": "bad\x00name.safetensors"}, total_size=1)
    with pytest.raises(CheckpointFormatError, match="path is invalid"):
        load_checkpoint(index)

    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "linked.safetensors").symlink_to(outside)
    with pytest.raises(CheckpointFormatError, match="stay inside"):
        load_checkpoint(plain)

    non_regular = tmp_path / "non-regular"
    non_regular.mkdir()
    (non_regular / "directory.safetensors").mkdir()
    with pytest.raises(CheckpointFormatError, match="not a regular file"):
        load_checkpoint(non_regular)


def test_json_and_safetensors_resource_limits(tmp_path: Path) -> None:
    assert MAX_INDEX_BYTES >= 64 * 1024 * 1024

    deep_index = tmp_path / "deep.safetensors.index.json"
    deep_index.write_text(
        "[" * (MAX_JSON_DEPTH + 1) + "0" + "]" * (MAX_JSON_DEPTH + 1),
        encoding="utf-8",
    )
    with pytest.raises(CheckpointFormatError, match="JSON nesting exceeds"):
        load_checkpoint(deep_index)

    huge_integer = tmp_path / "huge.safetensors.index.json"
    huge_integer.write_text("9" * 5000, encoding="utf-8")
    with pytest.raises(CheckpointFormatError, match="invalid JSON"):
        load_checkpoint(huge_integer)

    oversized_index = tmp_path / "oversized.safetensors.index.json"
    with oversized_index.open("wb") as file:
        file.truncate(MAX_INDEX_BYTES + 1)
    with pytest.raises(CheckpointFormatError, match="exceeds limit"):
        load_checkpoint(oversized_index)

    long_name = tmp_path / "long-name.safetensors"
    _write_safetensors(
        long_name,
        {"x" * (MAX_TENSOR_NAME_LENGTH + 1): ("U8", [0], b"")},
    )
    with pytest.raises(CheckpointFormatError, match="name exceeds maximum length"):
        load_checkpoint(long_name)

    too_many_dimensions = tmp_path / "many-dimensions.safetensors"
    _write_safetensors(
        too_many_dimensions,
        {"many": ("U8", [1] * (MAX_TENSOR_DIMENSIONS + 1), b"x")},
    )
    with pytest.raises(CheckpointFormatError, match="more than 64 dimensions"):
        load_checkpoint(too_many_dimensions)

    oversized_dimension = tmp_path / "oversized-dimension.safetensors"
    _write_safetensors(
        oversized_dimension,
        {"wide": ("U8", [MAX_TENSOR_DIMENSION_SIZE + 1, 0], b"")},
    )
    with pytest.raises(CheckpointFormatError, match="invalid shape"):
        load_checkpoint(oversized_dimension)

    bounded_product = tmp_path / "bounded-product.safetensors"
    _write_safetensors(
        bounded_product,
        {
            "wide": (
                "U8",
                [MAX_TENSOR_DIMENSION_SIZE] * MAX_TENSOR_DIMENSIONS,
                b"",
            )
        },
    )
    tensor = load_checkpoint(bounded_product)[0]
    assert tensor.proof_error == (
        "payload bytes 0 are smaller than dtype/shape requires"
    )


def test_duplicate_keys_and_payload_boundaries_are_rejected(
    tmp_path: Path,
) -> None:
    tensor_json = b'{"dtype":"U8","shape":[1],"data_offsets":[0,1]}'
    duplicate_header = b'{"same":' + tensor_json + b',"same":' + tensor_json + b"}"
    duplicate = tmp_path / "duplicate.safetensors"
    duplicate.write_bytes(
        struct.pack("<Q", len(duplicate_header)) + duplicate_header + b"x"
    )
    with pytest.raises(CheckpointFormatError, match="duplicate JSON key 'same'"):
        load_checkpoint(duplicate)

    null_metadata = tmp_path / "null-metadata.safetensors"
    null_metadata_header = json.dumps(
        {
            "__metadata__": None,
            "single": {
                "dtype": "U8",
                "shape": [1],
                "data_offsets": [0, 1],
            },
        }
    ).encode("utf-8")
    null_metadata.write_bytes(
        struct.pack("<Q", len(null_metadata_header)) + null_metadata_header + b"x"
    )
    with pytest.raises(CheckpointFormatError, match="metadata.*map strings"):
        load_checkpoint(null_metadata)

    index = tmp_path / "duplicate.safetensors.index.json"
    index.write_text(
        '{"weight_map":{"a":"one.safetensors"},'
        '"weight_map":{"a":"two.safetensors"}}',
        encoding="utf-8",
    )
    with pytest.raises(CheckpointFormatError, match="duplicate JSON key 'weight_map'"):
        load_checkpoint(index)

    overlap = tmp_path / "overlap.safetensors"
    overlap_header = {
        "a": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]},
        "b": {"dtype": "U8", "shape": [1], "data_offsets": [1, 2]},
    }
    encoded_overlap = json.dumps(overlap_header).encode("utf-8")
    overlap.write_bytes(
        struct.pack("<Q", len(encoded_overlap)) + encoded_overlap + b"xx"
    )
    with pytest.raises(CheckpointFormatError, match="overlaps prior tensor"):
        load_checkpoint(overlap)

    gap = tmp_path / "gap.safetensors"
    gap_header = {
        "a": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]},
        "b": {"dtype": "U8", "shape": [1], "data_offsets": [2, 3]},
    }
    encoded_gap = json.dumps(gap_header).encode("utf-8")
    gap.write_bytes(struct.pack("<Q", len(encoded_gap)) + encoded_gap + b"xxx")
    with pytest.raises(CheckpointFormatError, match="leaves a gap"):
        load_checkpoint(gap)

    trailing = tmp_path / "trailing.safetensors"
    trailing_header = {"a": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}
    encoded_trailing = json.dumps(trailing_header).encode("utf-8")
    trailing.write_bytes(
        struct.pack("<Q", len(encoded_trailing)) + encoded_trailing + b"xx"
    )
    with pytest.raises(CheckpointFormatError, match="cover 1 of 2 payload bytes"):
        load_checkpoint(trailing)


def test_index_total_size_type_is_strict(tmp_path: Path) -> None:
    shard = tmp_path / "model.safetensors"
    _write_safetensors(shard, {"single": ("U8", [1], b"x")})
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(
        json.dumps(
            {
                "metadata": {"total_size": True},
                "weight_map": {"single": shard.name},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(CheckpointFormatError, match="non-negative integer"):
        load_checkpoint(index)

    index.write_text(
        json.dumps(
            {
                "metadata": None,
                "weight_map": {"single": shard.name},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(CheckpointFormatError, match="metadata must be an object"):
        load_checkpoint(index)


def test_invalid_owner_is_unknown_fail_closed(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.safetensors"
    _write_safetensors(
        checkpoint,
        {"experts.2.weight": ("F32", [4], bytes(16))},
    )
    plan_data = {
        "version": 1,
        "stages": [{"id": "moe", "ranks": _rank_grid()}],
        "ownership": [{"pattern": ".*", "stage": "moe"}],
        "routes": [
            {
                "pattern": r"experts\.(?P<expert>\d+)\.weight$",
                "kind": "sharded",
                "owner": {"by": "ep", "capture": "expert", "count": 3},
            }
        ],
    }

    ledger = build_ledger(load_checkpoint(checkpoint), parse_plan(plan_data))

    assert {entry["classification"] for entry in ledger["entries"]} == {"unknown"}
    assert {entry["bytes"] for entry in ledger["entries"]} == {16}
    assert {entry["reason"] for entry in ledger["entries"]} == {
        "owner count 3 is not divisible by ep size 2"
    }


def test_json_and_jsonl_cli_are_deterministic(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoint = tmp_path / "model.safetensors"
    _write_safetensors(checkpoint, {"single": ("I32", [2], bytes(8))})
    plan_data = {
        "version": 1,
        "stages": [
            {
                "id": "single-stage",
                "ranks": [{"rank": 3, "tp_rank": 0, "ep_rank": 0}],
            }
        ],
        "ownership": [{"pattern": "^single$", "stage": "single-stage"}],
        "routes": [{"pattern": "^single$", "kind": "exact"}],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan_data), encoding="utf-8")

    arguments = [
        "--checkpoint",
        str(checkpoint),
        "--plan",
        str(plan_path),
        "--format",
        "json",
    ]
    assert main(arguments) == 0
    first_json = capsys.readouterr().out
    assert main(arguments) == 0
    second_json = capsys.readouterr().out
    assert first_json == second_json
    assert first_json == render_json(
        build_ledger(load_checkpoint(checkpoint), parse_plan(plan_data))
    )

    arguments[-1] = "jsonl"
    assert main(arguments) == 0
    cli_jsonl = capsys.readouterr().out
    ledger = build_ledger(load_checkpoint(checkpoint), parse_plan(plan_data))
    assert cli_jsonl == render_jsonl(ledger)
    records = [json.loads(line) for line in cli_jsonl.splitlines()]
    assert [record["record_type"] for record in records] == [
        "tensor",
        "stage_summary",
        "rank_summary",
        "total_summary",
    ]


def test_cli_normalizes_malformed_input_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint = tmp_path / "model.safetensors"
    _write_safetensors(checkpoint, {"single": ("U8", [1], b"x")})
    invalid_plan = tmp_path / "plan.json"
    invalid_plan.write_bytes(b"\xff")

    with pytest.raises(SystemExit) as exit_info:
        main(
            [
                "--checkpoint",
                str(checkpoint),
                "--plan",
                str(invalid_plan),
            ]
        )

    captured = capsys.readouterr()
    assert exit_info.value.code == 2
    assert captured.out == ""
    assert "plan is not UTF-8" in captured.err
    assert "Traceback" not in captured.err

    valid_plan = tmp_path / "valid-plan.json"
    valid_plan.write_text(
        json.dumps(
            {
                "version": 1,
                "stages": [
                    {
                        "id": "stage",
                        "ranks": [{"rank": 0, "tp_rank": 0, "ep_rank": 0}],
                    }
                ],
                "ownership": [{"pattern": ".*", "stage": "stage"}],
                "routes": [{"pattern": ".*", "kind": "exact"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as output_exit:
        main(
            [
                "--checkpoint",
                str(checkpoint),
                "--plan",
                str(valid_plan),
                "--output",
                "bad\x00output",
            ]
        )
    captured = capsys.readouterr()
    assert output_exit.value.code == 2
    assert "cannot write output" in captured.err
    assert "Traceback" not in captured.err
