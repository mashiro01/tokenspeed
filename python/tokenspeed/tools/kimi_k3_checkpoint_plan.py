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

"""Generate the Kimi-K3 PP/TP ownership plan for checkpoint byte ledgers."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tokenspeed.runtime.pipeline.adapters.kimi_k3 import (
    balanced_kimi_k3_stage_layer_counts,
)

MAX_CONFIG_BYTES = 4 * 1024 * 1024


class KimiK3PlanError(ValueError):
    """The model config cannot produce an exact Kimi-K3 ledger plan."""


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise KimiK3PlanError(f"{name} must be an object")
    return value


def _positive_int(config: Mapping[str, Any], key: str, owner: str) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise KimiK3PlanError(f"{owner}.{key} must be a positive integer")
    return value


def _validate_architecture(config: Mapping[str, Any]) -> None:
    architectures = config.get("architectures")
    if (
        not isinstance(architectures, list)
        or "KimiK3ForConditionalGeneration" not in architectures
    ):
        raise KimiK3PlanError(
            "config.architectures must contain KimiK3ForConditionalGeneration"
        )
    if config.get("model_type") != "kimi_k3":
        raise KimiK3PlanError("config.model_type must be 'kimi_k3'")


def _rank_stages(pipeline_size: int, tensor_parallel_size: int) -> list[dict[str, Any]]:
    width = max(1, len(str(pipeline_size - 1)))
    return [
        {
            "id": f"stage-{stage_index:0{width}d}",
            "ranks": [
                {
                    "rank": stage_index * tensor_parallel_size + tp_rank,
                    "tp_rank": tp_rank,
                    "ep_rank": 0,
                }
                for tp_rank in range(tensor_parallel_size)
            ],
        }
        for stage_index in range(pipeline_size)
    ]


def _ownership(
    stage_ids: list[str], stage_layer_counts: Sequence[int]
) -> list[dict[str, Any]]:
    ownership: list[dict[str, Any]] = [
        {"pattern": r"^vision_tower\.", "stage": stage_ids[0]},
        {"pattern": r"^mm_projector\.", "stage": stage_ids[0]},
        {
            "pattern": r"^language_model\.model\.embed_tokens\.",
            "stage": stage_ids[0],
        },
    ]
    first_layer = 0
    for stage_id, layer_count in zip(stage_ids, stage_layer_counts):
        for layer_id in range(first_layer, first_layer + layer_count):
            ownership.append(
                {
                    "pattern": rf"^language_model\.model\.layers\.{layer_id}\.",
                    "stage": stage_id,
                }
            )
        first_layer += layer_count
    ownership.extend(
        [
            {
                "pattern": r"^language_model\.model\.norm\.",
                "stage": stage_ids[-1],
            },
            {
                "pattern": r"^language_model\.model\.output_attn_res_norm\.",
                "stage": stage_ids[-1],
            },
            {
                "pattern": r"^language_model\.model\.output_attn_res_proj\.",
                "stage": stage_ids[-1],
            },
            {
                "pattern": r"^language_model\.lm_head\.",
                "stage": stage_ids[-1],
            },
        ]
    )
    return ownership


def _routes(num_attention_heads: int) -> list[dict[str, Any]]:
    shard_axis_0 = [{"axis": 0, "by": "tp"}]
    shard_axis_1 = [{"axis": 1, "by": "tp"}]
    return [
        {
            "pattern": r"^(?:vision_tower|mm_projector)\.",
            "kind": "replicated",
        },
        {
            "pattern": r"^language_model\.model\.embed_tokens\.weight$",
            "kind": "sharded",
            "partitions": shard_axis_0,
        },
        {
            "pattern": r"^language_model\.lm_head\.weight$",
            "kind": "sharded",
            "partitions": shard_axis_0,
        },
        {
            "pattern": (
                r"\.block_sparse_moe\.experts\.\d+\.w[13]\."
                r"(?:weight_packed|weight_scale)$"
            ),
            "kind": "sharded",
            "partitions": shard_axis_0,
        },
        {
            "pattern": (
                r"\.block_sparse_moe\.experts\.\d+\.w2\."
                r"(?:weight_packed|weight_scale)$"
            ),
            "kind": "sharded",
            "partitions": shard_axis_1,
        },
        {
            "pattern": (
                r"\.block_sparse_moe\.shared_experts\."
                r"(?:gate_proj|up_proj)\.weight$"
            ),
            "kind": "sharded",
            "partitions": shard_axis_0,
        },
        {
            "pattern": r"\.block_sparse_moe\.shared_experts\.down_proj\.weight$",
            "kind": "sharded",
            "partitions": shard_axis_1,
        },
        {
            "pattern": r"\.mlp\.(?:gate_proj|up_proj)\.weight$",
            "kind": "sharded",
            "partitions": shard_axis_0,
        },
        {
            "pattern": r"\.mlp\.down_proj\.weight$",
            "kind": "sharded",
            "partitions": shard_axis_1,
        },
        {
            "pattern": (
                r"\.self_attn\.(?:q_proj|k_proj|v_proj|g_proj|b_proj|"
                r"f_b_proj|q_b_proj|kv_b_proj)\.weight$"
            ),
            "kind": "sharded",
            "partitions": shard_axis_0,
        },
        {
            "pattern": r"\.self_attn\.o_proj\.weight$",
            "kind": "sharded",
            "partitions": shard_axis_1,
        },
        {
            "pattern": r"\.self_attn\.[qkv]_conv1d\.weight$",
            "kind": "sharded",
            "partitions": shard_axis_0,
        },
        {
            "pattern": r"\.self_attn\.dt_bias$",
            "kind": "sharded",
            "partitions": shard_axis_0,
        },
        {
            "pattern": r"\.self_attn\.A_log$",
            "kind": "sharded",
            "partitions": [
                {
                    "axis": 0,
                    "by": "tp",
                    "start": 0,
                    "length": num_attention_heads,
                }
            ],
        },
        {
            "pattern": (
                r"\.block_sparse_moe\.(?:gate\.(?:weight|"
                r"e_score_correction_bias)|routed_expert_"
                r"(?:down_proj|up_proj|norm)\.weight)$"
            ),
            "kind": "replicated",
        },
        {
            "pattern": (
                r"\.(?:input_layernorm|post_attention_layernorm|mlp_res_norm|"
                r"mlp_res_proj|self_attention_res_norm|"
                r"self_attention_res_proj)\.weight$"
            ),
            "kind": "replicated",
        },
        {
            "pattern": (
                r"\.self_attn\.(?:f_a_proj|q_a_proj|" r"kv_a_proj_with_mqa)\.weight$"
            ),
            "kind": "replicated",
        },
        {
            "pattern": (
                r"\.self_attn\.(?:o_norm|q_a_layernorm|" r"kv_a_layernorm)\.weight$"
            ),
            "kind": "replicated",
        },
        {
            "pattern": (
                r"^language_model\.model\.(?:norm|output_attn_res_norm|"
                r"output_attn_res_proj)\.weight$"
            ),
            "kind": "replicated",
        },
    ]


def build_kimi_k3_checkpoint_plan(
    config: Mapping[str, Any],
    *,
    pipeline_parallel_size: int,
    tensor_parallel_size: int,
    mm_encoder_tp_mode: str = "data",
) -> dict[str, Any]:
    """Build a checkpoint-ledger plan matching the native K3 runtime.

    Args:
        config: Decoded Kimi-K3 ``config.json`` object.
        pipeline_parallel_size: Number of contiguous text pipeline stages.
        tensor_parallel_size: Text attention and MoE tensor-parallel size.
        mm_encoder_tp_mode: Vision parallel mode. K3 PP qualification uses
            ``data`` because its 12 vision heads cannot be weight-sharded by TP8.

    Returns:
        A JSON-compatible plan accepted by ``checkpoint_ledger``.

    Raises:
        KimiK3PlanError: If the config or requested topology is incompatible.
    """
    config = _object(config, "config")
    _validate_architecture(config)
    if isinstance(pipeline_parallel_size, bool) or pipeline_parallel_size <= 0:
        raise KimiK3PlanError("pipeline_parallel_size must be a positive integer")
    if isinstance(tensor_parallel_size, bool) or tensor_parallel_size <= 0:
        raise KimiK3PlanError("tensor_parallel_size must be a positive integer")
    if mm_encoder_tp_mode != "data":
        raise KimiK3PlanError(
            "exact Kimi-K3 PP ledger generation currently requires "
            "mm_encoder_tp_mode='data'"
        )

    text_config = _object(config.get("text_config"), "config.text_config")
    _object(config.get("vision_config"), "config.vision_config")
    num_layers = _positive_int(text_config, "num_hidden_layers", "text_config")
    hidden_size = _positive_int(text_config, "hidden_size", "text_config")
    block_size = _positive_int(text_config, "attn_res_block_size", "text_config")
    num_attention_heads = _positive_int(
        text_config, "num_attention_heads", "text_config"
    )

    divisible_fields = (
        "vocab_size",
        "num_attention_heads",
        "intermediate_size",
        "moe_intermediate_size",
    )
    for field in divisible_fields:
        value = _positive_int(text_config, field, "text_config")
        if value % tensor_parallel_size:
            raise KimiK3PlanError(
                f"text_config.{field}={value} is not divisible by "
                f"tensor_parallel_size={tensor_parallel_size}"
            )
    if hidden_size <= 0:
        raise AssertionError("validated hidden size must be positive")

    try:
        stage_layer_counts = balanced_kimi_k3_stage_layer_counts(
            num_layers=num_layers,
            attn_res_block_size=block_size,
            stage_count=pipeline_parallel_size,
        )
    except ValueError as error:
        raise KimiK3PlanError(str(error)) from error

    stages = _rank_stages(pipeline_parallel_size, tensor_parallel_size)
    stage_ids = [stage["id"] for stage in stages]
    return {
        "version": 1,
        "stages": stages,
        "ownership": _ownership(stage_ids, stage_layer_counts),
        "routes": _routes(num_attention_heads),
    }


def load_config(path: str | Path) -> Mapping[str, Any]:
    """Load a bounded UTF-8 model config without importing Transformers."""
    config_path = Path(path)
    try:
        raw = config_path.read_bytes()
    except (OSError, ValueError) as error:
        raise KimiK3PlanError(f"cannot read config {config_path}: {error}") from error
    if len(raw) > MAX_CONFIG_BYTES:
        raise KimiK3PlanError(
            f"config size {len(raw)} exceeds limit {MAX_CONFIG_BYTES}"
        )
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as error:
        raise KimiK3PlanError(f"invalid config {config_path}: {error}") from error
    return _object(value, "config")


def render_plan(plan: Mapping[str, Any]) -> str:
    """Render a deterministic, reviewable ledger plan."""
    return json.dumps(plan, ensure_ascii=True, indent=2, sort_keys=True) + "\n"


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an exact Kimi-K3 PP/TP checkpoint ledger plan."
    )
    parser.add_argument("--config", required=True, help="Kimi-K3 config.json path.")
    parser.add_argument("--pipeline-parallel-size", type=int, default=8)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument(
        "--mm-encoder-tp-mode", choices=("data", "weights"), default="data"
    )
    parser.add_argument("--output", help="Output path. Omit or use '-' for stdout.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Kimi-K3 checkpoint-plan generator CLI."""
    parser = _argument_parser()
    arguments = parser.parse_args(argv)
    try:
        plan = build_kimi_k3_checkpoint_plan(
            load_config(arguments.config),
            pipeline_parallel_size=arguments.pipeline_parallel_size,
            tensor_parallel_size=arguments.tensor_parallel_size,
            mm_encoder_tp_mode=arguments.mm_encoder_tp_mode,
        )
    except KimiK3PlanError as error:
        parser.error(str(error))
    output = render_plan(plan)
    if arguments.output in (None, "-"):
        try:
            sys.stdout.write(output)
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
