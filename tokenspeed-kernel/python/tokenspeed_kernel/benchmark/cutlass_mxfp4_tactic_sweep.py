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

"""Sweep FlashInfer CUTLASS MXFP8-by-MXFP4 fused-MoE tactics.

FlashInfer 0.6.16 cannot autotune this operation safely on SM120: its profiling
preparation launches ``tactic=-1`` outside the per-candidate error boundary and
can poison the CUDA context.  Normal execution and explicit ``profile_ids`` are
safe, so this tool measures every shape-valid GEMM1 and GEMM2 tactic through the
real fused operation without entering FlashInfer's autotune context.

The output is an environment- and shape-pinned JSON artifact.  It contains the
native heuristic time, every candidate measurement, the independently selected
GEMM tactics, and the combined time for every runtime token bucket.  Failed or
numerically invalid candidates are retained in the artifact instead of being
silently discarded.

Example for DeepSeek-V4-Flash TP4::

    python -m tokenspeed_kernel.benchmark.cutlass_mxfp4_tactic_sweep \
        --output deepseek-v4-flash-sm120-tp4.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform as host_platform
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch

DEFAULT_BUCKETS = (
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    768,
    1024,
    1280,
    1536,
    1792,
    2048,
    2560,
    3072,
    3584,
    4096,
    8192,
)


def _parse_buckets(raw: str) -> tuple[int, ...]:
    buckets = tuple(sorted({int(value.strip()) for value in raw.split(",")}))
    if not buckets or buckets[0] <= 0:
        raise argparse.ArgumentTypeError("buckets must be positive integers")
    return buckets


def _best_valid(measurements: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [item for item in measurements if item["status"] == "ok"]
    if not valid:
        raise RuntimeError("no valid tactic remained after the sweep")
    return min(valid, key=lambda item: item["latency_us"])


def _cutlass_tactic_abi_version() -> str:
    try:
        from flashinfer.autotuner.autotuner import _nvfp4_cutlass_version
    except ImportError as exc:
        raise RuntimeError(
            "FlashInfer does not expose a CUTLASS tactic ABI version; explicit "
            "profile IDs cannot be persisted safely"
        ) from exc
    return str(_nvfp4_cutlass_version)


def _select_robust_bucket(
    routing_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    candidate_maps = {
        routing: {
            tuple(item["profile_ids"]): item
            for item in result["pair_candidates"]
            if item["status"] == "ok"
        }
        for routing, result in routing_results.items()
    }
    common_pairs = set.intersection(
        *(set(candidates) for candidates in candidate_maps.values())
    )
    if not common_pairs:
        raise RuntimeError("routing profiles have no common valid tactic pair")
    optimum = {
        routing: min(item["latency_us"] for item in candidates.values())
        for routing, candidates in candidate_maps.items()
    }

    def regret(pair: tuple[int, int]) -> float:
        return max(
            candidate_maps[routing][pair]["latency_us"] / optimum[routing]
            for routing in candidate_maps
        )

    selected = min(common_pairs, key=lambda pair: (regret(pair), pair))
    profiles = {}
    for routing, result in routing_results.items():
        latency = float(candidate_maps[routing][selected]["latency_us"])
        native = float(result["native"]["latency_us"])
        profiles[routing] = {
            "native_latency_us": native,
            "selected_latency_us": latency,
            "speedup_vs_native": round(native / latency, 6),
        }
    return {
        "num_tokens": next(iter(routing_results.values()))["num_tokens"],
        "selected_profile_ids": list(selected),
        "max_regret_vs_profile_optimum": round(regret(selected), 6),
        "profiles": profiles,
    }


def _cuda_time_us(call, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        call()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) * 1000.0 / iterations


def _make_weights(args, device: torch.device) -> tuple[torch.Tensor, ...]:
    from flashinfer import block_scale_interleave

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    w13 = torch.randint(
        0,
        256,
        (args.num_experts, 2 * args.intermediate_size, args.hidden_size // 2),
        generator=generator,
        dtype=torch.uint8,
    ).to(device)
    w2 = torch.randint(
        0,
        256,
        (args.num_experts, args.hidden_size, args.intermediate_size // 2),
        generator=generator,
        dtype=torch.uint8,
    ).to(device)
    s13 = torch.full(
        (args.num_experts, 2 * args.intermediate_size, args.hidden_size // 32),
        127,
        dtype=torch.uint8,
        device=device,
    )
    s2 = torch.full(
        (args.num_experts, args.hidden_size, args.intermediate_size // 32),
        127,
        dtype=torch.uint8,
        device=device,
    )
    return (
        w13.view(torch.long),
        w2.view(torch.long),
        block_scale_interleave(s13).reshape_as(s13).view(torch.int32),
        block_scale_interleave(s2).reshape_as(s2).view(torch.int32),
    )


def _make_tokens(args, num_tokens: int, device: torch.device):
    from flashinfer import mxfp8_quantize

    generator = torch.Generator(device="cpu").manual_seed(args.seed + num_tokens)
    x = (
        torch.randn(num_tokens, args.hidden_size, generator=generator)
        .mul_(0.05)
        .bfloat16()
        .to(device)
    )
    x_quant, x_scale = mxfp8_quantize(x, True, alignment=32)
    offsets = torch.arange(args.top_k, dtype=torch.int32, device=device)[None, :]
    if args.routing == "spread":
        rows = torch.arange(num_tokens, dtype=torch.int32, device=device)[:, None]
        topk_ids = (rows * 7 + offsets * 41).remainder(args.num_experts)
    else:
        topk_ids = offsets.repeat(num_tokens, 1)
    topk_weights = torch.full(
        (num_tokens, args.top_k),
        1.0 / args.top_k,
        dtype=torch.float32,
        device=device,
    )
    return x_quant, x_scale, topk_ids, topk_weights


def _runner_and_tactics(args, device: torch.device):
    from flashinfer import ActivationType
    from flashinfer.fused_moe.core import get_cutlass_fused_moe_module

    major, minor = torch.cuda.get_device_capability(device)
    module = get_cutlass_fused_moe_module(f"{major}{minor}")
    # FlashInfer does not expose tactic discovery as public Python API.  The
    # generated op captures its runner class; keep this version-coupled detail
    # isolated here and fail loudly if upstream changes the contract.
    closure = getattr(module.cutlass_fused_moe, "__closure__", None)
    if not closure or not isinstance(closure[0].cell_contents, type):
        raise RuntimeError("FlashInfer CUTLASS MoE runner discovery contract changed")
    runner_cls = closure[0].cell_contents
    runner = runner_cls(
        torch.float8_e4m3fn,
        torch.long,
        torch.bfloat16,
        args.top_k,
        args.tp_size,
        args.tp_rank,
        args.ep_size,
        args.ep_rank,
        1,
        0,
        False,
        False,
        False,
        True,
        False,
        args.enable_pdl,
        ActivationType.SwigluBias,
        False,
        True,
        False,
    )
    native = runner.fused_moe_runner
    gemm1_count = int(native.get_gemm1_tactic_count())
    gemm2_count = int(native.get_gemm2_tactic_count())
    gemm1 = [
        tactic
        for tactic in range(gemm1_count)
        if int(native.get_tactic_occupancy(tactic)) > 0
    ]
    gemm2 = [
        tactic
        for tactic in range(gemm1_count, gemm1_count + gemm2_count)
        if int(native.get_tactic_occupancy(tactic)) > 0
    ]
    shape_gemm1 = set(
        map(
            int,
            native.get_valid_tactics_for_shape(
                1, 2 * args.intermediate_size, args.hidden_size
            ),
        )
    )
    shape_gemm2 = set(
        map(
            int,
            native.get_valid_tactics_for_shape(
                2, args.hidden_size, args.intermediate_size
            ),
        )
    )
    gemm1 = [tactic for tactic in gemm1 if tactic in shape_gemm1]
    gemm2 = [tactic for tactic in gemm2 if tactic in shape_gemm2]
    if not gemm1 or not gemm2:
        raise RuntimeError(
            f"no shape-valid tactics: gemm1={gemm1!r}, gemm2={gemm2!r}"
        )
    return gemm1, gemm2


def _make_call(args, tensors, workspace, output, profile_ids):
    from flashinfer import ActivationType, cutlass_fused_moe

    (
        x_quant,
        x_scale,
        topk_ids,
        topk_weights,
        w13,
        w2,
        s13,
        s2,
        global_scale,
        swiglu_limit,
    ) = tensors

    def call():
        return cutlass_fused_moe(
            input=x_quant,
            input_sf=x_scale,
            swizzled_input_sf=True,
            token_selected_experts=topk_ids,
            token_final_scales=topk_weights,
            fc1_expert_weights=w13,
            fc2_expert_weights=w2,
            fc1_expert_biases=None,
            fc2_expert_biases=None,
            output_dtype=torch.bfloat16,
            output=output,
            quant_scales=[s13, global_scale, s2, global_scale],
            swiglu_alpha=global_scale,
            swiglu_beta=None,
            swiglu_limit=swiglu_limit,
            ep_size=args.ep_size,
            ep_rank=args.ep_rank,
            tp_size=args.tp_size,
            tp_rank=args.tp_rank,
            use_mxfp8_act_scaling=True,
            tune_max_num_tokens=max(args.buckets),
            enable_pdl=args.enable_pdl,
            activation_type=ActivationType.SwigluBias,
            profile_ids=profile_ids,
            workspace_buffer=workspace,
        )[0]

    return call


def _measure_tactic(
    args,
    tensors,
    workspace,
    reference: torch.Tensor,
    profile_ids: list[int] | None,
) -> dict[str, Any]:
    output = torch.empty_like(reference)
    call = _make_call(args, tensors, workspace, output, profile_ids)
    label = "native" if profile_ids is None else str(profile_ids)
    try:
        call()
        torch.cuda.synchronize()
        if not bool(torch.isfinite(output).all()):
            return {"profile_ids": profile_ids, "status": "nonfinite"}
        torch.testing.assert_close(output, reference, rtol=0.05, atol=0.05)
        latency = _cuda_time_us(call, args.warmup, args.iterations)
        return {
            "profile_ids": profile_ids,
            "status": "ok",
            "latency_us": round(latency, 3),
        }
    except Exception as exc:
        try:
            torch.cuda.synchronize()
        except Exception:
            raise RuntimeError(
                f"CUDA context failed while measuring tactic {label}"
            ) from exc
        return {
            "profile_ids": profile_ids,
            "status": "error",
            "error": f"{type(exc).__name__}: {str(exc).splitlines()[0]}",
        }


def _sweep_bucket(args, num_tokens, weights, workspace, gemm1, gemm2, device):
    x_quant, x_scale, topk_ids, topk_weights = _make_tokens(
        args, num_tokens, device
    )
    w13, w2, s13, s2 = weights
    global_scale = torch.ones(args.num_experts, dtype=torch.float32, device=device)
    swiglu_limit = torch.full_like(global_scale, 10.0)
    tensors = (
        x_quant,
        x_scale,
        topk_ids,
        topk_weights,
        w13,
        w2,
        s13,
        s2,
        global_scale,
        swiglu_limit,
    )
    reference = torch.empty(
        (num_tokens, args.hidden_size), dtype=torch.bfloat16, device=device
    )
    native_call = _make_call(args, tensors, workspace, reference, None)
    native_call()
    torch.cuda.synchronize()
    if not bool(torch.isfinite(reference).all()):
        raise RuntimeError(f"native heuristic produced non-finite output at {num_tokens}")
    native = _measure_tactic(args, tensors, workspace, reference, None)

    gemm1_results = [
        _measure_tactic(args, tensors, workspace, reference, [tactic, -1])
        for tactic in gemm1
    ]
    best_gemm1 = int(_best_valid(gemm1_results)["profile_ids"][0])
    gemm2_results = [
        _measure_tactic(args, tensors, workspace, reference, [best_gemm1, tactic])
        for tactic in gemm2
    ]
    valid_gemm1 = [
        int(item["profile_ids"][0])
        for item in gemm1_results
        if item["status"] == "ok"
    ]
    valid_gemm2 = [
        int(item["profile_ids"][1])
        for item in gemm2_results
        if item["status"] == "ok"
    ]
    pair_results = [
        _measure_tactic(args, tensors, workspace, reference, [gemm1_id, gemm2_id])
        for gemm1_id in valid_gemm1
        for gemm2_id in valid_gemm2
    ]
    best = _best_valid(pair_results)
    best_ids = list(map(int, best["profile_ids"]))
    combined = _measure_tactic(args, tensors, workspace, reference, best_ids)
    if combined["status"] != "ok":
        raise RuntimeError(
            f"selected tactic {best_ids} failed final measurement at {num_tokens}"
        )
    native_us = float(native["latency_us"])
    selected_us = float(combined["latency_us"])
    speedup = native_us / selected_us
    print(
        f"bucket={num_tokens:>4} native={native_us:>9.3f}us "
        f"selected={best_ids} {selected_us:>9.3f}us speedup={speedup:.3f}x",
        flush=True,
    )
    return {
        "num_tokens": num_tokens,
        "native": native,
        "gemm1_candidates": gemm1_results,
        "gemm2_candidates": gemm2_results,
        "pair_candidates": pair_results,
        "selected_profile_ids": best_ids,
        "selected_latency_us": round(selected_us, 3),
        "speedup_vs_native": round(speedup, 6),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--runtime-output",
        help="Optional compact tactic manifest for runtime packaging",
    )
    parser.add_argument("--buckets", type=_parse_buckets, default=DEFAULT_BUCKETS)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--ep-rank", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--routing",
        choices=("both", "spread", "concentrated"),
        default="both",
        help="Expert-token distribution(s) used for every bucket",
    )
    parser.add_argument("--enable-pdl", action="store_true")
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("warmup must be >= 0 and iterations must be > 0")
    if args.hidden_size % 128 or args.intermediate_size % 128:
        parser.error("hidden and intermediate sizes must be divisible by 128")

    device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    from flashinfer import ActivationType
    from flashinfer.fused_moe import cutlass_fused_moe_workspace_size

    gemm1, gemm2 = _runner_and_tactics(args, device)
    print(f"shape-valid tactics: gemm1={gemm1}, gemm2={gemm2}", flush=True)
    weights = _make_weights(args, device)
    workspace_size = cutlass_fused_moe_workspace_size(
        max(args.buckets),
        args.hidden_size,
        args.intermediate_size,
        args.num_experts,
        args.top_k,
        x_dtype=torch.float8_e4m3fn,
        weight_dtype=torch.long,
        output_dtype=torch.bfloat16,
        activation_type=ActivationType.SwigluBias,
        tp_size=args.tp_size,
        tp_rank=args.tp_rank,
        ep_size=args.ep_size,
        ep_rank=args.ep_rank,
        use_mxfp8_act_scaling=True,
        device=device,
    )
    workspace = torch.empty(workspace_size, dtype=torch.uint8, device=device)
    started = time.time()
    requested_routing = args.routing
    routing_profiles = (
        ("spread", "concentrated")
        if requested_routing == "both"
        else (requested_routing,)
    )
    routing_results = {}
    for routing in routing_profiles:
        args.routing = routing
        print(f"routing profile: {routing}", flush=True)
        routing_results[routing] = [
            _sweep_bucket(args, bucket, weights, workspace, gemm1, gemm2, device)
            for bucket in args.buckets
        ]
    args.routing = requested_routing
    if requested_routing == "both":
        results = [
            _select_robust_bucket(
                {
                    routing: routing_results[routing][index]
                    for routing in routing_profiles
                }
            )
            for index in range(len(args.buckets))
        ]
        for item in results:
            print(
                f"robust bucket={item['num_tokens']:>4} "
                f"selected={item['selected_profile_ids']} "
                f"max_regret={item['max_regret_vs_profile_optimum']:.3f}x",
                flush=True,
            )
    else:
        results = routing_results[requested_routing]
    major, minor = torch.cuda.get_device_capability(device)
    payload = {
        "schema_version": 1,
        "metadata": {
            "gpu": torch.cuda.get_device_name(device),
            "compute_capability": f"{major}.{minor}",
            "flashinfer_version": version("flashinfer-python"),
            "cutlass_tactic_abi_version": _cutlass_tactic_abi_version(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "hostname": host_platform.node(),
            "pid": os.getpid(),
        },
        "shape": {
            "hidden_size": args.hidden_size,
            "intermediate_size": args.intermediate_size,
            "num_experts": args.num_experts,
            "top_k": args.top_k,
            "tp_size": args.tp_size,
            "ep_size": args.ep_size,
            "activation_type": "swiglu_bias",
            "activation_dtype": "mxfp8_e4m3",
            "weight_dtype": "mxfp4_e2m1_e8m0",
            "output_dtype": "bfloat16",
            "fused_finalize": True,
            "enable_pdl": args.enable_pdl,
            "routing": requested_routing,
        },
        "candidate_counts": {"gemm1": len(gemm1), "gemm2": len(gemm2)},
        "warmup": args.warmup,
        "iterations": args.iterations,
        "elapsed_seconds": round(time.time() - started, 3),
        "buckets": results,
    }
    if requested_routing == "both":
        payload["routing_results"] = routing_results
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"saved {output}", flush=True)
    if args.runtime_output is not None:
        runtime_payload = {
            "schema_version": 1,
            "backend": "flashinfer_cutlass_mxfp8_mxfp4",
            "metadata": {
                key: payload["metadata"][key]
                for key in (
                    "gpu",
                    "compute_capability",
                    "flashinfer_version",
                    "cutlass_tactic_abi_version",
                    "torch_version",
                    "cuda_version",
                    "cudnn_version",
                )
            },
            "shape": payload["shape"],
            "tactics": {
                str(item["num_tokens"]): item["selected_profile_ids"]
                for item in results
            },
            "candidate_counts": payload["candidate_counts"],
        }
        runtime_output = Path(args.runtime_output)
        runtime_output.parent.mkdir(parents=True, exist_ok=True)
        runtime_output.write_text(json.dumps(runtime_payload, indent=2) + "\n")
        print(f"saved runtime manifest {runtime_output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
