"""Dependency-light coverage for pipeline runtime topology configuration."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[3]


@contextmanager
def _isolated_server_args_module():
    module_names = (
        "tokenspeed",
        "tokenspeed.runtime",
        "tokenspeed.runtime.distributed",
        "tokenspeed.runtime.distributed.mapping",
        "tokenspeed.runtime.utils",
        "tokenspeed.runtime.utils.launcher",
        "tokenspeed.runtime.utils.network",
        "tokenspeed_kernel",
        "tokenspeed_kernel.ops",
        "tokenspeed_kernel.ops.attention",
        "tokenspeed_kernel.ops.attention.triton",
        "tokenspeed_kernel.ops.attention.triton.linear",
        "tokenspeed_kernel.ops.attention.triton.linear.chunk_delta_h",
        "tokenspeed_kernel.platform",
        "_pipeline_server_args_under_test",
    )
    previous = {name: sys.modules.get(name) for name in module_names}

    def package(name: str) -> ModuleType:
        module = ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
        return module

    try:
        for name in (
            "tokenspeed",
            "tokenspeed.runtime",
            "tokenspeed.runtime.distributed",
            "tokenspeed_kernel",
            "tokenspeed_kernel.ops",
            "tokenspeed_kernel.ops.attention",
            "tokenspeed_kernel.ops.attention.triton",
            "tokenspeed_kernel.ops.attention.triton.linear",
        ):
            package(name)

        mapping_name = "tokenspeed.runtime.distributed.mapping"
        mapping_spec = importlib.util.spec_from_file_location(
            mapping_name,
            _ROOT / "python/tokenspeed/runtime/distributed/mapping.py",
        )
        assert mapping_spec is not None and mapping_spec.loader is not None
        mapping_module = importlib.util.module_from_spec(mapping_spec)
        sys.modules[mapping_name] = mapping_module
        mapping_spec.loader.exec_module(mapping_module)

        chunk_module = ModuleType(
            "tokenspeed_kernel.ops.attention.triton.linear.chunk_delta_h"
        )
        chunk_module.CHUNK_SIZE = 64
        sys.modules[chunk_module.__name__] = chunk_module

        platform_module = ModuleType("tokenspeed_kernel.platform")
        platform_module.current_platform = lambda: SimpleNamespace(
            is_amd=False, is_nvidia=True, is_hopper_plus=True
        )
        sys.modules[platform_module.__name__] = platform_module

        logger = SimpleNamespace(info=lambda *args, **kwargs: None)
        utils_module = package("tokenspeed.runtime.utils")
        utils_module.get_amdgpu_memory_capacity = lambda: None
        utils_module.get_colorful_logger = lambda name: logger
        utils_module.get_nvgpu_memory_capacity = lambda: None
        utils_module.is_valid_ipv6_address = lambda value: False
        utils_module.maybe_model_redirect = lambda value: value
        utils_module.nullable_str = lambda value: value

        launcher_module = ModuleType("tokenspeed.runtime.utils.launcher")
        launcher_module.check_dist_init_port = lambda port: None
        launcher_module.detect_topology = lambda: None
        sys.modules[launcher_module.__name__] = launcher_module

        network_module = ModuleType("tokenspeed.runtime.utils.network")
        network_module.is_port_available = lambda *args, **kwargs: True
        sys.modules[network_module.__name__] = network_module

        server_args_spec = importlib.util.spec_from_file_location(
            "_pipeline_server_args_under_test",
            _ROOT / "python/tokenspeed/runtime/utils/server_args.py",
        )
        assert server_args_spec is not None and server_args_spec.loader is not None
        server_args_module = importlib.util.module_from_spec(server_args_spec)
        sys.modules[server_args_spec.name] = server_args_module
        server_args_spec.loader.exec_module(server_args_module)
        yield server_args_module
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _server_args(module, **kwargs):
    module.ServerArgs.__post_init__ = lambda self: None
    return module.ServerArgs(model="test/model", **kwargs)


def test_global_world_is_split_into_stage_local_parallelism() -> None:
    with _isolated_server_args_module() as module:
        args = _server_args(
            module,
            world_size=64,
            pipeline_parallel_size=8,
            attn_tp_size=8,
        )
        args.resolve_parallelism()
        args.mapping.rank = 43

        assert args.mapping.world_size == 64
        assert args.mapping.pipeline.stage_count == 8
        assert args.mapping.pipeline.stage_world_size == 8
        assert args.mapping.pipeline.stage_group == tuple(range(40, 48))
        assert args.mapping.pipeline.pipeline_group == (3, 11, 19, 27, 35, 43, 51, 59)
        assert args.mapping.attn.tp_size == 8
        assert args.mapping.dense.tp_size == 8
        assert args.mapping.moe.tp_size == 8
        assert args.mapping.vision.tp_size == 8


def test_inferred_world_and_expert_parallelism_are_stage_local() -> None:
    with _isolated_server_args_module() as module:
        args = _server_args(
            module,
            pipeline_parallel_size=8,
            attn_tp_size=8,
            enable_expert_parallel=True,
        )
        args.resolve_parallelism()

        assert args.mapping.world_size == 64
        assert args.mapping.pipeline.stage_world_size == 8
        assert args.ep_size == 8
        assert args.mapping.moe.ep_size == 8
        assert args.mapping.moe.tp_size == 1


def test_pipeline_parallel_size_cli_and_pp1_regression() -> None:
    with _isolated_server_args_module() as module:
        parser = argparse.ArgumentParser()
        module.ServerArgs.add_cli_args(parser)
        namespace = parser.parse_args(
            ["--model", "test/model", "--pipeline-parallel-size", "8"]
        )
        assert namespace.pipeline_parallel_size == 8

        args = _server_args(module, world_size=8, attn_tp_size=8)
        args.resolve_parallelism()
        args.mapping.rank = 3

        assert args.mapping.pipeline.stage_count == 1
        assert args.mapping.pipeline.stage_world_size == 8
        assert args.mapping.pipeline.stage_group == tuple(range(8))
        assert args.mapping.pipeline.pipeline_group == (3,)
        assert args.mapping.attn.tp_group == tuple(range(8))
        assert args.mapping.world_group == tuple(range(8))


def test_pipeline_validation_fails_closed_for_unproven_runtime_features() -> None:
    with _isolated_server_args_module() as module:
        args = _server_args(
            module,
            world_size=64,
            pipeline_parallel_size=8,
            attn_tp_size=8,
        )
        args.resolve_parallelism()

        with pytest.raises(ValueError, match="pipeline parallelism.*CUDA graphs"):
            args.validate()


def test_pipeline_validation_rejects_online_weight_transfer_explicitly() -> None:
    with _isolated_server_args_module() as module:
        args = _server_args(
            module,
            world_size=64,
            pipeline_parallel_size=8,
            attn_tp_size=8,
            enforce_eager=True,
            disable_prefill_graph=True,
            disable_overlap_schedule=True,
            disable_autotune=True,
            enable_prefix_caching=False,
            enable_kvstore=False,
            grammar_backend="none",
            weight_transfer_config="{}",
        )
        args.resolve_parallelism()

        with pytest.raises(ValueError, match="online weight transfer"):
            args.validate()


def test_pipeline_validation_allows_stage_synchronous_autotune() -> None:
    with _isolated_server_args_module() as module:
        args = _server_args(
            module,
            world_size=64,
            pipeline_parallel_size=8,
            attn_tp_size=8,
            enforce_eager=True,
            disable_prefill_graph=True,
            disable_overlap_schedule=True,
            disable_autotune=False,
            enable_prefix_caching=False,
            enable_kvstore=False,
            grammar_backend="none",
        )
        args.resolve_parallelism()

        args.validate()


def test_pipeline_validation_allows_stage_local_warmup() -> None:
    with _isolated_server_args_module() as module:
        args = _server_args(
            module,
            world_size=64,
            pipeline_parallel_size=8,
            attn_tp_size=8,
            enforce_eager=True,
            disable_prefill_graph=True,
            disable_overlap_schedule=True,
            disable_autotune=True,
            enable_prefix_caching=False,
            enable_kvstore=False,
            grammar_backend="none",
            enable_pipeline_local_warmup=True,
            pipeline_local_warmup_max_tokens=8192,
        )
        args.resolve_parallelism()

        args.validate()


def test_pipeline_local_warmup_cli_and_pp1_validation() -> None:
    with _isolated_server_args_module() as module:
        parser = argparse.ArgumentParser()
        module.ServerArgs.add_cli_args(parser)
        namespace = parser.parse_args(
            [
                "--model",
                "test/model",
                "--enable-pipeline-local-warmup",
                "--pipeline-local-warmup-max-tokens",
                "4096",
            ]
        )
        assert namespace.enable_pipeline_local_warmup is True
        assert namespace.pipeline_local_warmup_max_tokens == 4096

        args = _server_args(module, world_size=8, attn_tp_size=8)
        args.enable_pipeline_local_warmup = True
        args.resolve_parallelism()
        with pytest.raises(ValueError, match="pipeline_parallel_size"):
            args.validate()


def test_single_node_pipeline_stage_keeps_allreduce_fusion_eligible() -> None:
    with _isolated_server_args_module() as module:
        args = _server_args(
            module,
            world_size=64,
            pipeline_parallel_size=8,
            attn_tp_size=8,
            nprocs_per_node=8,
            nnodes=8,
        )
        args.resolve_parallelism()
        args.resolve_communication()

        assert args.mapping.pipeline.stage_world_size == 8
        assert args.enable_allreduce_fusion is True


@pytest.mark.parametrize(
    ("world_size", "pipeline_parallel_size", "error", "message"),
    [
        (64, 0, ValueError, "pipeline_parallel_size must be positive"),
        (63, 8, ValueError, "must be divisible by pipeline_parallel_size"),
        (64, True, TypeError, "pipeline_parallel_size must be an int"),
    ],
)
def test_invalid_pipeline_topology_is_rejected(
    world_size, pipeline_parallel_size, error, message
) -> None:
    with _isolated_server_args_module() as module:
        args = _server_args(
            module,
            world_size=world_size,
            pipeline_parallel_size=pipeline_parallel_size,
        )
        with pytest.raises(error, match=message):
            args.resolve_parallelism()
