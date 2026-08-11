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

"""Factory helpers for model runners and model executors."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

import tokenspeed.runtime.layers.attention.backends  # noqa: F401  # trigger register_backend() calls
from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.distributed.consensus import raise_on_rank_error
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.execution.drafter import get_drafter_impl
from tokenspeed.runtime.execution.model_executor import (
    ModelExecutor,
    ModelExecutorConfig,
)
from tokenspeed.runtime.execution.model_runner import ModelRunner
from tokenspeed.runtime.pipeline.kimi_k3_dspark import (
    resolve_kimi_k3_dspark_placement,
    split_kimi_k3_dspark_context_projection,
)
from tokenspeed.runtime.sampling.registry import create_sampling_backend
from tokenspeed.runtime.utils.nvtx import set_nvtx_enabled
from tokenspeed.runtime.utils.server_args import ServerArgs
from tokenspeed.runtime.utils.hf_transformers_utils import resolve_architecture

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
    from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool


def _eagle_aux_layer_ids(hf_config) -> list[int] | None:
    """Return EAGLE3 capture ids from a draft config, including K3 text config.

    K3 wraps the language configuration in ``text_config``.  Draft exports may
    place ``eagle_config`` either on that text config or on the top-level
    wrapper, so inspect both without falling back to the target's defaults.
    """
    candidates = [hf_config]
    text_config = (
        hf_config.get("text_config")
        if isinstance(hf_config, dict)
        else getattr(hf_config, "text_config", None)
    )
    if text_config is not None:
        candidates.append(text_config)

    for config in candidates:
        if isinstance(config, dict):
            eagle_config = config.get("eagle_config")
            direct_ids = config.get("eagle_aux_hidden_state_layer_ids")
        else:
            eagle_config = getattr(config, "eagle_config", None)
            direct_ids = getattr(config, "eagle_aux_hidden_state_layer_ids", None)
        if isinstance(eagle_config, dict):
            ids = eagle_config.get("eagle_aux_hidden_state_layer_ids")
        elif eagle_config is not None:
            ids = getattr(eagle_config, "eagle_aux_hidden_state_layer_ids", None)
        else:
            ids = direct_ids
        if ids:
            return list(ids)
    return None


def _wire_draft_to_target_model(
    server_args: ServerArgs,
    model_runner: ModelRunner,
    draft_model_runner: ModelRunner,
) -> None:
    """Model-to-model wiring that must happen right after both models load.

    Runs before create_attn_components profiles free memory for the KV-cache
    budget, so weights the draft shares with the target (embed/LM head) are
    released before profiling instead of being double-counted.
    """
    DrafterImpl = get_drafter_impl(
        server_args.speculative_algorithm, draft_model_runner.model
    )
    if DrafterImpl.shares_target_embed_head:
        embed, head = model_runner.model.get_embed_and_head()
        draft_model_runner.model.set_embed_and_head(embed, head)
    if server_args.speculative_algorithm == "EAGLE3" and hasattr(
        model_runner.model, "set_eagle3_layers_to_capture"
    ):
        # capture the layers the draft was trained on, not the default
        aux_layer_ids = server_args.eagle3_layers_to_capture or _eagle_aux_layer_ids(
            draft_model_runner.model_config.hf_config
        )
        model_runner.model.set_eagle3_layers_to_capture(aux_layer_ids)


def _is_k3_dspark_pipeline(
    server_args: ServerArgs,
    model_config: ModelConfig,
    draft_model_config: ModelConfig | None,
) -> bool:
    """Return whether this launch uses the PP0-owned K3 DSpark runtime."""

    if (
        draft_model_config is None
        or server_args.speculative_algorithm != "DSPARK"
        or server_args.mapping.pipeline.stage_count <= 1
    ):
        return False
    target_architecture = resolve_architecture(model_config.hf_config)
    draft_architecture = resolve_architecture(draft_model_config.hf_config)
    return (
        target_architecture == "KimiK3ForConditionalGeneration"
        and draft_architecture == "K3DSparkModel"
    )


def _require_k3_dspark_pipeline_pair(
    server_args: ServerArgs,
    model_config: ModelConfig,
    draft_model_config: ModelConfig | None,
) -> bool:
    """Validate a PP DSpark launch before allocating any model runner."""

    if (
        server_args.speculative_algorithm != "DSPARK"
        or server_args.mapping.pipeline.stage_count <= 1
    ):
        return False
    if _is_k3_dspark_pipeline(server_args, model_config, draft_model_config):
        return True

    target_architecture = resolve_architecture(model_config.hf_config)
    draft_architecture = (
        None
        if draft_model_config is None
        else resolve_architecture(draft_model_config.hf_config)
    )
    raise ValueError(
        "K3 pipeline DSPARK requires target architecture "
        "KimiK3ForConditionalGeneration and an external draft architecture "
        "K3DSparkModel; got "
        f"target={target_architecture!r}, draft={draft_architecture!r}."
    )


def _device_for_rank(server_args: ServerArgs, gpu_id: int) -> torch.device:
    device = torch.device(server_args.device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", gpu_id)
    return device


def _pipeline_broadcast(
    tensor: torch.Tensor,
    *,
    source_rank: int,
    group,
) -> None:
    work = dist.broadcast(
        tensor,
        src=source_rank,
        group=group,
        async_op=True,
    )
    work.wait()


def _configure_k3_dspark_pipeline(
    *,
    server_args: ServerArgs,
    model_config: ModelConfig,
    draft_model_config: ModelConfig,
    model_runner: ModelRunner,
    draft_model_runner: ModelRunner | None,
    gpu_id: int,
) -> None:
    """Install PP K3 DSpark slices, PP0 draft head, and free full projection.

    Each pipeline lane has one PP0 source rank and one PP7 target-head source
    rank. The context projection is split once from the PP0 draft and each
    rank retains only the column slice for target taps that belong to its
    pipeline stage. PP0 then drops the full matrix before cache profiling.
    """

    mapping = server_args.mapping
    pipeline = mapping.pipeline
    device = _device_for_rank(server_args, gpu_id)
    pipeline_group = pg_manager.get_process_group("nccl", pipeline.pipeline_group)
    first_rank = pipeline.pipeline_group[0]
    last_rank = pipeline.pipeline_group[-1]
    placement = None
    source_slices = None
    local_error = None
    try:
        target_model = model_runner.model
        target_plan = getattr(target_model, "pipeline_plan", None)
        if target_plan is None:
            raise RuntimeError("K3 target model has no pipeline plan")
        draft_config = draft_model_config.hf_config
        placement = resolve_kimi_k3_dspark_placement(
            target_plan,
            target_layer_ids=tuple(
                int(layer) for layer in draft_config.target_layer_ids
            ),
            target_hidden_size=int(draft_config.target_hidden_size),
            context_hidden_size=int(draft_config.hidden_size),
        )
        if pipeline.is_first_stage:
            if draft_model_runner is None:
                raise RuntimeError("K3 DSpark pipeline did not load its PP0 draft")
            context_proj = getattr(draft_model_runner.model, "context_proj", None)
            weight = getattr(context_proj, "weight", None)
            if weight is None:
                raise RuntimeError("K3 DSpark draft has no loaded context projection")
            if weight.dtype != model_config.dtype:
                raise ValueError(
                    "K3 DSpark context projection dtype must match the target "
                    f"dtype: {weight.dtype} != {model_config.dtype}"
                )
            source_slices = split_kimi_k3_dspark_context_projection(weight, placement)
    except Exception as exc:  # noqa: BLE001 - fence every rank before NCCL
        local_error = exc
    raise_on_rank_error(local_error, mapping, "K3 DSpark pipeline preflight")
    assert placement is not None

    local_projection_weights: dict[int, torch.Tensor] = {}
    for projection_slice in placement.projection_slices:
        local_error = None
        tensor = None
        try:
            if pipeline.is_first_stage:
                assert source_slices is not None
                tensor = source_slices[projection_slice.stage_id][
                    projection_slice.target_layer_id
                ]
            else:
                tensor = torch.empty(
                    (
                        placement.context_hidden_size,
                        placement.target_hidden_size,
                    ),
                    dtype=model_config.dtype,
                    device=device,
                )
        except Exception as exc:  # noqa: BLE001 - fence allocation failures
            local_error = exc
        raise_on_rank_error(
            local_error,
            mapping,
            f"K3 DSpark projection slice {projection_slice.target_layer_id}",
        )
        assert tensor is not None
        _pipeline_broadcast(tensor, source_rank=first_rank, group=pipeline_group)
        if pipeline.stage_index == projection_slice.stage_id:
            local_projection_weights[projection_slice.target_layer_id] = tensor

    local_error = None
    try:
        model_runner.model.configure_dspark_pipeline(
            placement,
            projection_weights=local_projection_weights,
            activation_dtype=str(model_config.dtype).removeprefix("torch."),
        )
        # The PP0 drafter is instantiated later, but every target stage must
        # register its local DSpark tap before the first pipeline forward.
        # Registering the common placement here keeps the context accumulator
        # complete across PP0..PP7 rather than only enabling PP0's tap when
        # DFlash wires its local draft.
        model_runner.model.set_dflash_layers_to_capture(
            list(placement.target_layer_ids)
        )
    except Exception as exc:  # noqa: BLE001 - fail all ranks before head copy
        local_error = exc
    raise_on_rank_error(local_error, mapping, "K3 DSpark target configuration")

    head_shape = torch.empty(2, dtype=torch.int64, device=device)
    head_dtype = model_config.dtype
    local_error = None
    try:
        if pipeline.is_last_stage:
            source_weight = model_runner.model.get_pipeline_dspark_source_head_weight(
                expected_dtype=head_dtype
            )
            head_shape.copy_(
                torch.tensor(source_weight.shape, dtype=torch.int64, device=device)
            )
    except Exception as exc:  # noqa: BLE001 - fence head discovery failures
        local_error = exc
    raise_on_rank_error(local_error, mapping, "K3 DSpark head preflight")
    _pipeline_broadcast(head_shape, source_rank=last_rank, group=pipeline_group)

    local_error = None
    head_weight = None
    try:
        rows, cols = (int(value) for value in head_shape.tolist())
        if rows < 1 or cols != int(draft_model_config.hf_config.hidden_size):
            raise ValueError(
                "K3 DSpark target LM head geometry is invalid: "
                f"{(rows, cols)}"
            )
        if pipeline.is_last_stage:
            head_weight = model_runner.model.get_pipeline_dspark_source_head_weight(
                expected_dtype=head_dtype
            )
        else:
            head_weight = torch.empty(
                (rows, cols), dtype=head_dtype, device=device
            )
    except Exception as exc:  # noqa: BLE001 - fence allocation failures
        local_error = exc
    raise_on_rank_error(local_error, mapping, "K3 DSpark head allocation")
    assert head_weight is not None
    _pipeline_broadcast(head_weight, source_rank=last_rank, group=pipeline_group)

    local_error = None
    try:
        if pipeline.is_first_stage:
            model_runner.model.install_pipeline_dspark_draft_head(head_weight)
            assert draft_model_runner is not None
            draft_model_runner.model.enable_pipeline_projected_context()
    except Exception as exc:  # noqa: BLE001 - keep every rank in startup fence
        local_error = exc
    raise_on_rank_error(local_error, mapping, "K3 DSpark PP0 draft head setup")


def create_model_runner(
    server_args: ServerArgs,
    model_config: ModelConfig,
    draft_model_config: ModelConfig | None,
    gpu_id: int,
    global_rank: int,
):
    """Create the main model runner and optional draft model runner."""
    k3_dspark_pipeline = _require_k3_dspark_pipeline_pair(
        server_args,
        model_config,
        draft_model_config,
    )
    model_runner = ModelRunner(
        model_config=model_config,
        gpu_id=gpu_id,
        server_args=server_args,
        global_rank=global_rank,
    )

    draft_model_runner = None
    if draft_model_config is not None and (
        not k3_dspark_pipeline or server_args.mapping.pipeline.is_first_stage
    ):
        draft_model_runner = ModelRunner(
            model_config=draft_model_config,
            gpu_id=gpu_id,
            server_args=server_args,
            global_rank=global_rank,
            is_draft_worker=True,
        )
        if server_args.speculative_algorithm is not None:
            _wire_draft_to_target_model(server_args, model_runner, draft_model_runner)

    if k3_dspark_pipeline:
        assert draft_model_config is not None
        _configure_k3_dspark_pipeline(
            server_args=server_args,
            model_config=model_config,
            draft_model_config=draft_model_config,
            model_runner=model_runner,
            draft_model_runner=draft_model_runner,
            gpu_id=gpu_id,
        )

    return model_runner, draft_model_runner


def create_model_executor(
    server_args: ServerArgs,
    config: ModelExecutorConfig,
    model_runner: ModelRunner,
    attn_backend: AttentionBackend,
    token_to_kv_pool: CachePool,
    draft_model_runner: ModelRunner | None = None,
    draft_attn_backend: AttentionBackend | None = None,
    draft_token_to_kv_pool: CachePool | None = None,
) -> ModelExecutor:
    """Create the model executor with its sampler configuration."""
    if server_args.enable_nvtx:
        set_nvtx_enabled(True)

    max_bs = config.max_num_seqs // max(config.data_parallel_size, 1)

    max_draft_tokens_per_req = (
        config.spec_num_tokens if config.spec_algo is not None else 1
    )

    sampling_backend = create_sampling_backend(
        server_args,
        max_bs=max_bs,
        max_draft_tokens_per_req=max_draft_tokens_per_req,
        device=config.device,
        max_req_pool_size=config.max_req_pool_size,
        vocab_size=config.vocab_size,
        # Same TP group as LogitsProcessor.
        tp_group=model_runner.mapping.attn.tp_group,
    )

    return ModelExecutor(
        config=config,
        model_runner=model_runner,
        attn_backend=attn_backend,
        token_to_kv_pool=token_to_kv_pool,
        sampling_backend=sampling_backend,
        draft_model_runner=draft_model_runner,
        draft_attn_backend=draft_attn_backend,
        draft_token_to_kv_pool=draft_token_to_kv_pool,
    )
