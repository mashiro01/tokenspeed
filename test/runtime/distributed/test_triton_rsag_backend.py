"""Unit tests for topology-aware Triton RS/AG dispatch."""

from types import SimpleNamespace
from unittest.mock import Mock

import torch

import tokenspeed.runtime.distributed.comm_backend.triton_rsag as rsag_module
from tokenspeed.runtime.distributed.comm_backend.triton_rsag import TritonRSAGBackend
from tokenspeed.runtime.utils.env import global_server_args_dict


def _backend(monkeypatch):
    fallback = Mock()
    backend = TritonRSAGBackend(fallback=fallback)
    monkeypatch.setattr(
        rsag_module,
        "current_platform",
        lambda: SimpleNamespace(is_nvidia=True),
    )
    monkeypatch.setattr(
        rsag_module.pg_manager,
        "get_process_group",
        lambda *_: object(),
    )
    monkeypatch.setattr(rsag_module.dist, "get_rank", lambda: 0)
    monkeypatch.setitem(
        global_server_args_dict,
        "mapping",
        SimpleNamespace(
            attn=SimpleNamespace(tp_size=2),
            dense=SimpleNamespace(tp_size=2),
            moe=SimpleNamespace(tp_ep_size=2),
        ),
    )
    monkeypatch.setitem(global_server_args_dict, "chunked_prefill_size", 8)
    monkeypatch.setitem(global_server_args_dict, "max_prefill_tokens", 8)
    monkeypatch.setitem(global_server_args_dict, "max_model_len", 8)
    monkeypatch.setattr(
        rsag_module,
        "create_state",
        lambda **kwargs: SimpleNamespace(multicast_supported=False),
    )
    return backend, fallback


def test_non_multicast_token_all_gather_falls_back_to_nccl(monkeypatch):
    backend, fallback = _backend(monkeypatch)
    tensor = torch.empty(1, 4)
    fallback.token_all_gather.return_value = "nccl-result"

    result = backend.token_all_gather(tensor, (0, 1), [1, 1])

    assert result == "nccl-result"
    fallback.token_all_gather.assert_called_once_with(
        tensor,
        group=(0, 1),
        scattered_num_tokens=[1, 1],
    )


def test_non_multicast_token_reduce_scatter_falls_back_to_nccl(monkeypatch):
    backend, fallback = _backend(monkeypatch)
    tensor = torch.empty(2, 4)
    fallback.token_reduce_scatter.return_value = "nccl-result"

    result = backend.token_reduce_scatter(tensor, (0, 1), [1, 1])

    assert result == "nccl-result"
    fallback.token_reduce_scatter.assert_called_once_with(
        tensor,
        group=(0, 1),
        scattered_num_tokens=[1, 1],
    )


def test_non_multicast_last_dim_all_gather_falls_back_to_nccl(monkeypatch):
    backend, fallback = _backend(monkeypatch)
    tensor = torch.empty(1, 4, dtype=torch.bfloat16)
    fallback.all_gather.return_value = "nccl-result"

    result = backend.all_gather(tensor, (0, 1), dim=-1)

    assert result == "nccl-result"
    fallback.all_gather.assert_called_once_with(tensor, group=(0, 1), dim=-1)
