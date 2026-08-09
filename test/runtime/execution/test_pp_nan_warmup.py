"""Unit coverage for eager PP warmup NaN flag sizing."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tokenspeed.runtime.execution.model_executor import ModelExecutor
from tokenspeed.runtime.execution.nan_guard import NanGuard


class _PacketBus:
    def __init__(self) -> None:
        self.packet: torch.Tensor | None = None
        self.packet_lengths: list[int] = []
        self.flag_lengths: list[int] = []

    def publish(self, tokens: torch.Tensor, lengths: torch.Tensor, flags: torch.Tensor):
        self.flag_lengths.append(flags.numel())
        self.packet = torch.stack((tokens, lengths, flags), dim=1).reshape(-1).clone()
        self.packet_lengths.append(self.packet.numel())

    def consume(self, batch_size: int) -> torch.Tensor:
        assert self.packet is not None
        assert self.packet.numel() == 3 * batch_size
        return self.packet.reshape(batch_size, 3)[:, 2]


class _FakeCudaGraphWrapper:
    def __init__(self, forward_func) -> None:
        self._forward_func = forward_func

    def prewarm_comm_states(self, batch_sizes: tuple[int, ...]) -> None:
        for bs in batch_sizes:
            self._forward_func(
                bs=bs,
                ctx=SimpleNamespace(bs=bs),
                sampling_info=SimpleNamespace(),
            )


def _executor(forward_func) -> ModelExecutor:
    executor = ModelExecutor.__new__(ModelExecutor)
    executor.nan_guard = NanGuard(max_bs=4, device="cpu")
    executor.forward_step = _FakeCudaGraphWrapper(forward_func)
    return executor


def test_pp2_eager_warmup_resets_nan_flags_to_dummy_batch_and_cleans_up():
    bus = _PacketBus()

    def final_stage(*, bs, ctx, sampling_info):
        assert final_executor.nan_guard.flags_device.numel() == bs
        final_executor.nan_guard.flags[:bs].fill_(1)
        bus.publish(
            torch.arange(bs, dtype=torch.int32),
            torch.ones(bs, dtype=torch.int32),
            final_executor.nan_guard.flags_device,
        )

    def first_stage(*, bs, ctx, sampling_info):
        assert first_executor.nan_guard.flags_device.numel() == bs
        first_executor.nan_guard.load_pipeline_flags(bus.consume(bs))
        assert first_executor.nan_guard.flags_device.tolist() == [1] * bs

    final_executor = _executor(final_stage)
    first_executor = _executor(first_stage)

    final_executor._prewarm_eager_comm_states(batch_sizes=(2,))
    first_executor._prewarm_eager_comm_states(batch_sizes=(2,))

    assert bus.packet_lengths == [6]
    assert bus.flag_lengths == [2]
    assert final_executor.nan_guard.flags_device.numel() == 0
    assert first_executor.nan_guard.flags_device.numel() == 0
    assert final_executor.nan_guard.flags.tolist() == [0, 0, 0, 0]
    assert first_executor.nan_guard.flags.tolist() == [0, 0, 0, 0]
