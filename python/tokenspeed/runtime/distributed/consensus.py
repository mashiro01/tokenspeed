"""Fixed-count startup consensus helpers for distributed initialization."""

from __future__ import annotations

import torch
import torch.distributed as dist

from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)

_ERROR_MESSAGE_BYTES = 2048


def raise_on_rank_error(error: Exception | None, mapping, phase: str) -> None:
    """Propagate the first local startup exception to every global rank."""

    if mapping.world_size == 1:
        if error is not None:
            raise error
        return
    if not dist.is_initialized():
        raise RuntimeError(
            f"{phase} error consensus requires initialized distributed groups"
        )
    cpu_group = pg_manager.get_process_group("gloo", mapping.world_group)
    first_failure = torch.tensor(
        [mapping.rank if error is not None else mapping.world_size],
        dtype=torch.int64,
    )
    dist.all_reduce(first_failure, op=dist.ReduceOp.MIN, group=cpu_group)
    failure_rank = int(first_failure.item())
    if failure_rank == mapping.world_size:
        return

    message = torch.zeros(_ERROR_MESSAGE_BYTES, dtype=torch.uint8)
    if mapping.rank == failure_rank:
        assert error is not None
        encoded = f"{type(error).__name__}: {error}".encode("utf-8")
        encoded = encoded[: _ERROR_MESSAGE_BYTES - 1]
        message[: len(encoded)] = torch.tensor(list(encoded), dtype=torch.uint8)
    dist.broadcast(message, src=failure_rank, group=cpu_group)
    decoded = bytes(message.tolist()).split(b"\0", 1)[0].decode(
        "utf-8", errors="replace"
    )
    consensus_error = RuntimeError(
        f"{phase} failed on global rank {failure_rank}: {decoded}"
    )
    if error is not None:
        raise consensus_error from error
    raise consensus_error
