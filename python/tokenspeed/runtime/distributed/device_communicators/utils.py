# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
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

"""Distributed process-group helper utilities."""

import contextlib
from multiprocessing import shared_memory
from unittest.mock import patch

import torch
import torch.distributed
from torch.distributed import ProcessGroup

from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


def _ignore_resource_tracker_register(*args, **kwargs) -> None:
    return None


def in_the_same_node_as(pg: ProcessGroup, source_rank: int = 0) -> list[bool]:
    """
    Return whether every rank is on the same node as the source rank.

    This collective operation tests whether the processes share a memory system
    and can access the same shared-memory segments.
    """
    if torch.distributed.get_backend(pg) == torch.distributed.Backend.NCCL:
        raise ValueError("in_the_same_node_as should be tested with a non-NCCL group.")
    # Local rank within the group.
    rank = torch.distributed.get_rank(group=pg)
    world_size = torch.distributed.get_world_size(group=pg)

    # Local tensor in which each process stores the result.
    is_in_the_same_node = torch.tensor([0] * world_size, dtype=torch.int32)

    # Global ranks of the processes in the group.
    ranks = torch.distributed.get_process_group_ranks(pg)

    magic_message = b"magic_message"
    shm = None

    try:
        with contextlib.suppress(OSError):
            if rank == source_rank:
                # Create a shared-memory segment.
                shm = shared_memory.SharedMemory(create=True, size=128)
                shm.buf[: len(magic_message)] = magic_message
                torch.distributed.broadcast_object_list(
                    [shm.name], src=ranks[source_rank], group=pg
                )
                is_in_the_same_node[rank] = 1
            else:
                # Try to open the shared-memory segment.
                recv = [None]
                torch.distributed.broadcast_object_list(
                    recv, src=ranks[source_rank], group=pg
                )
                name = recv[0]
                # Work around https://stackoverflow.com/q/62748654/9191338.
                # Python incorrectly tracks shared memory even if it is not
                # created by the process. The following patch is a workaround.
                with patch(
                    "multiprocessing.resource_tracker.register",
                    _ignore_resource_tracker_register,
                ):
                    shm = shared_memory.SharedMemory(name=name)
                if shm.buf[: len(magic_message)] == magic_message:
                    is_in_the_same_node[rank] = 1
    except Exception as exc:
        logger.error("Error ignored in is_in_the_same_node: %s", exc)
    finally:
        if shm:
            shm.close()

    torch.distributed.barrier(group=pg)

    # Clean up the shared-memory segment.
    with contextlib.suppress(OSError):
        if rank == source_rank and shm:
            shm.unlink()
    torch.distributed.all_reduce(is_in_the_same_node, group=pg)

    return [x == 1 for x in is_in_the_same_node.tolist()]
