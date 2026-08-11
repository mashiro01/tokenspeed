import multiprocessing
from datetime import timedelta

import pytest

pytest.importorskip("torch")

from tokenspeed.runtime.distributed import process_group_manager as process_groups
from tokenspeed.runtime.distributed.process_group_manager import ProcessGroupManager


def _run_pairwise_pipeline(rank, init_file, results):
    import torch
    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=3,
        timeout=timedelta(seconds=5),
    )
    manager = ProcessGroupManager()
    groups = ((0, 1), (1, 2))
    try:
        manager.init_explicit_process_groups(
            groups,
            backend="gloo",
            role="pipeline-p2p",
        )
        value = torch.tensor([rank], dtype=torch.int64)
        if rank == 0:
            value.fill_(7)
            dist.send(
                value,
                dst=1,
                group=manager.get_process_group(
                    "gloo", (0, 1), role="pipeline-p2p"
                ),
            )
        elif rank == 1:
            dist.recv(
                value,
                src=0,
                group=manager.get_process_group(
                    "gloo", (0, 1), role="pipeline-p2p"
                ),
            )
            value.add_(1)
            dist.send(
                value,
                dst=2,
                group=manager.get_process_group(
                    "gloo", (1, 2), role="pipeline-p2p"
                ),
            )
        else:
            dist.recv(
                value,
                src=1,
                group=manager.get_process_group(
                    "gloo", (1, 2), role="pipeline-p2p"
                ),
            )
            results.put(int(value.item()))
    finally:
        dist.destroy_process_group()


def test_process_group_roles_keep_independent_collective_lanes():
    manager = ProcessGroupManager()
    group = (0, 2, 4, 6)
    default = object()
    result = object()

    manager.register_process_group("nccl", group, default)
    manager.register_process_group("nccl", group, result, role="pipeline-result")

    assert manager.get_process_group("nccl", group) is default
    assert manager.get_process_group("nccl", group, role="pipeline-result") is result
    assert manager.has_process_group("nccl", group)
    assert manager.has_process_group("nccl", group, role="pipeline-result")
    assert not manager.has_process_group("nccl", group, role="pipeline-fault")


@pytest.mark.parametrize("role", ["", 0, None])
def test_process_group_role_must_be_a_non_empty_string(role):
    manager = ProcessGroupManager()

    with pytest.raises(ValueError, match="non-empty string"):
        manager.has_process_group("gloo", (0,), role=role)


def test_process_group_can_override_the_global_operation_timeout(monkeypatch):
    manager = ProcessGroupManager()
    manager._pg_timeout = timedelta(seconds=1800)
    group = (0, 1)
    created = []

    monkeypatch.setattr(process_groups, "_make_all_groups", lambda value: [value])

    def new_group(ranks, *, backend, timeout):
        created.append((ranks, backend, timeout))
        return object()

    monkeypatch.setattr(process_groups.dist, "new_group", new_group)

    manager.init_process_group(
        group,
        backend="gloo",
        role="pipeline-fault-upstream",
        timeout_seconds=300,
    )

    assert created == [(group, "gloo", timedelta(seconds=300))]
    assert manager.has_process_group("gloo", group, role="pipeline-fault-upstream")


def test_explicit_process_groups_support_overlapping_pairs(monkeypatch):
    manager = ProcessGroupManager()
    manager._pg_timeout = timedelta(seconds=30)
    groups = ((0, 1), (1, 2))
    created = []

    monkeypatch.setattr(process_groups.dist, "get_rank", lambda: 1)

    def new_group(ranks, *, backend, timeout):
        process_group = object()
        created.append((ranks, backend, timeout, process_group))
        return process_group

    monkeypatch.setattr(process_groups.dist, "new_group", new_group)

    manager.init_explicit_process_groups(
        groups,
        backend="nccl",
        role="pipeline-p2p",
    )

    assert [entry[:3] for entry in created] == [
        ((0, 1), "nccl", timedelta(seconds=30)),
        ((1, 2), "nccl", timedelta(seconds=30)),
    ]
    assert manager.get_process_group(
        "nccl", (0, 1), role="pipeline-p2p"
    ) is created[0][3]
    assert manager.get_process_group(
        "nccl", (1, 2), role="pipeline-p2p"
    ) is created[1][3]


def test_explicit_pairs_complete_a_sequential_three_stage_pipeline(tmp_path):
    init_file = str(tmp_path / "pairwise-pipeline-init")
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    processes = [
        context.Process(
            target=_run_pairwise_pipeline,
            args=(rank, init_file, results),
        )
        for rank in range(3)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)

    assert [process.exitcode for process in processes] == [0, 0, 0]
    assert results.get(timeout=1) == 8
