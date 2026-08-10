from datetime import timedelta

import pytest

pytest.importorskip("torch")

from tokenspeed.runtime.distributed import process_group_manager as process_groups
from tokenspeed.runtime.distributed.process_group_manager import ProcessGroupManager


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
