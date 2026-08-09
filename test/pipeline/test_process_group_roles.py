import pytest

pytest.importorskip("torch")

from tokenspeed.runtime.distributed.process_group_manager import ProcessGroupManager


def test_process_group_roles_keep_independent_collective_lanes():
    manager = ProcessGroupManager()
    group = (0, 2, 4, 6)
    default = object()
    result = object()

    manager.register_process_group("nccl", group, default)
    manager.register_process_group("nccl", group, result, role="pipeline-result")

    assert manager.get_process_group("nccl", group) is default
    assert (
        manager.get_process_group("nccl", group, role="pipeline-result") is result
    )
    assert manager.has_process_group("nccl", group)
    assert manager.has_process_group("nccl", group, role="pipeline-result")
    assert not manager.has_process_group("nccl", group, role="pipeline-fault")


@pytest.mark.parametrize("role", ["", 0, None])
def test_process_group_role_must_be_a_non_empty_string(role):
    manager = ProcessGroupManager()

    with pytest.raises(ValueError, match="non-empty string"):
        manager.has_process_group("gloo", (0,), role=role)
