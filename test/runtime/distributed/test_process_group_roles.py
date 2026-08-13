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

import pytest

from tokenspeed.runtime.distributed.process_group_manager import ProcessGroupManager


def test_process_group_roles_keep_dcp_lane_independent():
    manager = ProcessGroupManager()
    group = (0, 1)
    default = object()
    dcp = object()

    manager.register_process_group("nccl", group, default)
    manager.register_process_group("nccl", group, dcp, role="dcp")

    assert manager.get_process_group("nccl", group) is default
    assert manager.get_process_group("nccl", group, role="dcp") is dcp
    assert manager.has_process_group("nccl", group)
    assert manager.has_process_group("nccl", group, role="dcp")
    assert not manager.has_process_group("nccl", group, role="missing")


@pytest.mark.parametrize("role", ["", 0, None])
def test_process_group_role_must_be_a_non_empty_string(role):
    manager = ProcessGroupManager()

    with pytest.raises(ValueError, match="non-empty string"):
        manager.has_process_group("nccl", (0,), role=role)
