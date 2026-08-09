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

"""Distributed runtime public exports.

Keep topology imports independent from optional communication kernels.  This
module is imported before Python resolves submodules such as ``mapping``; eager
imports here would otherwise load the CUDA/Triton communication stack merely
to construct a rank mapping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tokenspeed.runtime.distributed.comm_manager import CommManager
    from tokenspeed.runtime.distributed.mapping import Mapping

__all__ = ["CommManager", "Mapping"]


def __getattr__(name: str):
    if name == "CommManager":
        from tokenspeed.runtime.distributed.comm_manager import CommManager

        return CommManager
    if name == "Mapping":
        from tokenspeed.runtime.distributed.mapping import Mapping

        return Mapping
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
