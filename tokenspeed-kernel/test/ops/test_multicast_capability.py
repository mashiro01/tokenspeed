"""Regression coverage for the NVIDIA multicast capability contract."""

from types import SimpleNamespace

from tokenspeed_kernel.ops.communication.triton import state_supports_multicast


def test_state_supports_multicast_requires_explicit_capability() -> None:
    assert state_supports_multicast(SimpleNamespace(multicast_supported=True))
    assert not state_supports_multicast(SimpleNamespace(multicast_supported=False))
    assert not state_supports_multicast(SimpleNamespace())
