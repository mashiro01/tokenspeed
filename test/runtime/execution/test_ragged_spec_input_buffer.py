from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.execution.input_buffer import InputBuffers
from tokenspeed.runtime.execution.runtime_states import RuntimeStates


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compact_decode_packs_only_requested_candidate_prefixes():
    """A 3+1 verify batch must feed four, not eight, target-token rows."""
    device = "cuda"
    buffers = InputBuffers(
        max_bs=2,
        max_num_tokens=8,
        page_size=4,
        dummy_kv_slot=0,
        state_write_padding_pool_index=2,
        device=device,
    )
    runtime_states = RuntimeStates(
        req_pool_size=2,
        vocab_size=1_000,
        output_length=4,
        device=device,
    )
    runtime_states.future_input_map[0] = torch.tensor(
        [11, 12, 13, 14], device=device, dtype=torch.int32
    )
    runtime_states.future_input_map[1] = torch.tensor(
        [21, 22, 23, 24], device=device, dtype=torch.int32
    )
    runtime_states.valid_cache_lengths[:2] = torch.tensor(
        [2, 4], device=device, dtype=torch.int32
    )
    forward_op = SimpleNamespace(
        request_ids=["r0", "r1"],
        request_pool_indices=[0, 1],
        input_lengths=[3, 1],
        extend_prefix_lens=[],
        prefill_lengths=[],
        decode_input_ids=None,
        num_extends=lambda: 0,
    )
    page_table = torch.tensor(
        [[1, 2], [3, 4]], device=device, dtype=torch.int32
    )

    buffers.fill_input_buffers(
        forward_op=forward_op,
        runtime_states=runtime_states,
        total_tokens=4,
        page_table=page_table,
    )
    torch.cuda.synchronize()

    assert buffers.input_ids_buf[:4].cpu().tolist() == [11, 12, 13, 21]
    assert buffers.positions_buf[:4].cpu().tolist() == [2, 3, 4, 4]
    assert buffers.out_cache_loc_buf[:4].cpu().tolist() == [6, 7, 8, 16]
    assert buffers.seq_lens_buf[:2].cpu().tolist() == [5, 5]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_uniform_compact_decode_keeps_fused_input_preparation():
    """A shared short width still packs prefixes and uses valid cache offsets."""
    device = "cuda"
    buffers = InputBuffers(
        max_bs=2,
        max_num_tokens=8,
        page_size=4,
        dummy_kv_slot=0,
        state_write_padding_pool_index=2,
        device=device,
    )
    runtime_states = RuntimeStates(
        req_pool_size=2,
        vocab_size=1_000,
        output_length=4,
        device=device,
    )
    runtime_states.future_input_map[0] = torch.tensor(
        [31, 32, 33, 34], device=device, dtype=torch.int32
    )
    runtime_states.future_input_map[1] = torch.tensor(
        [41, 42, 43, 44], device=device, dtype=torch.int32
    )
    runtime_states.valid_cache_lengths[:2] = torch.tensor(
        [1, 5], device=device, dtype=torch.int32
    )
    forward_op = SimpleNamespace(
        request_ids=["r0", "r1"],
        request_pool_indices=[0, 1],
        input_lengths=[2, 2],
        extend_prefix_lens=[],
        prefill_lengths=[],
        decode_input_ids=None,
        num_extends=lambda: 0,
    )
    page_table = torch.tensor(
        [[1, 2], [3, 4]], device=device, dtype=torch.int32
    )

    buffers.fill_input_buffers(
        forward_op=forward_op,
        runtime_states=runtime_states,
        total_tokens=4,
        page_table=page_table,
    )
    torch.cuda.synchronize()

    assert buffers.input_ids_buf[:4].cpu().tolist() == [31, 32, 41, 42]
    assert buffers.positions_buf[:4].cpu().tolist() == [1, 2, 5, 6]
    assert buffers.seq_lens_buf[:2].cpu().tolist() == [3, 7]
