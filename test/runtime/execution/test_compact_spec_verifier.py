from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.drafter.dflash import DFlash
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.model_executor import ModelExecutor
from tokenspeed.runtime.execution.nan_guard import NanGuard
from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
from tokenspeed.runtime.sampling.backends.base import SamplingBackendConfig
from tokenspeed.runtime.sampling.backends.flashinfer import FlashInferSamplingBackend
from tokenspeed.runtime.sampling.backends.greedy import GreedySamplingBackend
from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo
from tokenspeed.runtime.sampling.sampling_params import _TOP_K_DISABLED


class _RecordingVerifier:
    def __init__(self):
        self.calls = []

    def verify(self, logits_output, sampling_info, candidates):
        self.calls.append(
            {
                "rows": sampling_info.verify_row_indices.tolist(),
                "pool_indices": sampling_info.req_pool_indices.tolist(),
                "mask": (
                    sampling_info.vocab_mask.squeeze(-1).tolist()
                    if sampling_info.vocab_mask is not None
                    else None
                ),
                "logit_rows": logits_output.next_token_logits.shape[0],
            }
        )
        return (
            logits_output.next_token_logits[:, 0].to(torch.int32),
            torch.full(
                (candidates.shape[0],),
                candidates.shape[1],
                dtype=torch.int32,
            ),
        )

    def sample(self, logits_output, sampling_info):
        self.calls.append(
            {
                "rows": sampling_info.verify_row_indices.tolist(),
                "pool_indices": sampling_info.req_pool_indices.tolist(),
                "mask": (
                    sampling_info.vocab_mask.squeeze(-1).tolist()
                    if sampling_info.vocab_mask is not None
                    else None
                ),
                "logit_rows": logits_output.next_token_logits.shape[0],
            }
        )
        return (
            logits_output.next_token_logits[:, 0].to(torch.int32),
            torch.ones(
                logits_output.next_token_logits.shape[0], dtype=torch.int32
            ),
        )


class _ConfidenceDrafter:
    def __init__(self, confidence_logits: torch.Tensor):
        self.confidence_logits = confidence_logits

    def get_last_confidence_logits(self) -> torch.Tensor:
        return self.confidence_logits


def _compact_executor() -> ModelExecutor:
    executor = ModelExecutor.__new__(ModelExecutor)
    executor.config = SimpleNamespace(spec_num_tokens=4, enable_output_logprobs=False)
    executor._spec_enabled = True
    executor._compact_spec_output_tokens_buf = torch.empty((2, 4), dtype=torch.int32)
    executor._compact_spec_accept_lengths_buf = torch.empty(2, dtype=torch.int32)
    executor._compact_spec_logprobs_buf = torch.empty((2, 4), dtype=torch.float32)
    executor.sampling_backend = _RecordingVerifier()
    return executor


def _compact_context() -> ForwardContext:
    return ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=2,
        num_extends=0,
        input_num_tokens=4,
        forward_mode=ForwardMode.DECODE,
        spec_verify_widths=(3, 1),
        compact_spec_verify=True,
    )


def test_compact_verify_groups_packed_logits_and_restores_fixed_output_layout():
    executor = _compact_executor()
    ctx = _compact_context()
    logits = torch.tensor(
        [[101, 0], [102, 0], [103, 0], [201, 0]], dtype=torch.float32
    )
    candidates = torch.tensor(
        [[11, 12, 13, 14], [21, 22, 23, 24]], dtype=torch.int32
    )
    sampling_info = SamplingBatchInfo(
        req_pool_indices=torch.tensor([7, 8], dtype=torch.int64),
        vocab_mask=torch.arange(8, dtype=torch.int32).view(8, 1),
    )

    output_tokens, accept_lengths, output_logprobs = (
        executor._verify_compact_spec_candidates(
            logits=logits,
            sampling_info=sampling_info,
            ctx=ctx,
            candidates=candidates,
        )
    )

    assert output_tokens.tolist() == [101, 102, 103, 0, 201, 0, 0, 0]
    assert accept_lengths.tolist() == [3, 1]
    assert output_logprobs is None
    assert executor.sampling_backend.calls == [
        {"rows": [0], "pool_indices": [7], "mask": [0, 1, 2], "logit_rows": 3},
        {"rows": [1], "pool_indices": [8], "mask": [4], "logit_rows": 1},
    ]


def test_nan_guard_attributes_compact_target_rows_to_their_request():
    guard = NanGuard(max_bs=2, device="cpu")
    ctx = _compact_context()
    guard.reset(2)

    guard._or_per_request(torch.tensor([False, True, False, True]), ctx)

    assert guard.flags.tolist() == [1, 1]


def test_confidence_profile_converts_draft_lengths_to_verify_widths():
    executor = ModelExecutor.__new__(ModelExecutor)
    executor.config = SimpleNamespace(spec_num_tokens=4)
    executor.drafter = _ConfidenceDrafter(torch.full((2, 3), 10.0))
    executor._dspark_schedule_temperatures = torch.ones(3)
    executor._dspark_schedule_steps_per_second = torch.ones(16)

    widths = executor._schedule_next_verify_widths(batch_size=2)

    assert widths.tolist() == [4, 4]


def test_compact_dflash_cache_selection_uses_packed_target_lengths():
    drafter = DFlash.__new__(DFlash)
    drafter.device = "cpu"
    drafter.input_buffers = SimpleNamespace(
        input_lengths_buf=torch.tensor([3, 1], dtype=torch.int32),
        req_pool_indices_buf=torch.tensor([0, 1], dtype=torch.int64),
        positions_buf=torch.tensor([10, 11, 12, 30], dtype=torch.int64),
        out_cache_loc_buf=torch.tensor([100, 101, 102, 300], dtype=torch.int32),
    )
    drafter.runtime_states = SimpleNamespace(
        valid_cache_lengths=torch.tensor([9, 29], dtype=torch.int32)
    )
    drafter.draft_seq_lens_buf = torch.empty(2, dtype=torch.int32)
    context = torch.arange(8, dtype=torch.float32).view(4, 2)

    selected, positions, cache_locs, decode_only = drafter._select_native_cache_rows(
        _compact_context(),
        context,
        torch.tensor([3, 1], dtype=torch.int32),
        label="test",
    )

    assert decode_only
    assert torch.equal(selected, context)
    assert positions.tolist() == [10, 11, 12, 30]
    assert cache_locs.tolist() == [100, 101, 102, 300]
    assert drafter.draft_seq_lens_buf.tolist() == [13, 31]


def test_mixed_compact_sampling_keeps_prefill_and_fixed_decode_stride():
    executor = _compact_executor()
    ctx = ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=3,
        num_extends=1,
        input_num_tokens=7,
        forward_mode=ForwardMode.MIXED,
        spec_verify_widths=(3, 1),
        compact_spec_verify=True,
    )
    executor._compact_spec_output_tokens_buf = torch.empty((3, 4), dtype=torch.int32)
    executor._compact_spec_accept_lengths_buf = torch.empty(3, dtype=torch.int32)
    executor._compact_spec_logprobs_buf = torch.empty((3, 4), dtype=torch.float32)
    logits_output = SimpleNamespace(
        next_token_logits=torch.tensor(
            [[90, 0], [101, 0], [102, 0], [103, 0], [201, 0]],
            dtype=torch.float32,
        ),
        next_token_logprobs=None,
    )
    candidates = torch.tensor(
        [[11, 12, 13, 14], [21, 22, 23, 24]], dtype=torch.int32
    )
    sampling_info = SamplingBatchInfo(
        req_pool_indices=torch.tensor([6, 7, 8], dtype=torch.int64),
        vocab_mask=torch.arange(12, dtype=torch.int32).view(12, 1),
    )

    output_tokens, accept_lengths = executor._run_sampling(
        logits_output, sampling_info, ctx, candidates
    )

    assert output_tokens.tolist() == [90, 101, 102, 103, 0, 201, 0, 0, 0]
    assert accept_lengths.tolist() == [1, 3, 1]
    assert executor.sampling_backend.calls == [
        {"rows": [0], "pool_indices": [6], "mask": [0], "logit_rows": 1},
        {"rows": [1], "pool_indices": [7], "mask": [4, 5, 6], "logit_rows": 3},
        {"rows": [2], "pool_indices": [8], "mask": [8], "logit_rows": 1},
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compact_verify_matches_cuda_greedy_prefixes():
    """Shortening a verify block must not change its accepted token prefix."""
    device = torch.device("cuda")
    backend = GreedySamplingBackend(
        SamplingBackendConfig(
            max_bs=2,
            max_draft_tokens_per_req=4,
            device=device,
            enable_tp_sync=False,
        )
    )
    candidates = torch.tensor(
        [[10, 11, 12, 13], [20, 21, 22, 23]],
        dtype=torch.int32,
        device=device,
    )
    target_ids = torch.tensor([11, 12, 99, 98, 88, 87, 86, 85], device=device)
    logits = torch.full((8, 128), -10.0, device=device)
    logits.scatter_(1, target_ids.view(-1, 1), 10.0)
    sampling_info = SamplingBatchInfo(
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int64, device=device),
        is_all_greedy=True,
    )

    static_tokens, static_lengths = backend.verify(
        LogitsProcessorOutput(next_token_logits=logits.clone()),
        sampling_info,
        candidates,
    )
    static_tokens = static_tokens.clone()
    static_lengths = static_lengths.clone()

    executor = _compact_executor()
    executor.sampling_backend = backend
    executor._compact_spec_output_tokens_buf = torch.empty(
        (2, 4), dtype=torch.int32, device=device
    )
    executor._compact_spec_accept_lengths_buf = torch.empty(
        2, dtype=torch.int32, device=device
    )
    executor._compact_spec_logprobs_buf = torch.empty(
        (2, 4), dtype=torch.float32, device=device
    )
    compact_logits = torch.cat([logits[:3], logits[4:5]])
    compact_tokens, compact_lengths, _ = executor._verify_compact_spec_candidates(
        logits=compact_logits,
        sampling_info=sampling_info,
        ctx=_compact_context(),
        candidates=candidates,
    )
    torch.cuda.synchronize()

    assert static_lengths.cpu().tolist() == [3, 1]
    assert compact_lengths.cpu().tolist() == [3, 1]
    static_rows = static_tokens.view(2, 4)
    compact_rows = compact_tokens.view(2, 4)
    assert torch.equal(static_rows[0, :3], compact_rows[0, :3])
    assert torch.equal(static_rows[1, :1], compact_rows[1, :1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compact_verify_preserves_flashinfer_per_request_coins():
    """Width grouping must not reuse row zero's random draws for row one."""
    device = torch.device("cuda")
    backend = FlashInferSamplingBackend(
        SamplingBackendConfig(
            max_bs=2,
            max_draft_tokens_per_req=4,
            max_req_pool_size=2,
            vocab_size=2,
            device=device,
            enable_tp_sync=False,
        )
    )
    backend._top_k_pool.fill_(_TOP_K_DISABLED)
    backend._top_p_pool.fill_(1.0)
    backend._temperature_pool.fill_(1.0)
    backend._coins_buf[0].fill_(0.01)
    backend._coins_buf[1].fill_(0.99)
    backend._final_coins_buf[0].fill_(0.01)
    backend._final_coins_buf[1].fill_(0.99)

    candidates = torch.zeros((2, 4), dtype=torch.int32, device=device)
    logits = torch.zeros((8, 2), dtype=torch.float32, device=device)
    # The third target row rejects candidate 4, so width=3 is a valid
    # lossless prefix for row zero. Row one still exposes its own coin at N=1.
    logits[2, 1] = 10.0
    sampling_info = SamplingBatchInfo(
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int64, device=device),
        is_all_greedy=False,
    )

    static_tokens, static_lengths = backend.verify(
        LogitsProcessorOutput(next_token_logits=logits.clone()),
        sampling_info,
        candidates,
    )
    static_tokens = static_tokens.clone()
    static_lengths = static_lengths.clone()

    executor = _compact_executor()
    executor.sampling_backend = backend
    executor._compact_spec_output_tokens_buf = torch.empty(
        (2, 4), dtype=torch.int32, device=device
    )
    executor._compact_spec_accept_lengths_buf = torch.empty(
        2, dtype=torch.int32, device=device
    )
    executor._compact_spec_logprobs_buf = torch.empty(
        (2, 4), dtype=torch.float32, device=device
    )
    compact_tokens, compact_lengths, _ = executor._verify_compact_spec_candidates(
        logits=torch.cat([logits[:3], logits[4:5]]),
        sampling_info=sampling_info,
        ctx=_compact_context(),
        candidates=candidates,
    )
    torch.cuda.synchronize()

    static_rows = static_tokens.view(2, 4)
    compact_rows = compact_tokens.view(2, 4)
    for row, accepted in enumerate(static_lengths.cpu().tolist()):
        assert compact_lengths[row].item() == accepted
        assert torch.equal(static_rows[row, :accepted], compact_rows[row, :accepted])
