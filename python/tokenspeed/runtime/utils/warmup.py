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

import numpy as np
import tqdm

from tokenspeed.runtime.engine.async_llm import AsyncLLM
from tokenspeed.runtime.engine.io_struct import GenerateReqInput
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

_warmup_registry = {}


def warmup(name: str) -> callable:
    def decorator(fn: callable):
        _warmup_registry[name] = fn
        return fn

    return decorator


async def execute_warmups(warmup_names: list[str], tokenizer_manager: AsyncLLM):
    for warmup_name in warmup_names:
        if warmup_name not in _warmup_registry:
            logger.warning("Could not find custom warmup %s", warmup_name)
            continue
        logger.info("Running warmup %s", warmup_name)
        await _warmup_registry[warmup_name](tokenizer_manager)


@warmup("voice_chat")
async def voice_chat(tokenizer_manager: AsyncLLM):
    # Warm up and cache the fused_moe Triton kernels. Without this warmup,
    # real-time voice-chat inference can fail.
    for i in tqdm.trange(1, 512):
        size = i * 4
        generate_req_input = GenerateReqInput(
            input_ids=(np.random.randint(2**16, size=[size])).tolist(),
            sampling_params={
                "max_new_tokens": 30,
                "temperature": 0.8,
                "stop_token_ids": [1],
                "min_p": 0.0,
            },
        )
        await tokenizer_manager.generate_request(generate_req_input).__anext__()
