// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

// The inline PTX e2m1x2 conversion used by the TensorRT-LLM communication
// kernels is a datacenter Blackwell feature. CUDA reports RTX Blackwell as
// __CUDA_ARCH__ == 1200, but ptxas rejects that instruction for sm_120.
#if defined(__CUDA_ARCH__) && \
    ((__CUDA_ARCH__ == 1000) || (__CUDA_ARCH__ == 1030))
#define TOKENSPEED_HAS_DATACENTER_BLACKWELL_FP4_CVT 1
#else
#define TOKENSPEED_HAS_DATACENTER_BLACKWELL_FP4_CVT 0
#endif
