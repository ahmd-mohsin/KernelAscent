import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 143
M, D, DT = 1024, 513, torch.float16


@triton.jit
def _softmax_scale_bias_kernel(
    X, B, Y,
    stride_xm, stride_ym,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=-float('inf')).to(tl.float32)

    row_max = tl.max(x, axis=0)
    x = x - row_max
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    sm = num / den

    # match PyTorch rounding: softmax -> fp16, then scale in fp32 -> fp16,
    # then bias add in fp32 -> fp16
    sm16 = sm.to(tl.float16)
    scaled16 = (sm16.to(tl.float32) * 1.2932).to(tl.float16)
    b = tl.load(B + cols, mask=mask, other=0.0)
    out16 = (scaled16.to(tl.float32) + b.to(tl.float32)).to(tl.float16)

    tl.store(Y + row * stride_ym + cols, out16, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 4096, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores)
        h = h.contiguous()
        Mr, N = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _softmax_scale_bias_kernel[(Mr,)](
            h, self.b3, y,
            h.stride(0), y.stride(0),
            N=N, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
