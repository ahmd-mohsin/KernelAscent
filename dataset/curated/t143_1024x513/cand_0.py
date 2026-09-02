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
    X, B, Out,
    N,
    stride_xm, stride_om,
    SCALE: tl.constexpr,
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
    p = num / den

    # match reference rounding: softmax -> fp16, * scale (fp32 math) -> fp16, + bias (fp32 math) -> fp16
    p16 = p.to(tl.float16)
    y16 = (p16.to(tl.float32) * SCALE).to(tl.float16)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    z = (y16.to(tl.float32) + b).to(tl.float16)

    tl.store(Out + row * stride_om + cols, z, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 4096, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 GEMM with fp32 accumulate
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _softmax_scale_bias_kernel[(m,)](
            h, self.b3, out,
            n,
            h.stride(0), out.stride(0),
            SCALE=1.2932,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
