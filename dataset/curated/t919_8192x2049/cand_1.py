import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 919
M, D, DT = 8192, 2049, torch.float16


@triton.jit
def _bias_scale_softmax_kernel(
    X, B1, B2, Out,
    N, stride_x, stride_o,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0)      # fp16
    b1 = tl.load(B1 + offs, mask=mask, other=0.0)                     # fp16
    b2 = tl.load(B2 + offs, mask=mask, other=0.0)                     # fp16

    # fp16 adds (IEEE half rounding, matches PyTorch elementwise add of half tensors)
    x = x + b1
    x = x + b2

    # scale in fp32 then round to fp16 (matches PyTorch opmath for half * scalar)
    xf = x.to(tl.float32) * SCALE
    xh = xf.to(tl.float16).to(tl.float32)

    xh = tl.where(mask, xh, float("-inf"))

    # softmax in fp32 accumulation (matches PyTorch half softmax acc type)
    m = tl.max(xh, axis=0)
    e = tl.math.exp(xh - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = (e / s).to(tl.float16)

    tl.store(Out + row * stride_o + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 1024, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x @ self.W0
            x = x + self.b1
            x = x + self.b2
            x = x * 1.4531
            return torch.softmax(x, dim=-1)

        # cuBLAS fp16 GEMM (same op as reference)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()

        Mrows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8

        _bias_scale_softmax_kernel[(Mrows,)](
            h, self.b1, self.b2, out,
            N, h.stride(0), out.stride(0),
            SCALE=1.4531,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
