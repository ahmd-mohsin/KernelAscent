import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 538
M, D, DT = 2048, 2049, torch.bfloat16


@triton.jit
def _bias_rmsnorm_kernel(
    X_ptr, B_ptr, W_ptr, Out_ptr,
    N, stride_x, stride_o,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0)

    # x + b1: PyTorch computes in fp32 opmath, rounds to bf16
    xb = (x.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)

    # RMS norm in fp32 on the bf16-rounded values
    xf = xb.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / N
    rs = tl.math.rsqrt(ms + eps)

    # (xf * rs).to(bf16) * w  (mul in fp32 opmath, round to bf16)
    normed = (xf * rs).to(tl.bfloat16)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    out = (normed.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    tl.store(Out_ptr + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 2048, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = torch.matmul(x, self.W0)
        Mrows, N = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(N)
        _bias_rmsnorm_kernel[(Mrows,)](
            y, self.b1, self.rms2_w, out,
            N, y.stride(0), out.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
