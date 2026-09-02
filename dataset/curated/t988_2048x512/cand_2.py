import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 988
M, D, DT = 2048, 512, torch.float16


@triton.jit
def _rms_softmax_kernel(
    X_ptr, W_ptr, Out_ptr,
    N, stride_xm, stride_om,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0)
    x32 = x.to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(x32 * x32, axis=0) / N
    r = 1.0 / tl.sqrt(ms + eps)
    xn = (x32 * r).to(tl.float16)  # cast to fp16 as in reference

    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    y = xn * w  # fp16 multiply as in reference
    y32 = y.to(tl.float32)

    # softmax in fp32 (matches PyTorch's fp32 accumulation for half inputs)
    y32 = tl.where(mask, y32, float("-inf"))
    m = tl.max(y32, axis=0)
    e = tl.exp(y32 - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(Out_ptr + row * stride_om + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK_N >= 2048 else 4
        _rms_softmax_kernel[(Mrows,)](
            x, self.rms1_w, out,
            N, x.stride(0), out.stride(0),
            1e-6,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return out
