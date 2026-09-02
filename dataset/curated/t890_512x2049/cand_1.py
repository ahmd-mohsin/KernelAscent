import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 890
M, D, DT = 512, 2049, torch.bfloat16


@triton.jit
def _rms_softmax_kernel(
    X_ptr, W_ptr, Out_ptr,
    N, stride_x, stride_o,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMSNorm (mean over N in fp32)
    ms = tl.sum(xf * xf, axis=0) / N
    r = tl.math.rsqrt(ms + eps)

    # cast normalized value to bf16 (matching .to(x.dtype)), then bf16*bf16 mul
    xn_bf = (xf * r).to(tl.bfloat16)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    y_bf = (xn_bf.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    # softmax in fp32 on the bf16 values
    yf = y_bf.to(tl.float32)
    yf = tl.where(mask, yf, float('-inf'))
    m = tl.max(yf, axis=0)
    e = tl.exp(yf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)

    tl.store(Out_ptr + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 2048, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # (M, 2048) bf16, uses tensor cores
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _rms_softmax_kernel[(Mrows,)](
            x, self.rms1_w, out,
            N, x.stride(0), out.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
