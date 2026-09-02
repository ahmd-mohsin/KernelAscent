import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 231
M, D, DT = 512, 1024, torch.bfloat16


@triton.jit
def _rms_softmax_relu_kernel(
    X_ptr, W_ptr, Out_ptr,
    N, stride_xm, stride_om,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMSNorm in float32
    ms = tl.sum(xf * xf, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + eps)
    xn = (xf * rstd).to(tl.bfloat16)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    # PyTorch bf16 elementwise mul uses fp32 opmath, rounds to bf16
    y = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    # Softmax in float32 (matches torch's accscalar path for bf16)
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, float('-inf'))
    m = tl.max(yf, axis=0)
    e = tl.exp(yf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)

    # ReLU (identity for softmax outputs, kept for exactness)
    out = tl.maximum(out, 0.0)

    tl.store(Out_ptr + row * stride_om + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 2048, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(N)
        _rms_softmax_relu_kernel[(Mrows,)](
            x, self.rms1_w, out,
            N, x.stride(0), out.stride(0),
            1e-6,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
