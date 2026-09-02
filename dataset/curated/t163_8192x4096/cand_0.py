import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 163
M, D, DT = 8192, 4096, torch.float16


@triton.jit
def _fused_ln_kernel(
    X, G, B, Y,
    stride_x, stride_y,
    N,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)  # fp16

    # replicate: x = x * 1.1762 (fp16 store), x = x * 1.0706 (fp16 store)
    x = (x.to(tl.float32) * 1.1762).to(tl.float16)
    x = (x.to(tl.float32) * 1.0706).to(tl.float16)

    xf = x.to(tl.float32)

    # layer norm statistics in fp32
    mean = tl.sum(tl.where(mask, xf, 0.0), axis=0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = (xf - mean) * rstd * g + b
    y16 = y.to(tl.float16)

    # relu in fp16
    y16 = tl.maximum(y16, tl.zeros_like(y16))

    # final scale: fp16 tensor * python float -> compute fp32, store fp16
    out = (y16.to(tl.float32) * 1.2325).to(tl.float16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln2_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_ln_kernel[(Mrows,)](
            x, self.ln2_g, self.ln2_b, y,
            x.stride(0), y.stride(0),
            N, 1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
