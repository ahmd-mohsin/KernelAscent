import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 710
M, D, DT = 4096, 4096, torch.bfloat16


@triton.jit
def _double_ln_kernel(
    X, Y, G0, B0, G1, B1,
    N, stride_x, stride_y,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # first layer norm (fp32 stats, like PyTorch bf16 layer_norm)
    mean0 = tl.sum(x, axis=0) / N
    d0 = tl.where(mask, x - mean0, 0.0)
    var0 = tl.sum(d0 * d0, axis=0) / N
    rstd0 = 1.0 / tl.sqrt(var0 + eps)
    g0 = tl.load(G0 + cols, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(B0 + cols, mask=mask, other=0.0).to(tl.float32)
    y0 = d0 * rstd0 * g0 + b0
    # round to bf16 as PyTorch would between the two layer_norm calls
    y0 = y0.to(tl.bfloat16).to(tl.float32)

    # second layer norm
    mean1 = tl.sum(tl.where(mask, y0, 0.0), axis=0) / N
    d1 = tl.where(mask, y0 - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / N
    rstd1 = 1.0 / tl.sqrt(var1 + eps)
    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    y1 = d1 * rstd1 * g1 + b1

    tl.store(Y + row * stride_y + cols, y1.to(tl.bfloat16), mask=mask)


@triton.jit
def _rmsnorm_kernel(
    X, Y, W,
    N, stride_x, stride_y,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / N
    r = 1.0 / tl.sqrt(ms + eps)
    xn = (x * r).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    out = (xn * w).to(tl.bfloat16)
    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.W2 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2d = x.contiguous().view(-1, N)
        Mrows = x2d.shape[0]

        y = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(N)
        _double_ln_kernel[(Mrows,)](
            x2d, y,
            self.ln0_g, self.ln0_b, self.ln1_g, self.ln1_b,
            N, x2d.stride(0), y.stride(0),
            1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )

        z = y @ self.W2

        K = z.shape[-1]
        z2d = z.view(-1, K)
        out = torch.empty_like(z2d)
        BLOCK2 = triton.next_power_of_2(K)
        _rmsnorm_kernel[(z2d.shape[0],)](
            z2d, out, self.rms3_w,
            K, z2d.stride(0), out.stride(0),
            1e-6,
            BLOCK=BLOCK2,
            num_warps=4,
        )
        return out.view(*orig_shape[:-1], K)
