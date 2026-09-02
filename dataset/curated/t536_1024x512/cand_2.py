import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 536
M, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _fused_kernel(
    X, Y, B1, B2, G, B,
    stride_xm, stride_ym,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)  # bf16
    b1 = tl.load(B1 + cols, mask=mask, other=0.0)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0)

    # elementwise ops: fp32 opmath, round to bf16 each step (matches PyTorch)
    x = (x.to(tl.float32) * SCALE).to(tl.bfloat16)
    x = (x.to(tl.float32) + b1.to(tl.float32)).to(tl.bfloat16)
    x = (x.to(tl.float32) + b2.to(tl.float32)).to(tl.bfloat16)

    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, 0.0)

    # layernorm in fp32
    mean = tl.sum(xf, axis=0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (diff * rstd) * g + b
    y = y.to(tl.bfloat16)  # layer_norm output is bf16

    # softmax in fp32 on the bf16 values
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, float('-inf'))
    m = tl.max(yf, axis=0)
    e = tl.exp(yf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)

    tl.store(Y + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x * 1.2195
            x = x + self.b1
            x = x + self.b2
            x = F.layer_norm(x, (x.shape[-1],), self.ln3_g, self.ln3_b)
            return torch.softmax(x, dim=-1)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        Mrows = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        _fused_kernel[(Mrows,)](
            x2, y, self.b1, self.b2, self.ln3_g, self.ln3_b,
            x2.stride(0), y.stride(0),
            N=N, BLOCK=BLOCK, EPS=1e-5, SCALE=1.2195,
            num_warps=4,
        )
        return y.view(orig_shape)
