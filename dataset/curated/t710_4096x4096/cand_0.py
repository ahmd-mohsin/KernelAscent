import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 710
M, D, DT = 4096, 4096, torch.bfloat16


@triton.jit
def _double_ln_kernel(X, G0, B0, G1, B1, Y, N, EPS,
                      BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm 0 (fp32 math, matching PyTorch mixed-precision LN)
    mean0 = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean0, 0.0)
    var0 = tl.sum(xc * xc, axis=0) / N
    inv0 = tl.math.rsqrt(var0 + EPS)
    g0 = tl.load(G0 + offs, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(B0 + offs, mask=mask, other=0.0).to(tl.float32)
    y = xc * inv0 * g0 + b0

    # round to bf16 between the two LNs, exactly like the reference
    y = y.to(tl.bfloat16).to(tl.float32)

    # LayerNorm 1
    mean1 = tl.sum(tl.where(mask, y, 0.0), axis=0) / N
    yc = tl.where(mask, y - mean1, 0.0)
    var1 = tl.sum(yc * yc, axis=0) / N
    inv1 = tl.math.rsqrt(var1 + EPS)
    g1 = tl.load(G1 + offs, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + offs, mask=mask, other=0.0).to(tl.float32)
    z = yc * inv1 * g1 + b1

    tl.store(Y + row * N + offs, z.to(tl.bfloat16), mask=mask)


@triton.jit
def _rmsnorm_kernel(X, W, Y, N, EPS,
                    BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / N
    inv = tl.math.rsqrt(ms + EPS)
    # reference: (.to(bf16)) * w  -> bf16 rounding before the final multiply
    xn = (x * inv).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    out = xn * w
    tl.store(Y + row * N + offs, out.to(tl.bfloat16), mask=mask)


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
        if not x.is_cuda:
            x = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            x = x @ self.W2
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
            return x

        orig_shape = x.shape
        N = orig_shape[-1]
        xr = x.contiguous().view(-1, N)
        rows = xr.shape[0]

        y = torch.empty_like(xr)
        BLOCK = triton.next_power_of_2(N)
        _double_ln_kernel[(rows,)](
            xr, self.ln0_g, self.ln0_b, self.ln1_g, self.ln1_b, y,
            N, 1e-5, BLOCK=BLOCK, num_warps=8,
        )

        h = torch.matmul(y, self.W2)  # bf16 tensor-core GEMM
        Nh = h.shape[-1]
        out = torch.empty_like(h)
        BLOCK2 = triton.next_power_of_2(Nh)
        _rmsnorm_kernel[(rows,)](
            h, self.rms3_w, out,
            Nh, 1e-6, BLOCK=BLOCK2, num_warps=4,
        )

        return out.view(*orig_shape[:-1], Nh)
