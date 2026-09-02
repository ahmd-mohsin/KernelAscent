import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 933
M, D, DT = 8192, 1024, torch.float16


@triton.jit
def _double_ln_kernel(
    X, OUT, G0, B0, G1, B1,
    stride_x, stride_o,
    N: tl.constexpr, BLOCK: tl.constexpr, EPS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # First LayerNorm (fp32 accumulate, like PyTorch)
    mean0 = tl.sum(x, axis=0) / N
    d0 = tl.where(mask, x - mean0, 0.0)
    var0 = tl.sum(d0 * d0, axis=0) / N
    rstd0 = 1.0 / tl.sqrt(var0 + EPS)

    g0 = tl.load(G0 + cols, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(B0 + cols, mask=mask, other=0.0).to(tl.float32)
    y = d0 * rstd0 * g0 + b0

    # Match reference: intermediate stored as fp16 between the two layer_norms
    y = y.to(tl.float16).to(tl.float32)
    y = tl.where(mask, y, 0.0)

    # Second LayerNorm
    mean1 = tl.sum(y, axis=0) / N
    d1 = tl.where(mask, y - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / N
    rstd1 = 1.0 / tl.sqrt(var1 + EPS)

    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    z = d1 * rstd1 * g1 + b1

    tl.store(OUT + row * stride_o + cols, z.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            return x

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        Mrows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8

        _double_ln_kernel[(Mrows,)](
            x2, out,
            self.ln0_g, self.ln0_b, self.ln1_g, self.ln1_b,
            x2.stride(0), out.stride(0),
            N=N, BLOCK=BLOCK, EPS=1e-5,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
