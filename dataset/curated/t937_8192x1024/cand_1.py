import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 937
M, D, DT = 8192, 1024, torch.bfloat16


@triton.jit
def _fused_scale_bias_ln_kernel(
    X, B1, G, B, Y,
    stride_xm, stride_ym,
    N, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)

    x = x * 1.148 + b1

    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = xc * rstd * g + b
    tl.store(Y + row * stride_ym + cols, y.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x * 1.148
            x = x + self.b1
            return F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        Mrows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK_N <= 1024 else 8

        _fused_scale_bias_ln_kernel[(Mrows,)](
            x2, self.b1, self.ln2_g, self.ln2_b, y,
            x2.stride(0), y.stride(0),
            N, 1e-5,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
