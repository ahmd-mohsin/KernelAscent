import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 734
M, D, DT = 2048, 2048, torch.bfloat16


@triton.jit
def _ln_relu_kernel(
    X, G, B, Y,
    N, eps,
    stride_xm, stride_ym,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = xc * rstd * g + b
    y = tl.maximum(y, 0.0)

    tl.store(Y + row * stride_ym + cols, y.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.W3 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        if x.is_cuda:
            orig_shape = x.shape
            N = orig_shape[-1]
            x2 = x.reshape(-1, N)
            if not x2.is_contiguous():
                x2 = x2.contiguous()
            Mrows = x2.shape[0]
            y = torch.empty_like(x2)
            BLOCK_N = triton.next_power_of_2(N)
            num_warps = 8 if BLOCK_N >= 2048 else 4
            _ln_relu_kernel[(Mrows,)](
                x2, self.ln1_g, self.ln1_b, y,
                N, 1e-5,
                x2.stride(0), y.stride(0),
                BLOCK_N=BLOCK_N,
                num_warps=num_warps,
            )
            x = y.reshape(orig_shape)
        else:
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            x = torch.relu(x)
        x = x @ self.W3
        return x
