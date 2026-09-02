import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 851
M, D, DT = 8192, 4096, torch.bfloat16


@triton.jit
def _relu_ln_relu_kernel(
    X, Y, G, B,
    N, eps,
    stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    # relu in input dtype (exact), then compute LN in fp32
    x = tl.maximum(x, 0.0)
    xf = x.to(tl.float32)

    mean = tl.sum(xf, axis=0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = (xf - mean) * rstd * g + b
    y = tl.maximum(y, 0.0)

    tl.store(Y + row * stride_y + cols, y.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.W4 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM 1 (cuBLAS is optimal for this shape)
        h = x @ self.W0
        h = h.contiguous()

        Mrows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 1024 else 4
        _relu_ln_relu_kernel[(Mrows,)](
            h, out, self.ln2_g, self.ln2_b,
            N, 1e-5,
            h.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )

        # GEMM 2 + scale
        y = out @ self.W4
        y.mul_(1.2434)
        return y
