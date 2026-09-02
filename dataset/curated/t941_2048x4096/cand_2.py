import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 941
M, D, DT = 2048, 4096, torch.bfloat16


@triton.jit
def _ln_relu_kernel(
    X, G, B, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = (x - mean) * rstd * g + b
    y = tl.maximum(y, 0.0)

    tl.store(Y + row * stride_y + cols, y.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.W2 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        n = orig_shape[-1]
        x2 = x.reshape(-1, n)
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        m = x2.shape[0]

        h = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _ln_relu_kernel[(m,)](
            x2, self.ln0_g, self.ln0_b, h,
            x2.stride(0), h.stride(0),
            N=n, EPS=1e-5, BLOCK=BLOCK,
            num_warps=num_warps,
        )

        out = h @ self.W2
        out = out * 1.007
        return out.reshape(*orig_shape[:-1], self.W2.shape[1])
