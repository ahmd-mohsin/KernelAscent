import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 320
M, D, DT = 4096, 512, torch.bfloat16


@triton.jit
def _ln_relu_kernel(
    X, G, B, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    xm = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xm * xm, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = xm * rstd * g + b
    y = tl.maximum(y, 0.0)
    tl.store(Y + row * stride_y + cols, y.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        orig_shape = h.shape
        N = orig_shape[-1]
        h2 = h.reshape(-1, N)
        if not h2.is_contiguous():
            h2 = h2.contiguous()
        rows = h2.shape[0]
        y = torch.empty_like(h2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8
        _ln_relu_kernel[(rows,)](
            h2, self.ln1_g, self.ln1_b, y,
            h2.stride(0), y.stride(0),
            N, 1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.reshape(orig_shape)
