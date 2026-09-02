import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 938
M, D, DT = 4096, 4096, torch.float16


@triton.jit
def _relu_ln_bias_kernel(
    X, G, B, B4, Y,
    N, eps,
    stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.maximum(x, 0.0)

    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = (x - mean) * rstd * g + b
    y16 = y.to(tl.float16)

    b4 = tl.load(B4 + cols, mask=mask, other=0.0)
    out = y16 + b4

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x2d = x.reshape(-1, x.shape[-1])
        m, n = x2d.shape
        y = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _relu_ln_bias_kernel[(m,)](
            x2d, self.ln3_g, self.ln3_b, self.b4, y,
            n, 1e-5,
            x2d.stride(0), y.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.reshape(x.shape)
