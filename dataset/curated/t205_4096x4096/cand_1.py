import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 205
M, D, DT = 4096, 4096, torch.bfloat16


@triton.jit
def _fused_kernel(
    X, G, B, B4, Y,
    N, stride_x, stride_y, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)

    # relu (in bf16, then cast to fp32 for gelu math like PyTorch opmath)
    x = tl.maximum(x, 0.0)

    # gelu #1 (exact, erf), computed in fp32, rounded back to bf16
    xf = x.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    xf = xf * 0.5 * (1.0 + tl.math.erf(xf * INV_SQRT2))
    x = xf.to(tl.bfloat16)

    # gelu #2
    xf = x.to(tl.float32)
    xf = xf * 0.5 * (1.0 + tl.math.erf(xf * INV_SQRT2))
    x = xf.to(tl.bfloat16)

    # layernorm in fp32
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, 0.0)
    mean = tl.sum(xf, axis=0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xf - mean) * rstd * g + b
    y = y.to(tl.bfloat16)

    # add b4 (opmath fp32, round to bf16)
    b4 = tl.load(B4 + cols, mask=mask, other=0.0)
    out = (y.to(tl.float32) + b4.to(tl.float32)).to(tl.bfloat16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln3_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = torch.relu(x)
            x = F.gelu(x)
            x = F.gelu(x)
            x = F.layer_norm(x, (x.shape[-1],), self.ln3_g, self.ln3_b)
            return x + self.b4

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(rows,)](
            x2, self.ln3_g, self.ln3_b, self.b4, y,
            N, x2.stride(0), y.stride(0), 1e-5,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
