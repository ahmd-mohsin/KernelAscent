import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 194
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _fused_gelu3_ln_relu(
    X, Y, W, B,
    N, stride_x, stride_y,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)

    # GELU #1 (exact, fp32 math, round back to bf16 like PyTorch)
    xf = x.to(tl.float32)
    xf = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    x = xf.to(x.dtype)

    # GELU #2
    xf = x.to(tl.float32)
    xf = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    x = xf.to(x.dtype)

    # GELU #3
    xf = x.to(tl.float32)
    xf = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    x = xf.to(x.dtype)

    # LayerNorm in fp32
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, 0.0)
    mean = tl.sum(xf, axis=0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xf - mean) * rstd * w + b

    # ReLU
    y = tl.maximum(y, 0.0)

    tl.store(Y + row * stride_y + cols, y.to(x.dtype), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = F.gelu(x)
            x = F.gelu(x)
            x = F.gelu(x)
            x = F.layer_norm(x, (x.shape[-1],), self.ln3_g, self.ln3_b)
            return torch.relu(x)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        Mrows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_gelu3_ln_relu[(Mrows,)](
            x2, y, self.ln3_g, self.ln3_b,
            N, x2.stride(0), y.stride(0),
            1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
