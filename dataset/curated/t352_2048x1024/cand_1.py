import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 352
M, D, DT = 2048, 1024, torch.bfloat16


@triton.jit
def _fused_ln_kernel(
    X, W, B, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    PRE_SCALE: tl.constexpr,
    POST_SCALE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    # x * 1.106 computed like PyTorch: fp32 mul, round back to bf16
    xs_bf16 = (x.to(tl.float32) * PRE_SCALE).to(tl.bfloat16)
    xf = xs_bf16.to(tl.float32)

    # layer norm in fp32 (as PyTorch does for bf16 inputs)
    mean = tl.sum(tl.where(mask, xf, 0.0), axis=0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = (xf - mean) * rstd * w + b
    y_bf16 = y.to(tl.bfloat16)
    # final scale: bf16 -> fp32 mul -> bf16 (matches PyTorch semantics)
    out = (y_bf16.to(tl.float32) * POST_SCALE).to(tl.bfloat16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2d = x.contiguous().view(-1, N)
        Mrows = x2d.shape[0]
        y = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 1024 else 4

        _fused_ln_kernel[(Mrows,)](
            x2d, self.ln1_g, self.ln1_b, y,
            x2d.stride(0), y.stride(0),
            N,
            1.106, 1.0831, 1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
