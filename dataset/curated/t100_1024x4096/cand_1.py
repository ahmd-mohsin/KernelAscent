import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100
M, D, DT = 1024, 4096, torch.float16


@triton.jit
def _fused_kernel(
    X, W, B2, G, B3, Y,
    N, stride_x, stride_y,
    eps_rms, eps_ln, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm in fp32, round to fp16
    ms = tl.sum(x * x, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps_rms)
    xh = (x * inv).to(tl.float16)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    # fp16 * fp16 computed in fp32, rounded to fp16 (matches PyTorch opmath)
    xh = (xh.to(tl.float32) * w).to(tl.float16)
    # scalar multiply
    xh = (xh.to(tl.float32) * scale).to(tl.float16)
    # add bias
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    xh = (xh.to(tl.float32) + b2).to(tl.float16)

    # LayerNorm in fp32
    xf = xh.to(tl.float32)
    mean = tl.sum(tl.where(mask, xf, 0.0), axis=0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps_ln)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xf - mean) * rstd * g + b3
    y = tl.maximum(y, 0.0).to(tl.float16)

    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        Mrows = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(Mrows,)](
            x2, self.rms0_w, self.b2, self.ln3_g, self.ln3_b, y,
            N, x2.stride(0), y.stride(0),
            1e-6, 1e-5, 1.0401,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
