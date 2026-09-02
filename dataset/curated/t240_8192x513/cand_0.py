import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 240
M, D, DT = 8192, 513, torch.bfloat16


@triton.jit
def _fused_ln_bias_scale(
    X, G, B, B1, Y,
    N, stride_x, stride_y,
    eps, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # mean
    mean = tl.sum(x, axis=0) / N
    # variance (biased)
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)

    # layer_norm output, rounded to bf16 (match PyTorch op boundary)
    y = (diff * rstd) * g + b
    y = y.to(tl.bfloat16).to(tl.float32)
    # add bias, round to bf16
    y = y + b1
    y = y.to(tl.bfloat16).to(tl.float32)
    # scale, round to bf16
    y = y * scale
    tl.store(Y + row * stride_y + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            y = y + self.b1
            return y * 1.0007

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_ln_bias_scale[(rows,)](
            x2, self.ln0_g, self.ln0_b, self.b1, y,
            N, x2.stride(0), y.stride(0),
            1e-5, 1.0007,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
