import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 240
M, D, DT = 8192, 513, torch.bfloat16


@triton.jit
def _fused_ln_bias_scale_kernel(
    X, G, B, B1, OUT,
    N, stride_x, stride_o,
    eps, scale,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x_ptr = X + row * stride_x + cols
    x = tl.load(x_ptr, mask=mask, other=0.0).to(tl.float32)

    # mean
    mean = tl.sum(x, axis=0) / N
    # variance
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)

    # layer_norm computed in fp32, then rounded to bf16 (matches PyTorch output dtype)
    y = (x - mean) * rstd * g + b
    y_bf16 = y.to(tl.bfloat16)

    # x + b1 : bf16 op with fp32 opmath, round to bf16
    z = (y_bf16.to(tl.float32) + b1).to(tl.bfloat16)

    # z * 1.0007 : fp32 opmath, round to bf16
    out = (z.to(tl.float32) * scale).to(tl.bfloat16)

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2d = x.contiguous().view(-1, N)
        Mrows = x2d.shape[0]
        out = torch.empty_like(x2d)

        BLOCK_SIZE = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK_SIZE <= 1024 else 8

        _fused_ln_bias_scale_kernel[(Mrows,)](
            x2d, self.ln0_g, self.ln0_b, self.b1, out,
            N, x2d.stride(0), out.stride(0),
            1e-5, 1.0007,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
