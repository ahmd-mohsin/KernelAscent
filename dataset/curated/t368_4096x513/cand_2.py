import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 368
M, D, DT = 4096, 513, torch.float16


@triton.jit
def _fused_kernel(
    X, OUT,
    LN1G, LN1B, B2, LN3G, LN3B,
    N, stride_x, stride_o,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    # relu
    x = tl.maximum(x, 0.0)

    # layernorm 1 (stats in fp32)
    mean1 = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    d1 = tl.where(mask, x - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / N
    rstd1 = 1.0 / tl.sqrt(var1 + eps)

    g1 = tl.load(LN1G + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(LN1B + cols, mask=mask, other=0.0).to(tl.float32)
    y = d1 * rstd1 * g1 + b1
    # round to fp16 as PyTorch would between ops
    y = y.to(tl.float16)

    # add b2 in fp16
    b2 = tl.load(B2 + cols, mask=mask, other=0.0)
    y = y + b2

    # layernorm 2
    z = y.to(tl.float32)
    mean2 = tl.sum(tl.where(mask, z, 0.0), axis=0) / N
    d2 = tl.where(mask, z - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + eps)

    g3 = tl.load(LN3G + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(LN3B + cols, mask=mask, other=0.0).to(tl.float32)
    out = d2 * rstd2 * g3 + b3
    out = out.to(tl.float16)

    # final relu (on fp16 values)
    zero = tl.zeros_like(out)
    out = tl.maximum(out, zero)

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = torch.relu(x)
            y = F.layer_norm(y, (y.shape[-1],), self.ln1_g, self.ln1_b)
            y = y + self.b2
            y = F.layer_norm(y, (y.shape[-1],), self.ln3_g, self.ln3_b)
            return torch.relu(y)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_kernel[(rows,)](
            x2, out,
            self.ln1_g, self.ln1_b, self.b2, self.ln3_g, self.ln3_b,
            N, x2.stride(0), out.stride(0),
            1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
