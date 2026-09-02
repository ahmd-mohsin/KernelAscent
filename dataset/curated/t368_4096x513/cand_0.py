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
    LN1_G, LN1_B, B2, LN3_G, LN3_B,
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

    # layernorm 1 (fp32 math, like PyTorch)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g1 = tl.load(LN1_G + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(LN1_B + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * g1 + b1
    # cast to fp16 to match PyTorch intermediate precision
    y = y.to(tl.float16)

    # add b2 in fp16
    b2 = tl.load(B2 + cols, mask=mask, other=0.0)
    y = y + b2

    # layernorm 2
    y32 = tl.where(mask, y.to(tl.float32), 0.0)
    mean2 = tl.sum(y32, axis=0) / N
    yc = tl.where(mask, y32 - mean2, 0.0)
    var2 = tl.sum(yc * yc, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + eps)
    g3 = tl.load(LN3_G + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(LN3_B + cols, mask=mask, other=0.0).to(tl.float32)
    z = yc * rstd2 * g3 + b3
    z = z.to(tl.float16)

    # relu
    z = tl.maximum(z, tl.zeros_like(z))

    tl.store(OUT + row * stride_o + cols, z, mask=mask)


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
        n = orig_shape[-1]
        x2d = x.contiguous().view(-1, n)
        m = x2d.shape[0]
        out = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(n)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_kernel[(m,)](
            x2d, out,
            self.ln1_g, self.ln1_b, self.b2, self.ln3_g, self.ln3_b,
            n, x2d.stride(0), out.stride(0),
            1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
