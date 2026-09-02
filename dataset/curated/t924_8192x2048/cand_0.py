import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 924
M, D, DT = 8192, 2048, torch.bfloat16


@triton.jit
def _fused_ln3_kernel(
    x_ptr, out_ptr,
    g0_ptr, b0_ptr, g1_ptr, b1_ptr, g2_ptr, b2_ptr, b3_ptr,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- LN 0 ----
    mean = tl.sum(x, axis=0) / N
    xm = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xm * xm, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(g0_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b0_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = xm * rstd * g + b
    x = y.to(tl.bfloat16).to(tl.float32)

    # ---- LN 1 ----
    mean = tl.sum(x, axis=0) / N
    xm = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xm * xm, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(g1_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b1_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = xm * rstd * g + b
    x = y.to(tl.bfloat16).to(tl.float32)

    # ---- LN 2 ----
    mean = tl.sum(x, axis=0) / N
    xm = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xm * xm, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(g2_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b2_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (xm * rstd * g + b).to(tl.bfloat16)

    # ---- + b3 (bf16 add, like eager) ----
    b3 = tl.load(b3_ptr + offs, mask=mask, other=0.0)
    out = y + b3

    tl.store(out_ptr + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            y = F.layer_norm(y, (y.shape[-1],), self.ln1_g, self.ln1_b)
            y = F.layer_norm(y, (y.shape[-1],), self.ln2_g, self.ln2_b)
            return y + self.b3

        orig_shape = x.shape
        N = orig_shape[-1]
        x2d = x.contiguous().view(-1, N)
        rows = x2d.shape[0]
        out = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_ln3_kernel[(rows,)](
            x2d, out,
            self.ln0_g, self.ln0_b,
            self.ln1_g, self.ln1_b,
            self.ln2_g, self.ln2_b,
            self.b3,
            N, 1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
