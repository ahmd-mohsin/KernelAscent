import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 975
M, D, DT = 4096, 512, torch.float16


@triton.jit
def _fused_kernel(
    x_ptr, b0_ptr, g1_ptr, b1_ptr, g2_ptr, b2_ptr, out_ptr,
    N, stride_x, stride_o,
    EPS: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(b0_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x = x + b0

    # LayerNorm 1
    mean1 = tl.sum(x, axis=0) / N
    d1 = tl.where(mask, x - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / N
    rstd1 = 1.0 / tl.sqrt(var1 + EPS)
    g1 = tl.load(g1_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    bb1 = tl.load(b1_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = d1 * rstd1 * g1 + bb1

    # LayerNorm 2
    mean2 = tl.sum(tl.where(mask, y, 0.0), axis=0) / N
    d2 = tl.where(mask, y - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)
    g2 = tl.load(g2_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    bb2 = tl.load(b2_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    z = d2 * rstd2 * g2 + bb2

    # Softmax
    z = tl.where(mask, z, float('-inf'))
    zmax = tl.max(z, axis=0)
    e = tl.exp(z - zmax)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(out_ptr + row * stride_o + cols, out.to(out_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = x + self.b0
            y = F.layer_norm(y, (y.shape[-1],), self.ln1_g, self.ln1_b)
            y = F.layer_norm(y, (y.shape[-1],), self.ln2_g, self.ln2_b)
            return torch.softmax(y, dim=-1)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_kernel[(rows,)](
            x2, self.b0, self.ln1_g, self.ln1_b, self.ln2_g, self.ln2_b, out,
            N, x2.stride(0), out.stride(0),
            EPS=1e-5, BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)
