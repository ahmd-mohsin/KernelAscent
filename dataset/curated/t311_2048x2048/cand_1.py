import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 311
M, D, DT = 2048, 2048, torch.bfloat16


@triton.jit
def _fused_kernel(
    x_ptr, b0_ptr, b2_ptr, g_ptr, beta_ptr, out_ptr,
    N, stride_row,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=0.0)
    b0 = tl.load(b0_ptr + cols, mask=mask, other=0.0)
    b2 = tl.load(b2_ptr + cols, mask=mask, other=0.0)

    # elementwise ops with bf16 rounding at each step (matches PyTorch semantics)
    v = (x.to(tl.float32) + b0.to(tl.float32)).to(tl.bfloat16)
    v = (v.to(tl.float32) * 1.357).to(tl.bfloat16)
    v = (v.to(tl.float32) + b2.to(tl.float32)).to(tl.bfloat16)

    vf = v.to(tl.float32)
    vf = tl.where(mask, vf, float("-inf"))

    # softmax in fp32, output rounded to bf16 (matches torch.softmax on bf16)
    mx = tl.max(vf, axis=0)
    e = tl.exp(vf - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = (e / s).to(tl.bfloat16)

    # layer_norm in fp32 on bf16 input (matches F.layer_norm on bf16)
    pf = p.to(tl.float32)
    pf = tl.where(mask, pf, 0.0)
    mean = tl.sum(pf, axis=0) / N
    diff = tl.where(mask, pf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(g_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * g + beta

    tl.store(out_ptr + row * stride_row + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = x + self.b0
            y = y * 1.357
            y = y + self.b2
            y = torch.softmax(y, dim=-1)
            return F.layer_norm(y, (y.shape[-1],), self.ln4_g, self.ln4_b)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_kernel[(rows,)](
            x2, self.b0, self.b2, self.ln4_g, self.ln4_b, out,
            N, x2.stride(0),
            EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
