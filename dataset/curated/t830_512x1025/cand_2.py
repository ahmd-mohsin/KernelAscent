import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 830
M, D, DT = 512, 1025, torch.bfloat16


@triton.jit
def _fused_ln_softmax_kernel(
    x_ptr, g0_ptr, b0_ptr, g3_ptr, b3_ptr, out_ptr,
    D, stride_x, stride_o,
    SCALE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm 0 (fp32 accumulation, matching PyTorch) ----
    mean = tl.sum(x, axis=0) / D
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + EPS)
    g0 = tl.load(g0_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(b0_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * g0 + b0
    # round to bf16 like the reference (output of layer_norm is bf16)
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- scale ----
    y = y * SCALE
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- Softmax 1 ----
    y_neg = tl.where(mask, y, float('-inf'))
    m1 = tl.max(y_neg, axis=0)
    e1 = tl.exp(y - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    p = e1 / s1
    p = p.to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm 3 ----
    p_masked = tl.where(mask, p, 0.0)
    mean2 = tl.sum(p_masked, axis=0) / D
    pc = tl.where(mask, p - mean2, 0.0)
    var2 = tl.sum(pc * pc, axis=0) / D
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)
    g3 = tl.load(g3_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(b3_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    z = pc * rstd2 * g3 + b3
    z = z.to(tl.bfloat16).to(tl.float32)

    # ---- Softmax 2 ----
    z_neg = tl.where(mask, z, float('-inf'))
    m2 = tl.max(z_neg, axis=0)
    e2 = tl.exp(z - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    o = e2 / s2

    tl.store(out_ptr + row * stride_o + offs, o.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            y = y * 1.4241
            y = torch.softmax(y, dim=-1)
            y = F.layer_norm(y, (y.shape[-1],), self.ln3_g, self.ln3_b)
            return torch.softmax(y, dim=-1)

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_ln_softmax_kernel[(m,)](
            x2, self.ln0_g, self.ln0_b, self.ln3_g, self.ln3_b, out,
            d, x2.stride(0), out.stride(0),
            SCALE=1.4241, EPS=1e-5, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
