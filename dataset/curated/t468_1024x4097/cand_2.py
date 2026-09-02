import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 468
M, D, DT = 1024, 4097, torch.float16


@triton.jit
def _fused_scale_double_ln_kernel(
    X_ptr, Y_ptr,
    G1_ptr, B1_ptr, G2_ptr, B2_ptr,
    N, stride_x, stride_y,
    SCALE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0)
    # match reference: fp16 tensor * python float -> compute in fp32, round to fp16
    xf = x.to(tl.float32) * SCALE
    xh = xf.to(tl.float16).to(tl.float32)
    xh = tl.where(mask, xh, 0.0)

    # ---- LayerNorm 1 (fp32 accumulation) ----
    mean1 = tl.sum(xh, axis=0) / N
    d1 = tl.where(mask, xh - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / N
    rstd1 = 1.0 / tl.sqrt(var1 + EPS)

    g1 = tl.load(G1_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y1 = d1 * rstd1 * g1 + b1
    # reference materializes fp16 between the two layer norms
    y1 = y1.to(tl.float16).to(tl.float32)
    y1 = tl.where(mask, y1, 0.0)

    # ---- LayerNorm 2 (fp32 accumulation) ----
    mean2 = tl.sum(y1, axis=0) / N
    d2 = tl.where(mask, y1 - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)

    g2 = tl.load(G2_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y2 = d2 * rstd2 * g2 + b2

    tl.store(Y_ptr + row * stride_y + offs, y2.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x * 1.3642
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
            return x

        orig_shape = x.shape
        N = orig_shape[-1]
        x2d = x.contiguous().view(-1, N)
        rows = x2d.shape[0]
        out = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK <= 4096 else 16

        _fused_scale_double_ln_kernel[(rows,)](
            x2d, out,
            self.ln1_g, self.ln1_b, self.ln2_g, self.ln2_b,
            N, x2d.stride(0), out.stride(0),
            SCALE=1.3642,
            EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
