import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 432
M, D, DT = 8192, 4097, torch.bfloat16


@triton.jit
def _fused_row_kernel(
    X_ptr, OUT_ptr,
    B2_ptr, G3_ptr, B3_ptr, G4_ptr, B4_ptr,
    N,  # row length
    stride_x, stride_o,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # relu
    x = tl.maximum(x, 0.0)

    # softmax (fp32 accumulation, like PyTorch)
    x_for_max = tl.where(mask, x, float('-inf'))
    row_max = tl.max(x_for_max, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    p = e / denom
    # round to bf16 (matches PyTorch storing bf16 output)
    p = p.to(tl.bfloat16).to(tl.float32)

    # add bias b2 (bf16 add -> round to bf16)
    b2 = tl.load(B2_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (p + b2).to(tl.bfloat16).to(tl.float32)

    n_f = N.to(tl.float32)

    # layer norm 3 (fp32 internal math, bf16 output)
    y_masked = tl.where(mask, y, 0.0)
    mean1 = tl.sum(y_masked, axis=0) / n_f
    d1 = tl.where(mask, y - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / n_f
    rstd1 = 1.0 / tl.sqrt(var1 + EPS)
    g3 = tl.load(G3_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    z = (y - mean1) * rstd1 * g3 + b3
    z = z.to(tl.bfloat16).to(tl.float32)

    # layer norm 4
    z_masked = tl.where(mask, z, 0.0)
    mean2 = tl.sum(z_masked, axis=0) / n_f
    d2 = tl.where(mask, z - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / n_f
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)
    g4 = tl.load(G4_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    w = (z - mean2) * rstd2 * g4 + b4

    tl.store(OUT_ptr + row * stride_o + offs, w.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b2 = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: reference path
            x = torch.relu(x)
            x = torch.softmax(x, dim=-1)
            x = x + self.b2
            x = F.layer_norm(x, (x.shape[-1],), self.ln3_g, self.ln3_b)
            x = F.layer_norm(x, (x.shape[-1],), self.ln4_g, self.ln4_b)
            return x

        orig_shape = x.shape
        n = orig_shape[-1]
        x2 = x.contiguous().view(-1, n)
        m = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(n)
        num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _fused_row_kernel[(m,)](
            x2, out,
            self.b2, self.ln3_g, self.ln3_b, self.ln4_g, self.ln4_b,
            n,
            x2.stride(0), out.stride(0),
            EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
