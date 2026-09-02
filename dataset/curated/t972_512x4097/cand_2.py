import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 972
M, D, DT = 512, 4097, torch.bfloat16


@triton.jit
def _fused_gelu_softmax_ln_kernel(
    X_ptr, G_ptr, B_ptr, Y_ptr,
    D, stride_x, stride_y,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    # emulate intermediate bf16 rounding of the reference module
    g = g.to(tl.bfloat16).to(tl.float32)

    s = g * 1.2255
    s = s.to(tl.bfloat16).to(tl.float32)

    # softmax over the row (fp32 accumulation, like PyTorch's bf16 softmax)
    s_m = tl.where(mask, s, float('-inf'))
    row_max = tl.max(s_m, axis=0)
    e = tl.exp(s_m - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    p = e / denom
    p = p.to(tl.bfloat16).to(tl.float32)

    # layer norm (fp32 internal, like PyTorch's bf16 layer_norm)
    p_masked = tl.where(mask, p, 0.0)
    mean = tl.sum(p_masked, axis=0) / D
    diff = tl.where(mask, p - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + EPS)

    gam = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    y = diff * rstd * gam + beta
    tl.store(Y_ptr + row * stride_y + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln3_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = F.gelu(x)
            x = x * 1.2255
            x = torch.softmax(x, dim=-1)
            return F.layer_norm(x, (x.shape[-1],), self.ln3_g, self.ln3_b)

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8
        if BLOCK >= 4096:
            num_warps = 16
        if BLOCK >= 16384:
            num_warps = 32

        _fused_gelu_softmax_ln_kernel[(rows,)](
            x2, self.ln3_g, self.ln3_b, y,
            d, x2.stride(0), y.stride(0),
            EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
