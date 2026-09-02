import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 830
M, D, DT = 512, 1025, torch.bfloat16


@triton.jit
def _fused_kernel(
    x_ptr, g0_ptr, b0_ptr, g3_ptr, b3_ptr, out_ptr,
    D, stride_x, stride_o,
    scale, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    g0 = tl.load(g0_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(b0_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    g3 = tl.load(g3_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(b3_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    Df = D.to(tl.float32)

    # ---- LayerNorm 0 ----
    mean = tl.sum(x, axis=0) / Df
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / Df
    rstd = 1.0 / tl.sqrt(var + eps)
    y = diff * rstd * g0 + b0
    y = y.to(tl.bfloat16).to(tl.float32)  # match bf16 output rounding

    # ---- scale ----
    y = y * scale
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- softmax 1 ----
    y_m = tl.where(mask, y, float('-inf'))
    mx = tl.max(y_m, axis=0)
    e = tl.exp(y_m - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm 3 ----
    mean2 = tl.sum(tl.where(mask, y, 0.0), axis=0) / Df
    diff2 = tl.where(mask, y - mean2, 0.0)
    var2 = tl.sum(diff2 * diff2, axis=0) / Df
    rstd2 = 1.0 / tl.sqrt(var2 + eps)
    z = diff2 * rstd2 * g3 + b3
    z = z.to(tl.bfloat16).to(tl.float32)

    # ---- softmax 2 ----
    z_m = tl.where(mask, z, float('-inf'))
    mx2 = tl.max(z_m, axis=0)
    e2 = tl.exp(z_m - mx2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    out = e2 / s2

    tl.store(out_ptr + row * stride_o + cols, out.to(tl.bfloat16), mask=mask)


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
        _fused_kernel[(m,)](
            x2, self.ln0_g, self.ln0_b, self.ln3_g, self.ln3_b, out,
            d, x2.stride(0), out.stride(0),
            1.4241, 1e-5,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)
