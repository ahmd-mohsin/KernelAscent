import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 882
M, D, DT = 512, 2049, torch.bfloat16


@triton.jit
def _fused_kernel(
    x_ptr, w_ptr, out_ptr,
    N, stride_row,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=0.0).to(tl.float32)

    # GELU (erf-based, exact), cast to bf16 like reference
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    g = g.to(tl.bfloat16).to(tl.float32)

    # Softmax 1 (fp32 accumulation, bf16 output like PyTorch)
    g_m = tl.where(mask, g, float('-inf'))
    mx = tl.max(g_m, axis=0)
    e = tl.exp(g_m - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm1 = (e / s).to(tl.bfloat16)

    # RMSNorm: fp32 mean of squares, rsqrt, cast to bf16, multiply by bf16 weight
    xf = sm1.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / N
    rms = tl.math.rsqrt(ms + 1e-6)
    normed = (xf * rms).to(tl.bfloat16)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.bfloat16)
    y = normed * w  # bf16 multiply

    # ReLU (bf16)
    y = tl.maximum(y, tl.zeros_like(y))

    # Softmax 2
    yf = y.to(tl.float32)
    yf_m = tl.where(mask, yf, float('-inf'))
    mx2 = tl.max(yf_m, axis=0)
    e2 = tl.exp(yf_m - mx2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    out = (e2 / s2).to(tl.bfloat16)

    tl.store(out_ptr + row * stride_row + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms2_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = F.gelu(x)
            x = torch.softmax(x, dim=-1)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            x = torch.relu(x)
            return torch.softmax(x, dim=-1)

        x = x.contiguous()
        rows, N = x.shape[-2], x.shape[-1]
        x2 = x.view(-1, N)
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        _fused_kernel[(x2.shape[0],)](
            x2, self.rms2_w, out,
            N, x2.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out.view(x.shape)
