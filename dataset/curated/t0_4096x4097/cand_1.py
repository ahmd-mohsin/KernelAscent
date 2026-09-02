import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 0
M, D, DT = 4096, 4097, torch.bfloat16


@triton.jit
def _fused_row_kernel(
    X, W, OUT,
    N,
    stride_x, stride_o,
    inv_n,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # exact GELU (erf-based), computed in fp32 then rounded to bf16 (matches PyTorch opmath)
    g = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # scale
    g = g * 1.2523
    g = g.to(tl.bfloat16).to(tl.float32)

    # RMSNorm (fp32 math on the bf16 values, as in reference)
    ms = tl.sum(tl.where(mask, g * g, 0.0), axis=0) * inv_n
    r = g * tl.math.rsqrt(ms + 1e-6)
    r = r.to(tl.bfloat16).to(tl.float32)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = (r * w).to(tl.bfloat16).to(tl.float32)

    # softmax 1 (fp32 accumulate, bf16 output)
    y = tl.where(mask, y, float('-inf'))
    m1 = tl.max(y, axis=0)
    e1 = tl.exp(y - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    y = (e1 / s1).to(tl.bfloat16).to(tl.float32)

    # softmax 2
    y = tl.where(mask, y, float('-inf'))
    m2 = tl.max(y, axis=0)
    e2 = tl.exp(y - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    out = (e2 / s2).to(tl.bfloat16)

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms2_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # fallback: reference implementation
            y = F.gelu(x) * 1.2523
            _yf = y.float()
            y = (_yf * torch.rsqrt(_yf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(y.dtype) * self.rms2_w
            y = torch.softmax(y, dim=-1)
            y = torch.softmax(y, dim=-1)
            return y

        orig_shape = x.shape
        n = orig_shape[-1]
        x2 = x.contiguous().view(-1, n)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(n)
        num_warps = 8
        if BLOCK >= 4096:
            num_warps = 16
        if BLOCK >= 16384:
            num_warps = 32

        _fused_row_kernel[(rows,)](
            x2, self.rms2_w, out,
            n,
            x2.stride(0), out.stride(0),
            1.0 / n,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
