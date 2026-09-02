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
    X_ptr, W_ptr, Y_ptr,
    N, stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # --- GELU (exact, erf-based), rounded to bf16 like PyTorch elementwise op ---
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # --- scale by 1.2523, rounded to bf16 ---
    g = g * 1.2523
    g = g.to(tl.bfloat16).to(tl.float32)

    # --- RMSNorm in fp32 ---
    ms = tl.sum(tl.where(mask, g * g, 0.0), axis=0) / N
    r = g * tl.math.rsqrt(ms + 1e-6)
    r = r.to(tl.bfloat16).to(tl.float32)

    # --- multiply by weight (bf16 op, fp32 compute, bf16 rounding) ---
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    r = (r * w).to(tl.bfloat16).to(tl.float32)

    # --- softmax #1 (fp32 accumulate, bf16 output rounding) ---
    r = tl.where(mask, r, float('-inf'))
    m1 = tl.max(r, axis=0)
    e1 = tl.math.exp(r - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    o1 = (e1 / s1).to(tl.bfloat16).to(tl.float32)

    # --- softmax #2 ---
    o1 = tl.where(mask, o1, float('-inf'))
    m2 = tl.max(o1, axis=0)
    e2 = tl.math.exp(o1 - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    o2 = e2 / s2

    tl.store(Y_ptr + row * stride_y + offs, o2.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms2_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = F.gelu(x)
            x = x * 1.2523
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            x = torch.softmax(x, dim=-1)
            x = torch.softmax(x, dim=-1)
            return x

        orig_shape = x.shape
        N = orig_shape[-1]
        x2d = x.contiguous().view(-1, N)
        rows = x2d.shape[0]
        y = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8
        if BLOCK >= 4096:
            num_warps = 16
        if BLOCK >= 16384:
            num_warps = 32

        _fused_row_kernel[(rows,)](
            x2d, self.rms2_w, y,
            N, x2d.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
