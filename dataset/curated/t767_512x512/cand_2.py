import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 767
M, D, DT = 512, 512, torch.float16


@triton.jit
def _fused_softmax_rms_gelu_softmax(
    X_ptr, W_ptr, OUT_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # load matmul output (fp16) and upcast to fp32
    x = tl.load(X_ptr + row * N + offs, mask=mask, other=-float('inf')).to(tl.float32)

    # ---- softmax #1 (fp32 accumulation, fp16 output like torch) ----
    m1 = tl.max(x, 0)
    e1 = tl.exp(x - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, 0)
    x16 = (e1 / s1).to(tl.float16)

    # ---- RMSNorm: computed in fp32 on the fp16-rounded values ----
    xf = x16.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), 0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    xn16 = (xf * inv).to(tl.float16)

    # multiply by rms2_w (fp16 elementwise)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0)
    x16 = xn16 * w

    # ---- exact GELU (erf form), fp32 math, round to fp16 ----
    gf = x16.to(tl.float32)
    g = 0.5 * gf * (1.0 + tl.math.erf(gf * 0.7071067811865476))
    g16 = g.to(tl.float16)

    # ---- softmax #2 ----
    xf2 = tl.where(mask, g16.to(tl.float32), -float('inf'))
    m2 = tl.max(xf2, 0)
    e2 = tl.exp(xf2 - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, 0)
    sm2 = (e2 / s2).to(tl.float16)

    # ---- scale by 1.2619 (fp32 opmath, round to fp16) ----
    out = (sm2.to(tl.float32) * 1.2619).to(tl.float16)
    tl.store(OUT_ptr + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 1024, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS tensor-core matmul (fp16 in, fp32 accumulate)
        y = torch.matmul(x, self.W0)
        if not y.is_cuda:
            # CPU fallback: reference path
            y = torch.softmax(y, dim=-1)
            _yf = y.float()
            y = (_yf * torch.rsqrt(_yf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(y.dtype) * self.rms2_w
            y = F.gelu(y)
            y = torch.softmax(y, dim=-1)
            return y * 1.2619

        y = y.contiguous()
        rows, N = y.shape[-2] if y.dim() > 1 else 1, y.shape[-1]
        y2d = y.view(-1, N)
        rows = y2d.shape[0]
        out = torch.empty_like(y2d)
        BLOCK = triton.next_power_of_2(N)
        _fused_softmax_rms_gelu_softmax[(rows,)](
            y2d, self.rms2_w, out,
            N=N, BLOCK=BLOCK,
            num_warps=8,
        )
        return out.view(y.shape)
