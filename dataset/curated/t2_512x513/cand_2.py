import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 2
M, D, DT = 512, 513, torch.float16


@triton.jit
def _fused_softmax_gelu_rms_softmax(
    X, W, Out,
    N, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # ---- softmax 1 (fp32 math, fp16 rounding to match PyTorch half softmax) ----
    m1 = tl.max(x, 0)
    e1 = tl.exp(x - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, 0)
    y = (e1 / s1).to(tl.float16).to(tl.float32)

    # ---- exact GELU (erf) in fp32, round to fp16 ----
    g = y * 0.5 * (1.0 + tl.math.erf(y * 0.7071067811865476))
    g16 = g.to(tl.float16)
    gf = g16.to(tl.float32)

    # ---- RMSNorm in fp32, round to fp16, scale by weight in fp16 ----
    ms = tl.sum(tl.where(mask, gf * gf, 0.0), 0) / N
    r = gf * tl.math.rsqrt(ms + 1e-6)
    r16 = r.to(tl.float16)
    w = tl.load(W + offs, mask=mask, other=0.0)
    z16 = r16 * w  # fp16 multiply, as in reference

    # ---- softmax 2 ----
    zf = tl.where(mask, z16.to(tl.float32), float('-inf'))
    m2 = tl.max(zf, 0)
    e2 = tl.exp(zf - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, 0)
    out = (e2 / s2).to(tl.float16)

    tl.store(Out + row * stride_o + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 1024, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # reference fallback for CPU
            x = x @ self.W0
            x = torch.softmax(x, dim=-1)
            x = F.gelu(x)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
            return torch.softmax(x, dim=-1)

        h = torch.matmul(x, self.W0)  # cuBLAS fp16 GEMM (fp32 accumulate)
        h = h.contiguous()
        rows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_softmax_gelu_rms_softmax[(rows,)](
            h, self.rms3_w, out,
            N, h.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
