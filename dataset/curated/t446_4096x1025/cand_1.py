import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 446
M, D, DT = 4096, 1025, torch.bfloat16


@triton.jit
def _fused_gelu_rms_relu(X, W, Y, N, stride_x, stride_y,
                         EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # exact (erf) GELU in fp32, then round to bf16 to match F.gelu on bf16 input
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g_bf = g.to(tl.bfloat16)
    gf = g_bf.to(tl.float32)

    # RMS norm in fp32
    ms = tl.sum(gf * gf, axis=0) / N
    inv = tl.math.rsqrt(ms + EPS)
    y = gf * inv

    # cast to bf16 (matches .to(x.dtype)), then multiply by weight in fp32 opmath
    y_bf = y.to(tl.bfloat16)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    out = y_bf.to(tl.float32) * w

    # relu
    out = tl.maximum(out, 0.0)

    tl.store(Y + row * stride_y + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            xg = F.gelu(x)
            _xf = xg.float()
            xg = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            return torch.relu(xg)

        orig_shape = x.shape
        n = orig_shape[-1]
        x2 = x.contiguous().view(-1, n)
        m = x2.shape[0]
        y = torch.empty_like(x2)
        w = self.rms1_w
        if not w.is_cuda:
            w = w.to(x.device)

        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_gelu_rms_relu[(m,)](
            x2, w, y, n,
            x2.stride(0), y.stride(0),
            EPS=1e-6, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
