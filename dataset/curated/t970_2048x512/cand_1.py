import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 970
M, D, DT = 2048, 512, torch.float16


@triton.jit
def _fused_gelu2_rms_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride + cols, mask=mask, other=0.0).to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # First GELU (computed in fp32, cast back to fp16 like PyTorch does)
    g = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    g = g.to(tl.float16).to(tl.float32)

    # Second GELU
    g2 = 0.5 * g * (1.0 + tl.math.erf(g * INV_SQRT2))
    h = g2.to(tl.float16)

    # RMSNorm in fp32 on fp16 values
    hf = h.to(tl.float32)
    hf = tl.where(mask, hf, 0.0)
    ms = tl.sum(hf * hf, axis=0) / N
    r = tl.math.rsqrt(ms + 1e-6)
    y = (hf * r).to(tl.float16)

    # weight multiply in fp16
    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    out = y * w
    tl.store(Y_ptr + row * stride + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # fallback (reference path)
            x = x @ self.W0
            x = F.gelu(x)
            x = F.gelu(x)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
            return x

        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_gelu2_rms_kernel[(Mrows,)](
            h, self.rms3_w, out,
            N, h.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
