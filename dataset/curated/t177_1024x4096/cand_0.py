import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 177
M, D, DT = 1024, 4096, torch.float16


@triton.jit
def _fused_kernel(
    X_ptr, W_ptr, G_ptr, B_ptr, Out_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # relu
    xf = tl.maximum(xf, 0.0)

    # rmsnorm (fp32), cast to fp16, multiply by weight in fp16
    ms = tl.sum(xf * xf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    y16 = (xf * r).to(tl.float16)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0)
    y16 = y16 * w  # fp16 multiply, matches reference

    # layernorm in fp32 (as PyTorch does for half inputs), output fp16
    yf = y16.to(tl.float32)
    mean = tl.sum(tl.where(mask, yf, 0.0), axis=0) / N
    diff = tl.where(mask, yf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    z16 = ((yf - mean) * inv * g + b).to(tl.float16)

    # relu (fp16)
    z16 = tl.maximum(z16, tl.full((1,), 0.0, tl.float16))

    # softmax in fp32, output fp16
    zf = z16.to(tl.float32)
    zf = tl.where(mask, zf, float('-inf'))
    m = tl.max(zf, axis=0)
    e = tl.exp(zf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(Out_ptr + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 2048, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_kernel[(m,)](
            h, self.rms2_w, self.ln3_g, self.ln3_b, out,
            N=n, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
