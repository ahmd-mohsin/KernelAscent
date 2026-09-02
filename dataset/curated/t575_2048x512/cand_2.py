import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 575
M, D, DT = 2048, 512, torch.float16


@triton.jit
def _fused_norms_kernel(
    X, OUT, W2, W3, G, B,
    stride_x, stride_o,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMSNorm 1
    ms = tl.sum(xf * xf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    y = (xf * r).to(tl.float16)  # round to fp16 (matches .to(x.dtype))
    w2 = tl.load(W2 + cols, mask=mask, other=0.0)
    y = (y.to(tl.float32) * w2.to(tl.float32)).to(tl.float16)  # fp16 mul

    # RMSNorm 2
    yf = y.to(tl.float32)
    ms2 = tl.sum(yf * yf, axis=0) / N
    r2 = 1.0 / tl.sqrt(ms2 + 1e-6)
    z = (yf * r2).to(tl.float16)
    w3 = tl.load(W3 + cols, mask=mask, other=0.0)
    z = (z.to(tl.float32) * w3.to(tl.float32)).to(tl.float16)

    # ReLU (in fp16)
    z = tl.maximum(z, 0.0)

    # LayerNorm (fp32 internal, like PyTorch on fp16 input)
    zf = z.to(tl.float32)
    mean = tl.sum(tl.where(mask, zf, 0.0), axis=0) / N
    diff = tl.where(mask, zf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    out = (zf - mean) * inv * g + b

    tl.store(OUT + row * stride_o + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.W1 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln5_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln5_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # Two matmuls kept separate for numerical equivalence (fp16 intermediate)
        x = x @ self.W0
        x = x @ self.W1

        x = x.contiguous()
        Mrows, N = x.shape
        out = torch.empty_like(x)
        _fused_norms_kernel[(Mrows,)](
            x, out,
            self.rms2_w, self.rms3_w, self.ln5_g, self.ln5_b,
            x.stride(0), out.stride(0),
            N=N, BLOCK=triton.next_power_of_2(N),
            num_warps=8,
        )
        return out
