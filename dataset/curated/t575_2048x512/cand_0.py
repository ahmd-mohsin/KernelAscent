import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 575
M, D, DT = 2048, 512, torch.float16


@triton.jit
def _fused_norm_kernel(
    X, W2, W3, G, B, Out,
    stride_x, stride_o,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm 1 (compute in fp32, round through fp16 like reference)
    ms = tl.sum(x * x, axis=0) / N
    x = x * (1.0 / tl.sqrt(ms + 1e-6))
    x16 = x.to(tl.float16)
    w2 = tl.load(W2 + cols, mask=mask, other=0.0)
    x16 = x16 * w2  # fp16 multiply, matching reference

    # RMSNorm 2
    xf = x16.to(tl.float32)
    ms2 = tl.sum(xf * xf, axis=0) / N
    xf = xf * (1.0 / tl.sqrt(ms2 + 1e-6))
    x16 = xf.to(tl.float16)
    w3 = tl.load(W3 + cols, mask=mask, other=0.0)
    x16 = x16 * w3  # fp16 multiply

    # ReLU (fp16)
    x16 = tl.maximum(x16, 0.0)

    # LayerNorm (fp32 accumulation, like F.layer_norm on fp16 input)
    xf = x16.to(tl.float32)
    mean = tl.sum(xf, axis=0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv_std = 1.0 / tl.sqrt(var + 1e-5)
    y = diff * inv_std
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = y * g + b

    tl.store(Out + row * stride_o + cols, y.to(tl.float16), mask=mask)


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
        # Two GEMMs (cuBLAS tensor cores on A100)
        x = x @ self.W0
        x = x @ self.W1

        x = x.contiguous()
        rows, N = x.shape
        out = torch.empty_like(x)

        _fused_norm_kernel[(rows,)](
            x, self.rms2_w, self.rms3_w, self.ln5_g, self.ln5_b, out,
            x.stride(0), out.stride(0),
            N=N, BLOCK=triton.next_power_of_2(N),
            num_warps=8,
        )
        return out
