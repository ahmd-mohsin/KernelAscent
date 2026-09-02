import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 916
M, D, DT = 8192, 1024, torch.float16


@triton.jit
def _fused_ln_relu_rms_gelu(
    X, G, B, W, Out,
    stride_row,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    ptr = X + row * stride_row + offs

    x = tl.load(ptr).to(tl.float32)

    # LayerNorm (fp32 math, eps=1e-5)
    mean = tl.sum(x, axis=0) / N
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G + offs).to(tl.float32)
    b = tl.load(B + offs).to(tl.float32)
    y = xc * rstd * g + b
    # cast to fp16 to match layer_norm output dtype
    y = y.to(tl.float16)

    # ReLU
    y = tl.maximum(y, 0.0)

    # RMSNorm: fp32 compute, cast to fp16, multiply by weight
    xf = y.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    w = tl.load(W + offs).to(tl.float32)
    z = (xf * r).to(tl.float16).to(tl.float32) * w
    z = z.to(tl.float16).to(tl.float32)

    # GELU (exact, erf) in fp32
    out = 0.5 * z * (1.0 + tl.math.erf(z * 0.7071067811865476))

    tl.store(Out + row * stride_row + offs, out.to(tl.float16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 2048, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores)
        y = x @ self.W0
        y = y.contiguous()
        m, n = y.shape
        out = torch.empty_like(y)
        _fused_ln_relu_rms_gelu[(m,)](
            y, self.ln1_g, self.ln1_b, self.rms3_w, out,
            y.stride(0),
            N=n,
            BLOCK=n,
            num_warps=8,
        )
        return out
