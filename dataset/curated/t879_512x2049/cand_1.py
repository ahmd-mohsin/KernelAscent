import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 879
M, D, DT = 512, 2049, torch.float16


@triton.jit
def _fused_relu_rms_ln_kernel(
    X_ptr, W2_ptr, G_ptr, B_ptr, Y_ptr,
    N, EPS_RMS, EPS_LN,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # load row (fp16 matmul output), relu
    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0)
    x = tl.maximum(x, 0.0)

    # RMSNorm in fp32
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    r = tl.math.rsqrt(ms + EPS_RMS)

    # cast back to fp16, multiply by rms weight in fp16 (matches reference)
    xh = (xf * r).to(tl.float16)
    w2 = tl.load(W2_ptr + offs, mask=mask, other=0.0)
    xh = xh * w2

    # LayerNorm: stats in fp32, affine, cast to fp16
    xln = xh.to(tl.float32)
    mean = tl.sum(tl.where(mask, xln, 0.0), axis=0) / N
    d = tl.where(mask, xln - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    inv = tl.math.rsqrt(var + EPS_LN)

    g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = d * inv * g + b

    tl.store(Y_ptr + row * N + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 512, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS fp16 GEMM (tensor cores)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_relu_rms_ln_kernel[(m,)](
            h, self.rms2_w, self.ln3_g, self.ln3_b, out,
            n, 1e-6, 1e-5,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
