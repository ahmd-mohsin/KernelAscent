import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 123
M, D, DT = 512, 2049, torch.bfloat16


@triton.jit
def _softmax_rms_relu_kernel(
    X, W, Out,
    N, stride_x, stride_o,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax in fp32 (matches torch's internal fp32 accumulation for bf16)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s

    # round to bf16 (torch softmax returns bf16), then upcast for RMS norm
    p_bf16 = p.to(tl.bfloat16)
    pf = p_bf16.to(tl.float32)

    ms = tl.sum(tl.where(mask, pf * pf, 0.0), axis=0) / N
    inv = tl.math.rsqrt(ms + eps)

    y = (pf * inv).to(tl.bfloat16)  # cast back to bf16 as in reference

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    z = (y.to(tl.float32) * w).to(tl.bfloat16)

    zero = tl.zeros_like(z)
    z = tl.maximum(z, zero)

    tl.store(Out + row * stride_o + cols, z, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 512, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 GEMM
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _softmax_rms_relu_kernel[(Mrows,)](
            h, self.rms2_w, out,
            N, h.stride(0), out.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
