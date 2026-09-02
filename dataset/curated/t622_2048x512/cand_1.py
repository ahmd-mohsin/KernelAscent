import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 622
M, D, DT = 2048, 512, torch.bfloat16


@triton.jit
def _fused_gelu_rms_softmax_kernel(
    X_ptr, W_ptr, Out_ptr,
    stride_xm,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)

    # exact (erf-based) GELU in fp32, then round to bf16 to match F.gelu output dtype
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(tl.where(mask, g * g, 0.0), axis=0) / N
    rs = tl.math.rsqrt(ms + 1e-6)
    xn = (g * rs).to(tl.bfloat16).to(tl.float32)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    h = (xn * w).to(tl.bfloat16).to(tl.float32)

    # softmax in fp32 (matches PyTorch's fp32 accumulation for bf16 softmax)
    h_masked = tl.where(mask, h, float('-inf'))
    m = tl.max(h_masked, axis=0)
    e = tl.exp(h_masked - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.bfloat16).to(tl.float32)

    out = (sm * 1.0258).to(tl.bfloat16)
    tl.store(Out_ptr + row * stride_xm + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 matmul (fp32 accumulate)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_gelu_rms_softmax_kernel[(Mrows,)](
            h, self.rms2_w, out,
            h.stride(0),
            N=N,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
