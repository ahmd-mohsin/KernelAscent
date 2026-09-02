import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 708
M, D, DT = 512, 2049, torch.bfloat16


@triton.jit
def _gelu_rms_kernel(X, W, Y, N, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    # exact gelu in fp32, rounded to bf16 (matches F.gelu on bf16 tensor)
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)
    # rms norm in fp32
    ms = tl.sum(g * g, axis=0) / N
    r = tl.math.rsqrt(ms + 1e-6)
    n = (g * r).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    y = (n * w).to(tl.bfloat16)
    tl.store(Y + row * stride_y + offs, y, mask=mask)


@triton.jit
def _scale_gelu_kernel(X, Y, numel, scale, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    # x * scale computed in fp32, rounded to bf16 (matches torch bf16 * scalar)
    t = (x * scale).to(tl.bfloat16).to(tl.float32)
    g = 0.5 * t * (1.0 + tl.math.erf(t * 0.7071067811865476))
    y = g.to(tl.bfloat16)
    tl.store(Y + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 2048, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.W3 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM 1 (cuBLAS)
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        BLOCK = triton.next_power_of_2(N)
        # fused gelu -> rmsnorm -> weight (in-place on h)
        _gelu_rms_kernel[(Mrows,)](
            h, self.rms2_w, h, N, h.stride(0), h.stride(0),
            BLOCK=BLOCK, num_warps=8,
        )
        # GEMM 2 (cuBLAS)
        o = h @ self.W3
        o = o.contiguous()
        numel = o.numel()
        EBLOCK = 1024
        _scale_gelu_kernel[(triton.cdiv(numel, EBLOCK),)](
            o, o, numel, 1.3239, BLOCK=EBLOCK, num_warps=4,
        )
        return o
