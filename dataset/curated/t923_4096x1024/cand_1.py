import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 923
M, D, DT = 4096, 1024, torch.float16


@triton.jit
def _rms_gelu_softmax_kernel(X, W, Y, N, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm (fp32 accumulate, matches reference)
    ms = tl.sum(x * x, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    xn = (x * inv).to(tl.float16)  # cast to fp16 like reference

    # weight multiply in fp16 (matches fp16*fp16 in reference)
    w = tl.load(W + offs, mask=mask, other=0.0)
    v = xn * w

    # exact (erf) GELU computed in fp32, result rounded to fp16
    vf = v.to(tl.float32)
    g = 0.5 * vf * (1.0 + tl.math.erf(vf * 0.7071067811865476))
    g16 = g.to(tl.float16)

    # softmax in fp32 (matches PyTorch half softmax which accumulates in fp32)
    gf = tl.where(mask, g16.to(tl.float32), float("-inf"))
    mmax = tl.max(gf, axis=0)
    e = tl.exp(gf - mmax)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(Y + row * stride_y + offs, out, mask=mask)


@triton.jit
def _gelu_kernel(X, Y, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    y = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    tl.store(Y + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 4096, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.W4 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM 1 (cuBLAS tensor cores)
        x = x @ self.W0

        rows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _rms_gelu_softmax_kernel[(rows,)](
            x, self.rms1_w, y, N,
            x.stride(0), y.stride(0),
            BLOCK=BLOCK, num_warps=16,
        )

        # GEMM 2
        z = y @ self.W4

        # fused final GELU (in-place buffer reuse)
        n = z.numel()
        out = torch.empty_like(z)
        grid = (triton.cdiv(n, 4096),)
        _gelu_kernel[grid](z, out, n, BLOCK=4096, num_warps=8)
        return out
