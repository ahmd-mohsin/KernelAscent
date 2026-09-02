import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 836
M, D, DT = 4096, 512, torch.float16


@triton.jit
def _gelu_kernel(X, Y, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    # exact (erf-based) GELU, computed in fp32 like PyTorch's opmath
    y = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    tl.store(Y + offs, y.to(tl.float16), mask=mask)


@triton.jit
def _softmax_kernel(X, Y, N, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    out = e / s
    tl.store(Y + row * stride_y + offs, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.W2 = nn.Parameter((torch.randn(512, 1024, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)

    def forward(self, x):
        # First GEMM via cuBLAS (tensor cores)
        h = torch.mm(x, self.W0)

        # Fused exact-GELU in Triton (fp32 math, fp16 storage), in-place
        n = h.numel()
        BLOCK = 1024
        grid = (triton.cdiv(n, BLOCK),)
        _gelu_kernel[grid](h, h, n, BLOCK=BLOCK, num_warps=4)

        # Second GEMM via cuBLAS
        z = torch.mm(h, self.W2)

        # Fused row-wise softmax in Triton (fp32 accumulation), in-place
        Mrows, N = z.shape
        BLOCK_N = triton.next_power_of_2(N)
        _softmax_kernel[(Mrows,)](
            z, z, N, z.stride(0), z.stride(0),
            BLOCK=BLOCK_N, num_warps=8,
        )
        return z
