import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 67
M, D, DT = 512, 512, torch.bfloat16


@triton.jit
def _relu_scale_kernel(X, Y, n_elements, scale, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    x = tl.maximum(x, 0.0)
    x = x * scale
    tl.store(Y + offs, x.to(tl.bfloat16), mask=mask)


@triton.jit
def _bias_softmax_kernel(X, B, Y, N, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    # match PyTorch: add computed in fp32, stored as bf16, then softmax in fp32
    z = (x + b).to(tl.bfloat16).to(tl.float32)
    z = tl.where(mask, z, float('-inf'))
    m = tl.max(z, axis=0)
    e = tl.exp(z - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s
    tl.store(Y + row * stride_y + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.W4 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b5 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM 1 (cuBLAS, TF32/bf16 tensor cores)
        h = x @ self.W0

        # Fused relu + scale (second relu is a no-op since scale > 0)
        h = h.contiguous()
        n = h.numel()
        BLOCK = 1024
        grid = (triton.cdiv(n, BLOCK),)
        _relu_scale_kernel[grid](h, h, n, 1.3219, BLOCK=BLOCK)

        # GEMM 2 (cuBLAS)
        z = h @ self.W4

        # Fused bias add + softmax
        z = z.contiguous()
        Mrows, N = z.shape
        out = torch.empty_like(z)
        _bias_softmax_kernel[(Mrows,)](
            z, self.b5, out, N, z.stride(0), out.stride(0),
            BLOCK=triton.next_power_of_2(N),
        )
        return out
