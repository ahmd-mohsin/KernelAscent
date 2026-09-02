import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 140
M, D, DT = 512, 513, torch.float16


@triton.jit
def _fused_gelu2_bias_softmax(
    X_ptr, B_ptr, Out_ptr,
    N, stride_row,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # First exact-erf GELU (compute fp32, round to fp16 like PyTorch storage)
    x = x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.float16).to(tl.float32)

    # Second exact-erf GELU
    x = x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.float16).to(tl.float32)

    # Bias add (fp16 semantics: fp32 add rounded to fp16)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x = (x + b).to(tl.float16).to(tl.float32)

    # Softmax in fp32 (matches PyTorch's internal upcast for half)
    x = tl.where(mask, x, float('-inf'))
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    y = e / denom

    tl.store(Out_ptr + row * stride_row + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 1024, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            h = x @ self.W0
            h = F.gelu(h)
            h = F.gelu(h)
            h = h + self.b3
            return torch.softmax(h, dim=-1)

        # Tensor-core GEMM via cuBLAS
        h = torch.matmul(x, self.W0)

        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 1024 else 4
        _fused_gelu2_bias_softmax[(m,)](
            h, self.b3, out,
            n, h.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
