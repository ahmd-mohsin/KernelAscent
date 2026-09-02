import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 952
M, D, DT = 512, 512, torch.float16


@triton.jit
def _fused_epilogue_kernel(
    Y_ptr, B_ptr, OUT_ptr,
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    y = tl.load(Y_ptr + row * N + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)

    # exact GELU in fp32 (matches PyTorch's opmath upcast for half)
    yf = y.to(tl.float32)
    g = yf * 0.5 * (1.0 + tl.math.erf(yf * 0.7071067811865476))
    # relu (monotone, commutes with fp16 rounding)
    g = tl.maximum(g, 0.0)
    g16 = g.to(tl.float16)

    # bias add in fp16 arithmetic (matches half + half on CUDA)
    z16 = g16 + b

    # softmax with fp32 accumulation (matches PyTorch softmax on half)
    zf = z16.to(tl.float32)
    zf = tl.where(mask, zf, float('-inf'))
    m = tl.max(zf, axis=0)
    e = tl.exp(zf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(OUT_ptr + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = x @ self.W0
            y = F.gelu(y)
            y = torch.relu(y)
            y = y + self.b3
            return torch.softmax(y, dim=-1)

        # cuBLAS fp16 GEMM (same as reference)
        y = torch.matmul(x, self.W0)
        y = y.contiguous()
        Mrows, N = y.shape
        out = torch.empty_like(y)

        BLOCK = triton.next_power_of_2(N)
        _fused_epilogue_kernel[(Mrows,)](
            y, self.b3, out,
            N,
            BLOCK=BLOCK,
            num_warps=8 if BLOCK >= 512 else 4,
        )
        return out
