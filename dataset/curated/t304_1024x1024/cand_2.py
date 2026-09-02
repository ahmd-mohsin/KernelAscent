import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 304
M, D, DT = 1024, 1024, torch.bfloat16


@triton.jit
def _gelu_bias_softmax_kernel(
    X_ptr, B_ptr, Out_ptr,
    N, stride_row,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)

    # exact (erf-based) GELU computed in fp32, rounded to bf16 (matches PyTorch opmath)
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # bias add in fp32, rounded back to bf16 (matches bf16 add on CUDA)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (g + b).to(tl.bfloat16).to(tl.float32)

    # numerically stable softmax in fp32
    y = tl.where(mask, y, float('-inf'))
    m = tl.max(y, axis=0)
    e = tl.exp(y - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)

    tl.store(Out_ptr + row * stride_row + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 2048, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS tensor cores (bf16)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _gelu_bias_softmax_kernel[(Mrows,)](
            h, self.b2, out,
            N, h.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
