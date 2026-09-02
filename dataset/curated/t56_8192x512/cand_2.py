import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 56
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _fused_softmax_gelu_bias_kernel(
    X_ptr, B_ptr, Y_ptr,
    stride_row,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_row + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # x * 1.1687 (opmath in fp32, rounded back to bf16, like PyTorch bf16 mul)
    x = (x * 1.1687).to(tl.bfloat16).to(tl.float32)

    # softmax with fp32 accumulation (matches PyTorch's acc_type for bf16)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = (e / s).to(tl.bfloat16).to(tl.float32)

    # exact (erf-based) GELU computed in fp32, rounded to bf16
    g = 0.5 * p * (1.0 + tl.math.erf(p * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # relu
    g = tl.maximum(g, 0.0)

    # bias add (fp32 opmath, round to bf16)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (g + b).to(tl.bfloat16)

    tl.store(Y_ptr + row * stride_row + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b4 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = x * 1.1687
            y = torch.softmax(y, dim=-1)
            y = F.gelu(y)
            y = torch.relu(y)
            return y + self.b4

        x = x.contiguous()
        rows, n = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _fused_softmax_gelu_bias_kernel[(rows,)](
            x, self.b4, out,
            x.stride(0),
            N=n,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
