import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 434
M, D, DT = 2048, 512, torch.bfloat16


@triton.jit
def _relu_bias_softmax_kernel(
    X_ptr, B2_ptr, B3_ptr, B4_ptr, Y_ptr,
    N, stride_row,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_row + offs, mask=mask, other=0.0)
    b2 = tl.load(B2_ptr + offs, mask=mask, other=0.0)
    b3 = tl.load(B3_ptr + offs, mask=mask, other=0.0)
    b4 = tl.load(B4_ptr + offs, mask=mask, other=0.0)

    # relu (exact on bf16)
    zero = tl.zeros_like(x)
    x = tl.maximum(x, zero)

    # sequential bf16 additions matching reference rounding behavior
    x = (x + b2).to(tl.bfloat16)
    x = (x + b3).to(tl.bfloat16)
    x = (x + b4).to(tl.bfloat16)

    # softmax computed in fp32 (matches PyTorch's internal upcast for bf16)
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))
    row_max = tl.max(xf, axis=0)
    e = tl.exp(xf - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    y = e / denom

    tl.store(Y_ptr + row * stride_row + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores (identical to reference matmul)
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        _relu_bias_softmax_kernel[(Mrows,)](
            h, self.b2, self.b3, self.b4, out,
            N, h.stride(0),
            BLOCK=BLOCK,
            num_warps=16,
        )
        return out
