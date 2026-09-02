import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 83
M, D, DT = 4096, 4097, torch.bfloat16


@triton.jit
def _softmax_bias_kernel(
    X_ptr, B_ptr, Y_ptr,
    stride_x, stride_y,
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax in fp32 (matches PyTorch's fp32 accumulation for bf16 softmax)
    m = tl.max(x, axis=0)
    e = tl.math.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s

    # PyTorch produces bf16 softmax output, then add is computed in fp32 (opmath)
    # and rounded back to bf16 -> replicate that rounding sequence.
    y_bf16 = y.to(tl.bfloat16)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = (y_bf16.to(tl.float32) + b).to(tl.bfloat16)

    tl.store(Y_ptr + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 1024, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100 for bf16)
        z = torch.matmul(x, self.W0)
        if not z.is_contiguous():
            z = z.contiguous()

        rows, n = z.shape
        out = torch.empty_like(z)

        BLOCK = triton.next_power_of_2(n)
        num_warps = 4
        if BLOCK >= 1024:
            num_warps = 8
        if BLOCK >= 4096:
            num_warps = 16

        _softmax_bias_kernel[(rows,)](
            z, self.b2, out,
            z.stride(0), out.stride(0),
            n,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
