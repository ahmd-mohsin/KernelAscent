import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 546
M, D, DT = 512, 2048, torch.float16


@triton.jit
def _epilogue_softmax_kernel(
    X_ptr, B_ptr, Out_ptr,
    N, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # Load matmul output (fp16) and bias (fp16)
    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float16)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float16)

    # Elementwise epilogue in fp16 (matches reference fp16 arithmetic)
    c1 = tl.full((), 1.4852, tl.float16)
    c2 = tl.full((), 1.1826, tl.float16)
    zero = tl.full((), 0.0, tl.float16)

    x = x * c1
    x = tl.maximum(x, zero)
    x = x + b
    x = x * c2

    # Softmax with fp32 accumulation (matches PyTorch half softmax semantics)
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))
    row_max = tl.max(xf, axis=0)
    xf = xf - row_max
    num = tl.exp(xf)
    num = tl.where(mask, num, 0.0)
    denom = tl.sum(num, axis=0)
    y = num / denom

    tl.store(Out_ptr + row * stride_o + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores) - identical to reference matmul
        h = x @ self.W0

        Mrows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 1024 else 4

        _epilogue_softmax_kernel[(Mrows,)](
            h, self.b3, out,
            N, h.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
