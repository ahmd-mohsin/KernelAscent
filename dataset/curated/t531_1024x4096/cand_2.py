import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 531
M, D, DT = 1024, 4096, torch.bfloat16


@triton.jit
def _scale_relu_softmax_kernel(
    X, Y,
    stride_xm, stride_ym,
    N,
    SCALE,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)  # bf16

    # scale in bf16 (matches reference: bf16 mul, then relu in bf16)
    s = tl.full((), SCALE, dtype=tl.float32).to(tl.bfloat16)
    x = x * s
    zero = tl.zeros((BLOCK_N,), dtype=tl.bfloat16)
    x = tl.maximum(x, zero)

    # softmax with fp32 accumulation (matches PyTorch internal upcast)
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))
    row_max = tl.max(xf, axis=0)
    num = tl.exp(xf - row_max)
    num = tl.where(mask, num, 0.0)
    denom = tl.sum(num, axis=0)
    out = num / denom

    tl.store(Y + row * stride_ym + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        h = torch.matmul(x, self.W0)  # cuBLAS bf16 matmul (tensor cores)
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(n)
        _scale_relu_softmax_kernel[(m,)](
            h, out,
            h.stride(0), out.stride(0),
            n,
            1.3067,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
