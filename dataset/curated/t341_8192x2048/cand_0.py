import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 341
M, D, DT = 8192, 2048, torch.float16


@triton.jit
def _relu_gelu_softmax_kernel(
    X_ptr, Y_ptr,
    N, stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride + offs, mask=mask, other=0.0).to(tl.float32)

    # ReLU
    x = tl.maximum(x, 0.0)

    # exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))

    # match reference fp16 intermediate precision
    g = g.to(tl.float16).to(tl.float32)

    # softmax (fp32 accumulation, like PyTorch half softmax)
    g_masked = tl.where(mask, g, float('-inf'))
    row_max = tl.max(g_masked, axis=0)
    e = tl.exp(g_masked - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Y_ptr + row * stride + offs, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS half GEMM (tensor cores on A100)
        z = torch.matmul(x, self.W0)
        z = z.contiguous()
        M_rows, N = z.shape
        y = torch.empty_like(z)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _relu_gelu_softmax_kernel[(M_rows,)](
            z, y,
            N, z.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
