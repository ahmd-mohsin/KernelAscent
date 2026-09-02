import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 729
M, D, DT = 2048, 2048, torch.float16


@triton.jit
def _gelu_bias_softmax_kernel(
    Y_ptr, B_ptr, O_ptr,
    N, stride_row,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # load matmul output row (fp16), compute in fp32 like PyTorch opmath
    y = tl.load(Y_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)

    # exact (erf) GELU, rounded back to fp16 as an intermediate tensor would be
    g = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)

    # scalar multiply (fp32 compute, fp16 storage semantics)
    g = (g * 1.3162).to(tl.float16).to(tl.float32)

    # bias add
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    g = (g + b).to(tl.float16).to(tl.float32)

    # softmax in fp32 (matches PyTorch's fp32 accumulation for fp16 softmax)
    g = tl.where(mask, g, float('-inf'))
    row_max = tl.max(g, axis=0)
    e = tl.exp(g - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = (e / denom).to(tl.float16)

    tl.store(O_ptr + row * stride_row + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS fp16 GEMM (tensor cores)
        y = x @ self.W0
        y = y.contiguous()
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(n)
        _gelu_bias_softmax_kernel[(m,)](
            y, self.b3, out,
            n, y.stride(0),
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
