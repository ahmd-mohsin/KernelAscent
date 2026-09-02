import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 393
M, D, DT = 2048, 2049, torch.float16


@triton.jit
def _fused_gelu_bias_gelu_softmax(
    X_ptr, B_ptr, Y_ptr,
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # gelu (exact, computed in fp32 like PyTorch's half opmath), round to fp16
    x = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.float16).to(tl.float32)

    # + bias (fp32 opmath), round to fp16
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x = x + b
    x = x.to(tl.float16).to(tl.float32)

    # gelu again, round to fp16
    x = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.float16).to(tl.float32)

    # softmax in fp32 (matches PyTorch half softmax accumulation)
    x = tl.where(mask, x, float('-inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Y_ptr + row * N + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 1024, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.W5 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM 1 (cuBLAS tensor cores)
        h = x @ self.W0  # (M, 1024), fp16, contiguous

        m, n = h.shape
        BLOCK = triton.next_power_of_2(n)
        _fused_gelu_bias_gelu_softmax[(m,)](
            h, self.b2, h, n,
            BLOCK=BLOCK,
            num_warps=8,
        )

        # GEMM 2 (cuBLAS tensor cores)
        return h @ self.W5
