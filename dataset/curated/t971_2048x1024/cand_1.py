import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 971
M, D, DT = 2048, 1024, torch.bfloat16


@triton.jit
def _fused_gelu_relu_bias_softmax(
    X_ptr, B_ptr, O_ptr,
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # exact (erf) GELU computed in fp32, rounded to bf16 (matches PyTorch opmath)
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16)

    # relu in bf16
    zero = tl.zeros_like(g)
    g = tl.maximum(g, zero)

    # bias add: fp32 opmath, rounded back to bf16 (matches PyTorch)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    z = (g.to(tl.float32) + b).to(tl.bfloat16)

    # softmax with fp32 accumulation (matches PyTorch bf16 softmax)
    zf = z.to(tl.float32)
    zf = tl.where(mask, zf, float('-inf'))
    row_max = tl.max(zf, axis=0)
    e = tl.exp(zf - row_max)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)

    tl.store(O_ptr + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 1024, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS bf16 matmul (tensor cores on A100)
        y = torch.matmul(x, self.W0)
        y = y.contiguous()
        Mrows, N = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(N)
        _fused_gelu_relu_bias_softmax[(Mrows,)](
            y, self.b3, out, N,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
