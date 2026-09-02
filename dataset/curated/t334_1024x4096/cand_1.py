import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 334
M, D, DT = 1024, 4096, torch.bfloat16


@triton.jit
def _softmax_gelu_bias_scale_kernel(
    X, B, Y,
    stride_xm, stride_ym,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax in fp32 (matches PyTorch accumulate type), round to bf16
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    p = e / s
    p = p.to(tl.bfloat16).to(tl.float32)

    # exact (erf) GELU in fp32, round to bf16
    g = p * 0.5 * (1.0 + tl.math.erf(p * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # + bias (fp32 opmath, round to bf16)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    h = (g + b).to(tl.bfloat16).to(tl.float32)

    # * 1.0072 (fp32 opmath, round to bf16)
    out = (h * 1.0072).to(tl.bfloat16)

    tl.store(Y + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 matmul
        Mrows, N = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _softmax_gelu_bias_scale_kernel[(Mrows,)](
            h, self.b3, y,
            h.stride(0), y.stride(0),
            N, BLOCK,
            num_warps=8,
        )
        return y
