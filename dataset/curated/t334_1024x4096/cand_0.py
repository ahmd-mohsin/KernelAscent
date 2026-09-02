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
    X_ptr, B_ptr, Y_ptr,
    N, stride_xm, stride_ym,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=float('-inf'))
    xf = x.to(tl.float32)

    # softmax (fp32 accumulation, matching PyTorch), round to bf16
    row_max = tl.max(xf, axis=0)
    e = tl.exp(xf - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    s = (e / denom).to(tl.bfloat16)

    # exact (erf) GELU in fp32 opmath, round to bf16
    sf = s.to(tl.float32)
    g = (0.5 * sf * (1.0 + tl.math.erf(sf * 0.7071067811865476))).to(tl.bfloat16)

    # bias add (fp32 opmath, round to bf16)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0)
    y = (g.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)

    # scale (fp32 opmath, round to bf16)
    y = (y.to(tl.float32) * SCALE).to(tl.bfloat16)

    tl.store(Y_ptr + row * stride_ym + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 GEMM (TF32/BF16 tensor cores)
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(n)
        _softmax_gelu_bias_scale_kernel[(m,)](
            h, self.b3, out,
            n, h.stride(0), out.stride(0),
            SCALE=1.0072,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
