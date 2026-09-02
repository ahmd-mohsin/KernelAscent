import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 218
M, D, DT = 2048, 1025, torch.bfloat16


@triton.jit
def _rms_gelu_kernel(
    X, W, Y,
    stride_xm,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMS norm in fp32
    ms = tl.sum(xf * xf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)

    # cast to bf16 after normalization (matches .to(x.dtype))
    t = (xf * r).to(tl.bfloat16)

    # * rms1_w  (bf16 elementwise, fp32 opmath, bf16 result)
    w = tl.load(W + cols, mask=mask, other=0.0)
    t = (t.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    # * 1.1772 (fp32 opmath, bf16 result)
    t = (t.to(tl.float32) * 1.1772).to(tl.bfloat16)
    # * 1.3155
    t = (t.to(tl.float32) * 1.3155).to(tl.bfloat16)

    # exact GELU in fp32 opmath, bf16 result
    g = t.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    out = g * 0.5 * (1.0 + tl.math.erf(g * INV_SQRT2))
    out = out.to(tl.bfloat16)

    tl.store(Y + row * N + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 512, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS bf16 matmul
        m, n = x.shape
        x = x.contiguous()
        y = torch.empty((m, n), dtype=x.dtype, device=x.device)
        _rms_gelu_kernel[(m,)](
            x, self.rms1_w, y,
            x.stride(0),
            N=n,
            BLOCK=triton.next_power_of_2(n),
            num_warps=4,
        )
        return y
