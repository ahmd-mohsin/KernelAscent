import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 561
M, D, DT = 8192, 1025, torch.bfloat16


@triton.jit
def _rms_gelu_kernel(
    X, W, Y,
    N, stride_x, stride_y,
    eps, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMS norm in fp32
    ms = tl.sum(xf * xf, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + eps)
    n = xf * rstd
    # cast to bf16 (matches .to(x.dtype))
    n_bf = n.to(tl.bfloat16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    # bf16 * bf16 -> fp32 compute, round to bf16
    t = (n_bf.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    # exact GELU in fp32, round to bf16
    tf = t.to(tl.float32)
    g = 0.5 * tf * (1.0 + tl.math.erf(tf * 0.7071067811865476))
    g_bf = g.to(tl.bfloat16)

    # scalar multiply: fp32 compute, round to bf16
    out = (g_bf.to(tl.float32) * scale).to(tl.bfloat16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 2048, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        M_, N_ = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N_)
        _rms_gelu_kernel[(M_,)](
            x, self.rms1_w, y,
            N_, x.stride(0), y.stride(0),
            1e-6, 1.3603,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
