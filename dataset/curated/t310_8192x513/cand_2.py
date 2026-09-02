import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 310
M, D, DT = 8192, 513, torch.bfloat16


@triton.jit
def _gelu_rms_softmax_kernel(
    X_ptr, W_ptr, Out_ptr,
    N, stride_x, stride_o,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # exact GELU (erf variant), computed in fp32 then rounded to bf16
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    g_bf = g.to(tl.bfloat16)
    gf = g_bf.to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(tl.where(mask, gf * gf, 0.0), axis=0) / N
    r = tl.math.rsqrt(ms + eps)
    y_bf = (gf * r).to(tl.bfloat16)

    # weight multiply in bf16 (matches reference dtype semantics)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.bfloat16)
    y_bf = y_bf * w

    # softmax in fp32
    yf = tl.where(mask, y_bf.to(tl.float32), float('-inf'))
    m = tl.max(yf, axis=0)
    e = tl.exp(yf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)

    tl.store(Out_ptr + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 2048, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # (M, 2048) bf16, tensor-core matmul
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _gelu_rms_softmax_kernel[(Mrows,)](
            h, self.rms2_w, out,
            N, h.stride(0), out.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
