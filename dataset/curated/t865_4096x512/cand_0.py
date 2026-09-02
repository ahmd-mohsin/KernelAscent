import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 865
M, D, DT = 4096, 512, torch.float16


@triton.jit
def _fused_norm_kernel(
    X, G, B, W, OUT,
    N,
    stride_x, stride_o,
    LN_EPS: tl.constexpr,
    RMS_EPS: tl.constexpr,
    S1: tl.constexpr,   # 1.4994
    S2: tl.constexpr,   # 1.1216
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm (fp32 math, matching PyTorch half opmath) ----
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + LN_EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = (x - mean) * rstd * g + b
    y = y.to(tl.float16)                       # layer_norm output rounding

    # ---- scale by 1.4994 (half result, float opmath) ----
    y = (y.to(tl.float32) * S1).to(tl.float16)

    # ---- RMSNorm in fp32 ----
    xf = y.to(tl.float32)
    xf = tl.where(mask, xf, 0.0)
    ms = tl.sum(xf * xf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + RMS_EPS)
    z = (xf * r).to(tl.float16)                # .to(x.dtype) rounding

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    z = (z.to(tl.float32) * w).to(tl.float16)  # * rms3_w (half mul, float opmath)

    out = (z.to(tl.float32) * S2).to(tl.float16)  # * 1.1216

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores (fp32 accumulate, same as reference)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        _fused_norm_kernel[(Mrows,)](
            h, self.ln1_g, self.ln1_b, self.rms3_w, out,
            N,
            h.stride(0), out.stride(0),
            LN_EPS=1e-5,
            RMS_EPS=1e-6,
            S1=1.4994,
            S2=1.1216,
            BLOCK=BLOCK,
            num_warps=16,
        )
        return out
