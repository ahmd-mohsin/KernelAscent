import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 280
M, D, DT = 2048, 4096, torch.float16


@triton.jit
def _fused_rms_bias_ln_kernel(
    X, W1, B2, G, B, Y,
    N,
    EPS_RMS, EPS_LN,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x_ptr = X + row * N + offs

    # ---- RMSNorm (compute in fp32, cast to fp16, scale by weight in fp16) ----
    x = tl.load(x_ptr, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + EPS_RMS)
    xh = (x * rstd).to(tl.float16)

    w1 = tl.load(W1 + offs, mask=mask, other=0.0)
    b2 = tl.load(B2 + offs, mask=mask, other=0.0)

    # fp16 multiply, then fp16 add (matches PyTorch elementwise op order/rounding)
    t = xh * w1
    t = t + b2

    # ---- LayerNorm (fp32 internal math like PyTorch's half layer_norm) ----
    tf = t.to(tl.float32)
    tf = tl.where(mask, tf, 0.0)
    mean = tl.sum(tf, axis=0) / N
    d = tl.where(mask, tf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    inv = 1.0 / tl.sqrt(var + EPS_LN)

    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    y = d * inv * g + b

    tl.store(Y + row * N + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 2048, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.W4 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM 1 (cuBLAS tensor cores)
        h = x @ self.W0
        h = h.contiguous()

        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_rms_bias_ln_kernel[(Mrows,)](
            h, self.rms1_w, self.b2, self.ln3_g, self.ln3_b, out,
            N, 1e-6, 1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )

        # GEMM 2
        return out @ self.W4
