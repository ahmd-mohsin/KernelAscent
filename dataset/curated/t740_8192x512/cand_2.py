import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 740
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _fused_rms_softmax_ln_kernel(
    X, W, G, B, Y,
    N, stride_x, stride_y,
    RMS_EPS: tl.constexpr,
    LN_EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    # ---- load row (bf16 -> fp32) ----
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- RMSNorm (compute in fp32, round to bf16 like reference) ----
    ms = tl.sum(x * x, axis=0) / N
    inv_rms = 1.0 / tl.sqrt(ms + RMS_EPS)
    xn = (x * inv_rms).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xn * w).to(tl.bfloat16).to(tl.float32)  # bf16 rounding as in `* self.rms1_w`

    # ---- softmax (fp32 accumulation, bf16 output rounding) ----
    y = tl.where(mask, y, float("-inf"))
    row_max = tl.max(y, axis=0)
    e = tl.exp(y - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    p = (e / denom).to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm (fp32 stats, bf16 output) ----
    mean = tl.sum(tl.where(mask, p, 0.0), axis=0) / N
    diff = tl.where(mask, p - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + LN_EPS)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    out = diff * rstd * g + b

    tl.store(Y + row * stride_y + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # Matmul via cuBLAS (tensor cores)
        x = x @ self.W0
        x = x.contiguous()
        rows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_rms_softmax_ln_kernel[(rows,)](
            x, self.rms1_w, self.ln3_g, self.ln3_b, y,
            N, x.stride(0), y.stride(0),
            RMS_EPS=1e-6, LN_EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return y
