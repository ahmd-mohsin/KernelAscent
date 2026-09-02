import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 153
M, D, DT = 4096, 513, torch.bfloat16


@triton.jit
def _fused_kernel(
    x_ptr, w_ptr, g_ptr, b_ptr, out_ptr,
    N, stride_row,
    RMS_EPS: tl.constexpr, LN_EPS: tl.constexpr, SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- RMSNorm (fp32 stats, bf16 output, bf16 weight mul in fp32 opmath) ----
    ms = tl.sum(x * x, axis=0) / N
    r = 1.0 / tl.sqrt(ms + RMS_EPS)
    xn = (x * r).to(tl.bfloat16).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xn * w).to(tl.bfloat16).to(tl.float32)

    # ---- Softmax (fp32 accumulation, bf16 output) ----
    y_m = tl.where(mask, y, float('-inf'))
    mx = tl.max(y_m, axis=0)
    e = tl.exp(y_m - mx)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    p = (e / denom).to(tl.bfloat16).to(tl.float32)
    p = tl.where(mask, p, 0.0)

    # ---- LayerNorm (fp32 stats/compute, bf16 output) ----
    mean = tl.sum(p, axis=0) / N
    diff = tl.where(mask, p - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + LN_EPS)
    g = tl.load(g_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    o = ((p - mean) * rstd * g + b).to(tl.bfloat16).to(tl.float32)

    # ---- Scale (fp32 opmath, bf16 output) ----
    out = (o * SCALE).to(tl.bfloat16)
    tl.store(out_ptr + row * stride_row + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        n = orig_shape[-1]
        x2 = x.contiguous().view(-1, n)
        m = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n)
        _fused_kernel[(m,)](
            x2, self.rms0_w, self.ln2_g, self.ln2_b, out,
            n, x2.stride(0),
            RMS_EPS=1e-6, LN_EPS=1e-5, SCALE=1.3856,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out.view(orig_shape)
