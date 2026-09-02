import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 321
M, D, DT = 2048, 4097, torch.bfloat16


@triton.jit
def _fused_softmax_rms_gelu_bias(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, stride_row,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # ---- softmax (fp32 accumulation, matching PyTorch bf16 softmax) ----
    x = tl.load(x_ptr + row * stride_row + offs, mask=mask,
                other=float('-inf')).to(tl.float32)
    row_max = tl.max(x, axis=0)
    e = tl.math.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    sm = e / denom
    # softmax output is rounded to bf16 (as in reference)
    sm_bf = sm.to(tl.bfloat16)

    # ---- RMS norm (computed in fp32 on the bf16-rounded softmax output) ----
    xf = sm_bf.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + 1e-6)
    y_bf = (xf * rstd).to(tl.bfloat16)

    # ---- scale by rms1_w (bf16 op: fp32 opmath, round to bf16) ----
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y_bf = (y_bf.to(tl.float32) * w).to(tl.bfloat16)

    # ---- exact GELU (erf-based, fp32 opmath, round to bf16) ----
    yf = y_bf.to(tl.float32)
    g = 0.5 * yf * (1.0 + tl.math.erf(yf * 0.7071067811865476))
    g_bf = g.to(tl.bfloat16)

    # ---- add bias b3 (bf16 op: fp32 opmath, round to bf16) ----
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = (g_bf.to(tl.float32) + b).to(tl.bfloat16)

    tl.store(out_ptr + row * stride_row + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        rows, N = x.shape[0], x.shape[-1]
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_softmax_rms_gelu_bias[(rows,)](
            x, self.rms1_w, self.b3, out,
            N, x.stride(0),
            BLOCK=BLOCK,
            num_warps=16,
        )
        return out
