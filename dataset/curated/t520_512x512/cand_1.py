import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 520
M, D, DT = 512, 512, torch.bfloat16


@triton.jit
def _fused_kernel(
    x_ptr, w0_ptr, w3_ptr, g_ptr, b_ptr, out_ptr,
    D: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    base = row * D

    # ---- RMSNorm 0 (compute in fp32, round to bf16, then bf16 weight mul) ----
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / D
    r = 1.0 / tl.sqrt(ms + 1e-6)
    y_bf = (x * r).to(tl.bfloat16)
    w0 = tl.load(w0_ptr + offs, mask=mask, other=0.0)
    y_bf = y_bf * w0  # bf16 * bf16 (upcast-mul-round, matches PyTorch opmath)

    # ---- scalar mul (opmath fp32, round to bf16) ----
    y_bf = (y_bf.to(tl.float32) * 1.1232).to(tl.bfloat16)

    # ---- softmax (accumulate in fp32, output bf16) ----
    xf = y_bf.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))
    mx = tl.max(xf, axis=0)
    e = tl.exp(xf - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y_bf = (e / s).to(tl.bfloat16)

    # ---- RMSNorm 3 ----
    xf = y_bf.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / D
    r = 1.0 / tl.sqrt(ms + 1e-6)
    y_bf = (xf * r).to(tl.bfloat16)
    w3 = tl.load(w3_ptr + offs, mask=mask, other=0.0)
    y_bf = y_bf * w3

    # ---- LayerNorm (fp32 stats + fp32 affine, round to bf16) ----
    xf = y_bf.to(tl.float32)
    xf = tl.where(mask, xf, 0.0)
    mean = tl.sum(xf, axis=0) / D
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = (diff * rstd * g + b).to(tl.bfloat16)
    tl.store(out_ptr + base + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        n_rows = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(n_rows,)](
            x2, self.rms0_w, self.rms3_w, self.ln4_g, self.ln4_b, out,
            D=d, BLOCK=BLOCK,
            num_warps=4 if BLOCK <= 1024 else 8,
        )
        return out.view(orig_shape)
