import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 502
M, D, DT = 8192, 2049, torch.bfloat16


@triton.jit
def _fused_ln_rms_kernel(
    x_ptr, g_ptr, b_ptr, w_ptr, out_ptr,
    N, stride_x, stride_o,
    LN_EPS: tl.constexpr, RMS_EPS: tl.constexpr, SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    n_f = N.to(tl.float32)

    # LayerNorm (fp32 math, bf16 rounding of output like PyTorch)
    mean = tl.sum(x, axis=0) / n_f
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / n_f
    rstd = 1.0 / tl.sqrt(var + LN_EPS)
    y = diff * rstd * g + b
    y = y.to(out_ptr.dtype.element_ty).to(tl.float32)  # round to bf16, then _xf = x.float()

    # RMSNorm
    ms = tl.sum(tl.where(mask, y * y, 0.0), axis=0) / n_f
    r = 1.0 / tl.sqrt(ms + RMS_EPS)
    t1 = (y * r).to(out_ptr.dtype.element_ty).to(tl.float32)   # .to(x.dtype)
    t2 = (t1 * w).to(out_ptr.dtype.element_ty).to(tl.float32)  # * rms1_w (bf16 opmath rounding)
    out = (t2 * SCALE).to(out_ptr.dtype.element_ty)            # * 1.1595

    tl.store(out_ptr + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK <= 4096 else 16

        _fused_ln_rms_kernel[(rows,)](
            x2, self.ln0_g, self.ln0_b, self.rms1_w, out,
            N, x2.stride(0), out.stride(0),
            LN_EPS=1e-5, RMS_EPS=1e-6, SCALE=1.1595,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)
