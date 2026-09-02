import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 433
M, D, DT = 1024, 4097, torch.bfloat16


@triton.jit
def _fused_rms_softmax_gelu_kernel(
    X_ptr, W_ptr, Y_ptr,
    D, stride_x, stride_y,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    # ---- RMSNorm (fp32 math, matching reference) ----
    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    mean_sq = tl.sum(x * x, axis=0) / D
    inv_rms = 1.0 / tl.sqrt(mean_sq + eps)
    xn_bf16 = (x * inv_rms).to(tl.bfloat16)  # round to input dtype like .to(x.dtype)

    w = tl.load(W_ptr + offs, mask=mask, other=0.0)
    # bf16 * bf16 in PyTorch computes in fp32 (opmath), rounds to bf16
    y_bf16 = (xn_bf16.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    # ---- Softmax (fp32 accumulation, bf16 output rounding) ----
    yf = y_bf16.to(tl.float32)
    yf = tl.where(mask, yf, float('-inf'))
    row_max = tl.max(yf, axis=0)
    e = tl.exp(yf - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    sm_bf16 = (e / denom).to(tl.bfloat16)

    # ---- GELU (erf-based, fp32 opmath, bf16 rounding) ----
    g = sm_bf16.to(tl.float32)
    gelu = g * 0.5 * (1.0 + tl.math.erf(g * 0.7071067811865476))
    gelu_bf16 = gelu.to(tl.bfloat16)

    # ---- Scale by 1.158 (fp32 opmath, bf16 rounding) ----
    out = (gelu_bf16.to(tl.float32) * 1.158).to(tl.bfloat16)

    tl.store(Y_ptr + row * stride_y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.bfloat16 or x.dim() != 2:
            # Fallback: reference implementation
            _xf = x.float()
            xr = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            xr = torch.softmax(xr, dim=-1)
            xr = F.gelu(xr)
            return xr * 1.158

        x = x.contiguous()
        n_rows, d = x.shape
        w = self.rms0_w
        if not w.is_cuda:
            w = w.to(x.device)
        y = torch.empty_like(x)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4
        if BLOCK >= 8192:
            num_warps = 16

        _fused_rms_softmax_gelu_kernel[(n_rows,)](
            x, w, y,
            d, x.stride(0), y.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
