import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 865
M, D, DT = 4096, 512, torch.float16


@triton.jit
def _fused_ln_scale_rms_kernel(
    X_ptr, G_ptr, B_ptr, W_ptr, Y_ptr,
    N,
    EPS_LN: tl.constexpr,
    EPS_RMS: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    # ---- LayerNorm (fp32 accumulation, like PyTorch's fp16 layer_norm) ----
    x = tl.load(X_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = tl.math.rsqrt(var + EPS_LN)
    xn = d * rstd

    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = xn * g + b
    # round to fp16 (layer_norm output dtype)
    y = y.to(tl.float16)

    # ---- x * 1.4994 (fp32 opmath, rounded back to fp16) ----
    y = (y.to(tl.float32) * S1).to(tl.float16)

    # ---- RMSNorm computed on fp16 values cast to fp32 ----
    yf = y.to(tl.float32)
    ms = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0) / N
    r = tl.math.rsqrt(ms + EPS_RMS)
    z = (yf * r).to(tl.float16)

    # ---- * rms3_w (fp32 opmath, round fp16) ----
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    z = (z.to(tl.float32) * w).to(tl.float16)

    # ---- * 1.1216 (fp32 opmath, round fp16) ----
    z = (z.to(tl.float32) * S2).to(tl.float16)

    tl.store(Y_ptr + row * N + cols, z, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS fp16 GEMM
        h = torch.matmul(x, self.W0)

        if not h.is_cuda:
            # CPU fallback (reference path)
            h = F.layer_norm(h, (h.shape[-1],), self.ln1_g, self.ln1_b)
            h = h * 1.4994
            _hf = h.float()
            h = (_hf * torch.rsqrt(_hf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(h.dtype) * self.rms3_w
            return h * 1.1216

        h = h.contiguous()
        orig_shape = h.shape
        N = orig_shape[-1]
        h2 = h.view(-1, N)
        rows = h2.shape[0]
        out = torch.empty_like(h2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_ln_scale_rms_kernel[(rows,)](
            h2, self.ln1_g, self.ln1_b, self.rms3_w, out,
            N,
            EPS_LN=1e-5,
            EPS_RMS=1e-6,
            S1=1.4994,
            S2=1.1216,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
