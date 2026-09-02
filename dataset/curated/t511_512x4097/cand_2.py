import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 511
M, D, DT = 512, 4097, torch.float16


@triton.jit
def _fused_row_kernel(
    X_ptr, W_ptr, G_ptr, B_ptr, Y_ptr,
    N, stride_x, stride_y,
    S1, S2,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # ---- load row (fp16 -> fp32) ----
    x = tl.load(X_ptr + row * stride_x + offs, mask=mask,
                other=float('-inf')).to(tl.float32)

    # ---- softmax (fp32 accumulation, like PyTorch fp16 softmax) ----
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    p = e / denom
    # cast to fp16 (softmax output dtype), then scale in fp32 opmath -> fp16
    p16 = p.to(tl.float16)
    y = (p16.to(tl.float32) * S1).to(tl.float16)

    # ---- RMSNorm (fp32) ----
    yf = y.to(tl.float32)
    ms = tl.sum(yf * yf, axis=0) / N
    inv_rms = 1.0 / tl.sqrt(ms + 1e-6)
    z16 = (yf * inv_rms).to(tl.float16)

    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    z16 = (z16.to(tl.float32) * w).to(tl.float16)
    z16 = (z16.to(tl.float32) * S2).to(tl.float16)

    # ---- LayerNorm (fp32 statistics, like PyTorch) ----
    zf = z16.to(tl.float32)
    zf = tl.where(mask, zf, 0.0)
    mean = tl.sum(zf, axis=0) / N
    diff = tl.where(mask, zf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = (diff * rstd * g + b).to(tl.float16)

    tl.store(Y_ptr + row * stride_y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms2_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.float16:
            # fallback (reference path)
            x = torch.softmax(x, dim=-1)
            x = x * 1.0652
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            x = x * 1.061
            x = F.layer_norm(x, (x.shape[-1],), self.ln4_g, self.ln4_b)
            return x

        orig_shape = x.shape
        N = orig_shape[-1]
        x2d = x.contiguous().view(-1, N)
        Mrows = x2d.shape[0]
        y = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8
        if BLOCK >= 4096:
            num_warps = 16
        if BLOCK >= 8192:
            num_warps = 32

        _fused_row_kernel[(Mrows,)](
            x2d, self.rms2_w, self.ln4_g, self.ln4_b, y,
            N, x2d.stride(0), y.stride(0),
            1.0652, 1.061,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
