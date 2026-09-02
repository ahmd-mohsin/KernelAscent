import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 511
M, D, DT = 512, 4097, torch.float16


@triton.jit
def _fused_softmax_rms_ln_kernel(
    X, W, G, B, Y,
    D: tl.constexpr,
    stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    # ---- softmax (fp32 accumulation, fp16 output like PyTorch half softmax) ----
    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    sm16 = (e / denom).to(tl.float16)

    # ---- x = x * 1.0652 (opmath float, store fp16) ----
    x1 = (sm16.to(tl.float32) * 1.0652).to(tl.float16)

    # ---- RMSNorm in fp32, cast back to fp16, multiply by weight ----
    xf = x1.to(tl.float32)
    xf = tl.where(mask, xf, 0.0)
    ms = tl.sum(xf * xf, axis=0) / D
    rrms = 1.0 / tl.sqrt(ms + 1e-6)
    y16 = (xf * rrms).to(tl.float16)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    z16 = (y16.to(tl.float32) * w).to(tl.float16)

    # ---- x = x * 1.061 ----
    z16 = (z16.to(tl.float32) * 1.061).to(tl.float16)

    # ---- LayerNorm (fp32 internal math like PyTorch half layer_norm) ----
    zf = z16.to(tl.float32)
    zf = tl.where(mask, zf, 0.0)
    mean = tl.sum(zf, axis=0) / D
    diff = tl.where(mask, zf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    out = (diff * rstd * g + b).to(tl.float16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms2_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.float16:
            return self._forward_ref(x)

        orig_shape = x.shape
        d = orig_shape[-1]
        x2d = x.contiguous().view(-1, d)
        m = x2d.shape[0]
        y = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK <= 8192 else 16

        _fused_softmax_rms_ln_kernel[(m,)](
            x2d, self.rms2_w, self.ln4_g, self.ln4_b, y,
            d,
            x2d.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)

    def _forward_ref(self, x):
        x = torch.softmax(x, dim=-1)
        x = x * 1.0652
        _xf = x.float()
        x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
        x = x * 1.061
        x = F.layer_norm(x, (x.shape[-1],), self.ln4_g, self.ln4_b)
        return x
