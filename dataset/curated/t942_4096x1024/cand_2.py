import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 942
M, D, DT = 4096, 1024, torch.float16


@triton.jit
def _fused_kernel(
    X_ptr, W_ptr, G_ptr, B_ptr, Out_ptr,
    N,  # row length
    stride_x, stride_o,
    RMS_EPS: tl.constexpr,
    LN_EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # ---- RMSNorm (fp32) ----
    ms = tl.sum(xf * xf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + RMS_EPS)
    y = (xf * r).to(tl.float16)  # round to half (matches .to(x.dtype))

    # multiply by rms0_w: half*half computed in fp32, rounded to half
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y.to(tl.float32) * w).to(tl.float16)

    # relu
    y = tl.maximum(y, 0.0)

    # ---- LayerNorm (stats in fp32) ----
    z = y.to(tl.float32)
    zm = tl.where(mask, z, 0.0)
    mean = tl.sum(zm, axis=0) / N
    diff = tl.where(mask, z - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv = 1.0 / tl.sqrt(var + LN_EPS)

    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    h = ((z - mean) * inv * g + b).to(tl.float16)

    # ---- GELU (erf, computed in fp32 like PyTorch opmath) ----
    t = h.to(tl.float32)
    out = t * 0.5 * (1.0 + tl.math.erf(t * 0.70710678118654752440))
    out = out.to(tl.float16)

    tl.store(Out_ptr + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float()
            y = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            y = torch.relu(y)
            y = F.layer_norm(y, (y.shape[-1],), self.ln2_g, self.ln2_b)
            return F.gelu(y)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_kernel[(rows,)](
            x2, self.rms0_w, self.ln2_g, self.ln2_b, out,
            N,
            x2.stride(0), out.stride(0),
            RMS_EPS=1e-6, LN_EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
