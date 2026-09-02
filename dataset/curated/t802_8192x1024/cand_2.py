import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 802
M, D, DT = 8192, 1024, torch.bfloat16


@triton.jit
def _fused_norm_kernel(
    X_ptr, OUT_ptr,
    W0_ptr, G1_ptr, B1_ptr, W2_ptr, B3_ptr,
    stride_x, stride_o,
    D_: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D_

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- RMSNorm 0 ----
    ms0 = tl.sum(x * x, axis=0) / D_
    y = x * (1.0 / tl.sqrt(ms0 + 1e-6))
    # round to bf16 (matches .to(x.dtype)) then multiply by weight (bf16 mul -> fp32 opmath, bf16 out)
    y = y.to(tl.bfloat16).to(tl.float32)
    w0 = tl.load(W0_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x1 = (y * w0).to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm ----
    mean = tl.sum(x1, axis=0) / D_
    diff = tl.where(mask, x1 - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D_
    inv = 1.0 / tl.sqrt(var + 1e-5)
    g1 = tl.load(G1_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    z = diff * inv * g1 + b1
    z = z.to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm 2 ----
    ms2 = tl.sum(z * z, axis=0) / D_
    u = z * (1.0 / tl.sqrt(ms2 + 1e-6))
    u = u.to(tl.bfloat16).to(tl.float32)
    w2 = tl.load(W2_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    v = (u * w2).to(tl.bfloat16).to(tl.float32)

    # ---- bias add ----
    b3 = tl.load(B3_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = (v + b3).to(tl.bfloat16)

    tl.store(OUT_ptr + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2d = x.contiguous().view(-1, d)
        m = x2d.shape[0]
        out = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(d)
        _fused_norm_kernel[(m,)](
            x2d, out,
            self.rms0_w, self.ln1_g, self.ln1_b, self.rms2_w, self.b3,
            x2d.stride(0), out.stride(0),
            D_=d, BLOCK=BLOCK,
            num_warps=8,
        )
        return out.view(orig_shape)
