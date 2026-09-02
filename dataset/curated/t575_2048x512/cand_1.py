import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 575
M, D, DT = 2048, 512, torch.float16


@triton.jit
def _fused_norms_kernel(
    X_ptr, W2_ptr, W3_ptr, G_ptr, B_ptr, Out_ptr,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    base = row * N

    # ---- load row (fp16 -> fp32) ----
    xf = tl.load(X_ptr + base + offs).to(tl.float32)

    # ---- RMSNorm #1 : (xf * rsqrt(mean(xf^2)+eps)).half() * w2  (fp16 mult) ----
    inv1 = tl.math.rsqrt(tl.sum(xf * xf, axis=0) / N + 1e-6)
    w2 = tl.load(W2_ptr + offs)                      # fp16
    y16 = (xf * inv1).to(tl.float16) * w2            # fp16 multiply (matches ref)

    # ---- RMSNorm #2 ----
    xf = y16.to(tl.float32)
    inv2 = tl.math.rsqrt(tl.sum(xf * xf, axis=0) / N + 1e-6)
    w3 = tl.load(W3_ptr + offs)                      # fp16
    y16 = (xf * inv2).to(tl.float16) * w3            # fp16 multiply

    # ---- ReLU (fp16 value, exact) ----
    xf = tl.maximum(y16.to(tl.float32), 0.0)

    # ---- LayerNorm (fp32 stats, like PyTorch half layer_norm) ----
    mean = tl.sum(xf, axis=0) / N
    d = xf - mean
    var = tl.sum(d * d, axis=0) / N
    inv = tl.math.rsqrt(var + 1e-5)
    g = tl.load(G_ptr + offs).to(tl.float32)
    b = tl.load(B_ptr + offs).to(tl.float32)
    out = d * inv * g + b

    tl.store(Out_ptr + base + offs, out.to(tl.float16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.W1 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln5_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln5_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # Two GEMMs via cuBLAS (kept separate to preserve fp16 rounding exactly)
        x = x @ self.W0
        x = x @ self.W1

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        _fused_norms_kernel[(rows,)](
            x2, self.rms2_w, self.rms3_w, self.ln5_g, self.ln5_b, out,
            N=N, BLOCK=N, num_warps=8,
        )
        return out.view(orig_shape)
