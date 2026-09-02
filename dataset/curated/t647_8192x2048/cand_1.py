import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 647
M, D, DT = 8192, 2048, torch.bfloat16


@triton.jit
def _fused_norm_softmax_gelu(
    X_ptr, W_ptr, G_ptr, B_ptr, Y_ptr,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)

    # ---- load row (bf16 -> fp32) ----
    x = tl.load(X_ptr + row * stride_x + cols).to(tl.float32)

    # ---- RMSNorm (fp32 accum, matches _xf.pow(2).mean + rsqrt) ----
    ms = tl.sum(x * x, axis=0) / N
    inv_rms = tl.math.rsqrt(ms + 1e-6)
    # cast to bf16 (matches .to(x.dtype)) then bf16*bf16 elementwise mul
    # (PyTorch upcasts to fp32 for the mul, rounds result to bf16)
    y_b = (x * inv_rms).to(tl.bfloat16)
    w = tl.load(W_ptr + cols).to(tl.float32)
    y = (y_b.to(tl.float32) * w).to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm (fp32 stats, eps=1e-5, affine in fp32, out bf16) ----
    mean = tl.sum(y, axis=0) / N
    diff = y - mean
    var = tl.sum(diff * diff, axis=0) / N
    inv_std = tl.math.rsqrt(var + 1e-5)
    g = tl.load(G_ptr + cols).to(tl.float32)
    b = tl.load(B_ptr + cols).to(tl.float32)
    z = (diff * inv_std) * g + b
    z = z.to(tl.bfloat16).to(tl.float32)

    # ---- Softmax (fp32 internally, out bf16) ----
    zmax = tl.max(z, axis=0)
    e = tl.exp(z - zmax)
    s = e / tl.sum(e, axis=0)
    s = s.to(tl.bfloat16).to(tl.float32)

    # ---- GELU (exact erf, fp32 opmath, out bf16) ----
    out = s * 0.5 * (1.0 + tl.math.erf(s * 0.7071067811865476))

    tl.store(Y_ptr + row * stride_y + cols, out.to(tl.bfloat16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul on tensor cores
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        _fused_norm_softmax_gelu[(Mrows,)](
            h, self.rms1_w, self.ln2_g, self.ln2_b, out,
            h.stride(0), out.stride(0),
            N=N, BLOCK=1024,
            num_warps=8,
        )
        return out
