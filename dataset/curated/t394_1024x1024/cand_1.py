import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 394
M, D, DT = 1024, 1024, torch.float16


@triton.jit
def _fused_norm_chain(
    X, W1, G2, B2, W4, W5, Out,
    N, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- RMSNorm 1 (compute in fp32, round to fp16 like reference) ----
    ms = tl.sum(x * x, 0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    y = (x * r).to(tl.float16).to(tl.float32)
    w1 = tl.load(W1 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y * w1).to(tl.float16).to(tl.float32)

    # ---- LayerNorm (PyTorch computes half LN in fp32 internally) ----
    mu = tl.sum(tl.where(mask, y, 0.0), 0) / N
    d = tl.where(mask, y - mu, 0.0)
    var = tl.sum(d * d, 0) / N
    inv = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    z = ((y - mu) * inv * g + b).to(tl.float16).to(tl.float32)

    # ---- GELU (erf variant, fp32 opmath, round to fp16) ----
    z = (0.5 * z * (1.0 + tl.math.erf(z * 0.7071067811865476))).to(tl.float16).to(tl.float32)

    # ---- RMSNorm 4 ----
    ms = tl.sum(tl.where(mask, z * z, 0.0), 0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    z = (z * r).to(tl.float16).to(tl.float32)
    w4 = tl.load(W4 + cols, mask=mask, other=0.0).to(tl.float32)
    z = (z * w4).to(tl.float16).to(tl.float32)

    # ---- RMSNorm 5 ----
    ms = tl.sum(tl.where(mask, z * z, 0.0), 0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    z = (z * r).to(tl.float16).to(tl.float32)
    w5 = tl.load(W5 + cols, mask=mask, other=0.0).to(tl.float32)
    out = (z * w5).to(tl.float16)

    tl.store(Out + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 2048, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS tensor-core GEMM
        h = torch.matmul(x, self.W0)

        orig_shape = h.shape
        N = orig_shape[-1]
        h2 = h.reshape(-1, N)
        if not h2.is_contiguous():
            h2 = h2.contiguous()
        rows = h2.shape[0]

        out = torch.empty_like(h2)
        BLOCK = triton.next_power_of_2(N)
        _fused_norm_chain[(rows,)](
            h2, self.rms1_w, self.ln2_g, self.ln2_b, self.rms4_w, self.rms5_w,
            out, N, h2.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=8,
        )
        return out.reshape(orig_shape)
