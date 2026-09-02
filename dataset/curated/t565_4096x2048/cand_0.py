import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 565
M, D, DT = 4096, 2048, torch.bfloat16


@triton.jit
def _fused_post_kernel(
    X, OUT,
    G1, B1, W2, G5, B5,
    stride_row,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    ptr = X + row * stride_row + offs

    x = tl.load(ptr).to(tl.float32)

    # ---- LayerNorm 1 (fp32 internal, eps=1e-5) ----
    mean = tl.sum(x, axis=0) / N
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / N
    y = xc * tl.rsqrt(var + 1e-5)
    g1 = tl.load(G1 + offs).to(tl.float32)
    b1 = tl.load(B1 + offs).to(tl.float32)
    y = y * g1 + b1
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm (fp32, eps=1e-6), cast bf16, then * w in fp32 -> bf16 ----
    rms = tl.rsqrt(tl.sum(y * y, axis=0) / N + 1e-6)
    y = (y * rms).to(tl.bfloat16).to(tl.float32)
    w2 = tl.load(W2 + offs).to(tl.float32)
    y = (y * w2).to(tl.bfloat16).to(tl.float32)

    # ---- Softmax (fp32 internal) ----
    mx = tl.max(y, axis=0)
    e = tl.exp(y - mx)
    y = e / tl.sum(e, axis=0)
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- GELU (erf, fp32 opmath) ----
    y = y * 0.5 * (1.0 + tl.math.erf(y * 0.7071067811865476))
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm 5 (fp32 internal, eps=1e-5) ----
    mean2 = tl.sum(y, axis=0) / N
    yc = y - mean2
    var2 = tl.sum(yc * yc, axis=0) / N
    z = yc * tl.rsqrt(var2 + 1e-5)
    g5 = tl.load(G5 + offs).to(tl.float32)
    b5 = tl.load(B5 + offs).to(tl.float32)
    z = z * g5 + b5

    tl.store(OUT + row * stride_row + offs, z.to(tl.bfloat16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln5_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln5_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        out = torch.empty_like(h)
        rows, N = h.shape[0], h.shape[1]
        _fused_post_kernel[(rows,)](
            h, out,
            self.ln1_g, self.ln1_b, self.rms2_w, self.ln5_g, self.ln5_b,
            h.stride(0),
            N=N, BLOCK=N,
            num_warps=4,
        )
        return out
