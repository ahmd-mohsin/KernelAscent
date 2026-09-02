import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 959
M, D, DT = 4096, 1025, torch.bfloat16


@triton.jit
def _fused_ln_ln_bias_relu_softmax(
    X, G1, B1, G2, B2, B3, OUT,
    N, stride_x, stride_o,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm 1 (fp32 compute, bf16 output like PyTorch)
    mean1 = tl.sum(x, axis=0) / N
    d1 = tl.where(mask, x - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / N
    rstd1 = 1.0 / tl.sqrt(var1 + EPS)
    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    y = d1 * rstd1 * g1 + b1
    y = y.to(tl.bfloat16).to(tl.float32)  # round to bf16 as PyTorch does between ops

    # LayerNorm 2
    mean2 = tl.sum(tl.where(mask, y, 0.0), axis=0) / N
    d2 = tl.where(mask, y - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)
    g2 = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    z = d2 * rstd2 * g2 + b2
    z_bf = z.to(tl.bfloat16)

    # bias add in bf16 (matches PyTorch bf16 + bf16)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0)
    z_bf = z_bf + b3

    # relu
    z_bf = tl.maximum(z_bf, z_bf * 0)

    # softmax (fp32 accumulation, bf16 output like PyTorch)
    zf = tl.where(mask, z_bf.to(tl.float32), float('-inf'))
    m = tl.max(zf, axis=0)
    e = tl.exp(zf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 512, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_ln_ln_bias_relu_softmax[(Mrows,)](
            h, self.ln1_g, self.ln1_b, self.ln2_g, self.ln2_b, self.b3, out,
            N, h.stride(0), out.stride(0),
            EPS=1e-5, BLOCK=BLOCK,
            num_warps=4,
        )
        return out
