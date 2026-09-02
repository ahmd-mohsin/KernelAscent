import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 556
M, D, DT = 4096, 1024, torch.bfloat16


@triton.jit
def _fused_softmax_ln_kernel(
    X, LN_G, LN_B, B4, OUT,
    stride_xm, stride_om,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax (fp32 accumulation, like PyTorch on bf16)
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    sm = e / denom

    # cast to bf16 (softmax output dtype), relu (identity for >=0, kept for exactness)
    sm_bf = sm.to(tl.bfloat16)
    sm_bf = tl.maximum(sm_bf, 0.0)

    # layernorm: upcast to fp32, biased variance
    v = sm_bf.to(tl.float32)
    mean = tl.sum(v, axis=0) / N
    diff = tl.where(mask, v - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(LN_G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(LN_B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (v - mean) * rstd * g + b
    y_bf = y.to(tl.bfloat16)

    # + b4 in bf16
    b4 = tl.load(B4 + cols, mask=mask, other=0.0)
    out = y_bf + b4

    tl.store(OUT + row * stride_om + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 matmul (tensor cores)
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_softmax_ln_kernel[(m,)](
            h, self.ln3_g, self.ln3_b, self.b4, out,
            h.stride(0), out.stride(0),
            N=n, EPS=1e-5, BLOCK=BLOCK,
            num_warps=4,
        )
        return out
