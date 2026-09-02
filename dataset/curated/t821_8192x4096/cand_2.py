import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 821
M, D, DT = 8192, 4096, torch.bfloat16


@triton.jit
def _fused_ln_softmax_ln(
    X, Y, G1, B1, G3, B3,
    stride_x, stride_y,
    N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm 1 (fp32 math, round to bf16 like PyTorch output)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)
    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * g1 + b1
    y = y.to(tl.bfloat16).to(tl.float32)

    # Softmax
    y_m = tl.where(mask, y, float('-inf'))
    mx = tl.max(y_m, axis=0)
    e = tl.exp(y_m - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s
    p = p.to(tl.bfloat16).to(tl.float32)

    # LayerNorm 3
    mean2 = tl.sum(p, axis=0) / N
    pc = tl.where(mask, p - mean2, 0.0)
    var2 = tl.sum(pc * pc, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)
    g3 = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    out = pc * rstd2 * g3 + b3

    tl.store(Y + row * stride_y + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 matmul (tensor cores)
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_ln_softmax_ln[(m,)](
            h, out,
            self.ln1_g, self.ln1_b, self.ln3_g, self.ln3_b,
            h.stride(0), out.stride(0),
            N=n, EPS=1e-5, BLOCK=BLOCK,
            num_warps=4,
        )
        return out
