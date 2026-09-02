import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 953
M, D, DT = 512, 1024, torch.bfloat16


@triton.jit
def _fused_triple_ln_kernel(
    X, Y,
    G1, B1, G2, B2, B4, G5, B5,
    stride_x, stride_y,
    N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm 1 (fp32 math, round to bf16 like PyTorch) ----
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)
    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * g1 + b1
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm 2 ----
    mean2 = tl.sum(tl.where(mask, y, 0.0), axis=0) / N
    yc = tl.where(mask, y - mean2, 0.0)
    var2 = tl.sum(yc * yc, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)
    g2 = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    z = yc * rstd2 * g2 + b2
    z = z.to(tl.bfloat16).to(tl.float32)

    # ---- ReLU + bias (bf16 rounding like PyTorch elementwise) ----
    z = tl.maximum(z, 0.0)
    b4 = tl.load(B4 + cols, mask=mask, other=0.0).to(tl.float32)
    z = (z + b4).to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm 5 ----
    mean5 = tl.sum(tl.where(mask, z, 0.0), axis=0) / N
    zc = tl.where(mask, z - mean5, 0.0)
    var5 = tl.sum(zc * zc, axis=0) / N
    rstd5 = 1.0 / tl.sqrt(var5 + EPS)
    g5 = tl.load(G5 + cols, mask=mask, other=0.0).to(tl.float32)
    b5 = tl.load(B5 + cols, mask=mask, other=0.0).to(tl.float32)
    out = zc * rstd5 * g5 + b5

    tl.store(Y + row * stride_y + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln5_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln5_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: reference path
            x = x @ self.W0
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
            x = torch.relu(x)
            x = x + self.b4
            x = F.layer_norm(x, (x.shape[-1],), self.ln5_g, self.ln5_b)
            return x

        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1])

        # cuBLAS matmul (tensor cores)
        h = torch.matmul(x2, self.W0)
        h = h.contiguous()

        Mrows, N = h.shape
        y = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        _fused_triple_ln_kernel[(Mrows,)](
            h, y,
            self.ln1_g, self.ln1_b,
            self.ln2_g, self.ln2_b,
            self.b4,
            self.ln5_g, self.ln5_b,
            h.stride(0), y.stride(0),
            N=N, EPS=1e-5, BLOCK=BLOCK,
            num_warps=4,
        )

        return y.reshape(*orig_shape[:-1], N)
