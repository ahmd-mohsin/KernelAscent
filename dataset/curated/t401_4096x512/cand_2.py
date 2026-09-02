import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 401
M, D, DT = 4096, 512, torch.bfloat16


@triton.jit
def _bias_triple_ln_kernel(
    X, B2,
    G3, B3, G4, B4, G5, B5,
    Y,
    N, stride_x, stride_y,
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0)

    # bias add in bf16 (matches reference rounding), then upcast to fp32
    v = (x + b2).to(tl.bfloat16).to(tl.float32)

    n_f = N.to(tl.float32)

    # ----- LN 1 -----
    mean = tl.sum(tl.where(mask, v, 0.0), axis=0) / n_f
    d = tl.where(mask, v - mean, 0.0)
    var = tl.sum(d * d, axis=0) / n_f
    rstd = 1.0 / tl.sqrt(var + EPS)
    g = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    v = ((v - mean) * rstd * g + b)
    v = v.to(tl.bfloat16).to(tl.float32)

    # ----- LN 2 -----
    mean = tl.sum(tl.where(mask, v, 0.0), axis=0) / n_f
    d = tl.where(mask, v - mean, 0.0)
    var = tl.sum(d * d, axis=0) / n_f
    rstd = 1.0 / tl.sqrt(var + EPS)
    g = tl.load(G4 + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B4 + cols, mask=mask, other=0.0).to(tl.float32)
    v = ((v - mean) * rstd * g + b)
    v = v.to(tl.bfloat16).to(tl.float32)

    # ----- LN 3 -----
    mean = tl.sum(tl.where(mask, v, 0.0), axis=0) / n_f
    d = tl.where(mask, v - mean, 0.0)
    var = tl.sum(d * d, axis=0) / n_f
    rstd = 1.0 / tl.sqrt(var + EPS)
    g = tl.load(G5 + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B5 + cols, mask=mask, other=0.0).to(tl.float32)
    out = ((v - mean) * rstd * g + b).to(tl.bfloat16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.W1 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln5_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln5_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x @ self.W0
            x = x @ self.W1
            x = x + self.b2
            x = F.layer_norm(x, (x.shape[-1],), self.ln3_g, self.ln3_b)
            x = F.layer_norm(x, (x.shape[-1],), self.ln4_g, self.ln4_b)
            x = F.layer_norm(x, (x.shape[-1],), self.ln5_g, self.ln5_b)
            return x

        h = torch.matmul(torch.matmul(x, self.W0), self.W1)
        h = h.contiguous()
        m, n = h.shape
        y = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(n)
        _bias_triple_ln_kernel[(m,)](
            h, self.b2,
            self.ln3_g, self.ln3_b,
            self.ln4_g, self.ln4_b,
            self.ln5_g, self.ln5_b,
            y,
            n, h.stride(0), y.stride(0),
            EPS=1e-5,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y
