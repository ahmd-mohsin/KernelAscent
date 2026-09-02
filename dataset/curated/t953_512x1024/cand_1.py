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
    N, stride_x, stride_y,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    g1 = tl.load(G1 + offs, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + offs, mask=mask, other=0.0).to(tl.float32)
    g2 = tl.load(G2 + offs, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + offs, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + offs, mask=mask, other=0.0).to(tl.float32)
    g5 = tl.load(G5 + offs, mask=mask, other=0.0).to(tl.float32)
    b5 = tl.load(B5 + offs, mask=mask, other=0.0).to(tl.float32)

    n_f = N.to(tl.float32)

    # ---- LayerNorm 1 (fp32 compute, round to bf16 like PyTorch) ----
    mean = tl.sum(x, axis=0) / n_f
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / n_f
    rstd = 1.0 / tl.sqrt(var + eps)
    y = xc * rstd * g1 + b1
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm 2 ----
    mean = tl.sum(y, axis=0) / n_f
    yc = tl.where(mask, y - mean, 0.0)
    var = tl.sum(yc * yc, axis=0) / n_f
    rstd = 1.0 / tl.sqrt(var + eps)
    y = yc * rstd * g2 + b2
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- ReLU + bias add (bf16 semantics) ----
    y = tl.maximum(y, 0.0)
    y = (y + b4).to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm 5 ----
    mean = tl.sum(y, axis=0) / n_f
    yc = tl.where(mask, y - mean, 0.0)
    var = tl.sum(yc * yc, axis=0) / n_f
    rstd = 1.0 / tl.sqrt(var + eps)
    y = yc * rstd * g5 + b5

    tl.store(Y + row * stride_y + offs, y.to(tl.bfloat16), mask=mask)


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

        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        rows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        _fused_triple_ln_kernel[(rows,)](
            h, out,
            self.ln1_g, self.ln1_b,
            self.ln2_g, self.ln2_b,
            self.b4,
            self.ln5_g, self.ln5_b,
            N, h.stride(0), out.stride(0),
            1e-5,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
