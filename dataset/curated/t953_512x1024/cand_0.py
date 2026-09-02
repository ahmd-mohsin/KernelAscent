import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 953
M, D, DT = 512, 1024, torch.bfloat16


@triton.jit
def _fused_ln3_kernel(
    X, OUT,
    G1, B1, G2, B2, B4, G5, B5,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    g1 = tl.load(G1 + offs, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + offs, mask=mask, other=0.0).to(tl.float32)
    g2 = tl.load(G2 + offs, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + offs, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + offs, mask=mask, other=0.0).to(tl.float32)
    g5 = tl.load(G5 + offs, mask=mask, other=0.0).to(tl.float32)
    b5 = tl.load(B5 + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm 1 ----
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    y = xc * rstd * g1 + b1
    # round to bf16 (matches PyTorch intermediate output dtype)
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm 2 ----
    mean = tl.sum(tl.where(mask, y, 0.0), axis=0) / N
    yc = tl.where(mask, y - mean, 0.0)
    var = tl.sum(yc * yc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    y = yc * rstd * g2 + b2
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- ReLU ----
    y = tl.maximum(y, 0.0)

    # ---- add bias (fp32 opmath, round to bf16) ----
    y = (y + b4).to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm 3 ----
    mean = tl.sum(tl.where(mask, y, 0.0), axis=0) / N
    yc = tl.where(mask, y - mean, 0.0)
    var = tl.sum(yc * yc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    y = yc * rstd * g5 + b5

    tl.store(OUT + row * N + offs, y.to(tl.bfloat16), mask=mask)


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
        h = torch.matmul(x, self.W0)
        if not h.is_cuda:
            h = F.layer_norm(h, (h.shape[-1],), self.ln1_g, self.ln1_b)
            h = F.layer_norm(h, (h.shape[-1],), self.ln2_g, self.ln2_b)
            h = torch.relu(h)
            h = h + self.b4
            h = F.layer_norm(h, (h.shape[-1],), self.ln5_g, self.ln5_b)
            return h

        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_ln3_kernel[(Mrows,)](
            h, out,
            self.ln1_g, self.ln1_b,
            self.ln2_g, self.ln2_b,
            self.b4,
            self.ln5_g, self.ln5_b,
            N, 1e-5,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
