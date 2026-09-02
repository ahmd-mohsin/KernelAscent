import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 787
M, D, DT = 2048, 4096, torch.float16


@triton.jit
def _fused_triple_ln_kernel(
    X, Y,
    G0, B0, G1, B1, G2, B2, B3,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    n_f = N.to(tl.float32)

    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm 0 ----
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / n_f
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / n_f
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G0 + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B0 + offs, mask=mask, other=0.0).to(tl.float32)
    x = (d * rstd) * g + b
    # match reference: intermediate cast to fp16
    x = x.to(tl.float16).to(tl.float32)

    # ---- LayerNorm 1 ----
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / n_f
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / n_f
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G1 + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B1 + offs, mask=mask, other=0.0).to(tl.float32)
    x = (d * rstd) * g + b
    x = x.to(tl.float16).to(tl.float32)

    # ---- LayerNorm 2 ----
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / n_f
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / n_f
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G2 + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B2 + offs, mask=mask, other=0.0).to(tl.float32)
    x = (d * rstd) * g + b
    x = x.to(tl.float16)

    # ---- bias add (in fp16, matching reference x + b3) ----
    b3 = tl.load(B3 + offs, mask=mask, other=0.0)
    x = x + b3

    tl.store(Y + row * N + offs, x, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
            return x + self.b3

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 4096 else 4

        _fused_triple_ln_kernel[(rows,)](
            x2, y,
            self.ln0_g, self.ln0_b,
            self.ln1_g, self.ln1_b,
            self.ln2_g, self.ln2_b,
            self.b3,
            N, 1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
