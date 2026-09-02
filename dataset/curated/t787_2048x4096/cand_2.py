import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 787
M, D, DT = 2048, 4096, torch.float16


@triton.jit
def _fused_ln3_bias_kernel(
    X, Y,
    G0, B0, G1, B1, G2, B2, B3,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm 0 (fp32 math, round to fp16 like PyTorch) ----
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G0 + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B0 + cols, mask=mask, other=0.0).to(tl.float32)
    x = (xc * rstd * g + b).to(tl.float16).to(tl.float32)

    # ---- LayerNorm 1 ----
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    x = (xc * rstd * g + b).to(tl.float16).to(tl.float32)

    # ---- LayerNorm 2 ----
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    h = (xc * rstd * g + b).to(tl.float16)

    # ---- bias add (fp16, matching x + b3 in half) ----
    b3 = tl.load(B3 + cols, mask=mask, other=0.0)
    out = h + b3

    tl.store(Y + row * N + cols, out, mask=mask)


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
            y = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            y = F.layer_norm(y, (y.shape[-1],), self.ln1_g, self.ln1_b)
            y = F.layer_norm(y, (y.shape[-1],), self.ln2_g, self.ln2_b)
            return y + self.b3

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_ln3_bias_kernel[(rows,)](
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
