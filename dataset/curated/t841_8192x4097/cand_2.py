import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 841
M, D, DT = 8192, 4097, torch.float16


@triton.jit
def _fused_ln3_softmax_kernel(
    X, G0, B0, G1, B1, G2, B2, OUT,
    D, EPS,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    Df = D.to(tl.float32)

    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm 0 ----
    mean = tl.sum(x, axis=0) / Df
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / Df
    rstd = 1.0 / tl.sqrt(var + EPS)
    g = tl.load(G0 + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B0 + offs, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * g + b
    y = y.to(tl.float16).to(tl.float32)  # match intermediate fp16 cast

    # ---- LayerNorm 1 ----
    mean = tl.sum(tl.where(mask, y, 0.0), axis=0) / Df
    diff = tl.where(mask, y - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / Df
    rstd = 1.0 / tl.sqrt(var + EPS)
    g = tl.load(G1 + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B1 + offs, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * g + b
    y = y.to(tl.float16).to(tl.float32)

    # ---- LayerNorm 2 ----
    mean = tl.sum(tl.where(mask, y, 0.0), axis=0) / Df
    diff = tl.where(mask, y - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / Df
    rstd = 1.0 / tl.sqrt(var + EPS)
    g = tl.load(G2 + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B2 + offs, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * g + b
    y = y.to(tl.float16).to(tl.float32)

    # ---- Softmax (fp32 accumulation, fp16 output) + ReLU (no-op on softmax) ----
    ymax = tl.max(tl.where(mask, y, float("-inf")), axis=0)
    e = tl.exp(y - ymax)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s
    out = tl.maximum(out, 0.0)

    tl.store(OUT + row * D + offs, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            y = F.layer_norm(y, (y.shape[-1],), self.ln1_g, self.ln1_b)
            y = F.layer_norm(y, (y.shape[-1],), self.ln2_g, self.ln2_b)
            return torch.relu(torch.softmax(y, dim=-1))

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        _fused_ln3_softmax_kernel[(m,)](
            x2, self.ln0_g, self.ln0_b,
            self.ln1_g, self.ln1_b,
            self.ln2_g, self.ln2_b,
            out, d, 1e-5,
            BLOCK=BLOCK,
            num_warps=16,
        )
        return out.view(orig_shape)
