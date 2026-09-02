import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 764
M, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _fused_kernel(X, Y, G3, B3, G4, B4,
                  N_COLS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N_COLS

    x = tl.load(X + row * N_COLS + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax (fp32 math, round to bf16 like eager op boundary)
    m = tl.max(x, 0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    sm = (e / s).to(tl.bfloat16).to(tl.float32)

    # scale
    v = (sm * 1.3875).to(tl.bfloat16).to(tl.float32)

    # exact GELU
    g = (0.5 * v * (1.0 + tl.math.erf(v * 0.7071067811865476))).to(tl.bfloat16).to(tl.float32)

    # LayerNorm 1
    g_m = tl.where(mask, g, 0.0)
    mean1 = tl.sum(g_m, 0) / N_COLS
    d1 = tl.where(mask, g - mean1, 0.0)
    var1 = tl.sum(d1 * d1, 0) / N_COLS
    r1 = 1.0 / tl.sqrt(var1 + 1e-5)
    g3 = tl.load(G3 + offs, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + offs, mask=mask, other=0.0).to(tl.float32)
    y1 = (d1 * r1 * g3 + b3).to(tl.bfloat16).to(tl.float32)

    # LayerNorm 2
    y1_m = tl.where(mask, y1, 0.0)
    mean2 = tl.sum(y1_m, 0) / N_COLS
    d2 = tl.where(mask, y1 - mean2, 0.0)
    var2 = tl.sum(d2 * d2, 0) / N_COLS
    r2 = 1.0 / tl.sqrt(var2 + 1e-5)
    g4 = tl.load(G4 + offs, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + offs, mask=mask, other=0.0).to(tl.float32)
    out = (d2 * r2 * g4 + b4).to(tl.bfloat16)

    tl.store(Y + row * N_COLS + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = torch.softmax(x, dim=-1)
            y = y * 1.3875
            y = F.gelu(y)
            y = F.layer_norm(y, (y.shape[-1],), self.ln3_g, self.ln3_b)
            y = F.layer_norm(y, (y.shape[-1],), self.ln4_g, self.ln4_b)
            return y

        orig_shape = x.shape
        n_cols = orig_shape[-1]
        x2 = x.contiguous().view(-1, n_cols)
        n_rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_kernel[(n_rows,)](
            x2, out,
            self.ln3_g, self.ln3_b, self.ln4_g, self.ln4_b,
            N_COLS=n_cols, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
