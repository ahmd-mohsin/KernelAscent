import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 908
M, D, DT = 512, 4097, torch.float16


@triton.jit
def _fused_ln_gelu_ln_ln(
    X, Y,
    G0, B0, G2, B2, G3, B3,
    N, stride_x, stride_y, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm 0 ----
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G0 + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B0 + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b
    y = y.to(tl.float16).to(tl.float32)  # round to fp16 like PyTorch op boundary

    # ---- GELU (exact, erf) ----
    y = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))
    y = y.to(tl.float16).to(tl.float32)

    # ---- LayerNorm 2 ----
    y = tl.where(mask, y, 0.0)
    mean = tl.sum(y, axis=0) / N
    d = tl.where(mask, y - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b
    y = y.to(tl.float16).to(tl.float32)

    # ---- LayerNorm 3 ----
    y = tl.where(mask, y, 0.0)
    mean = tl.sum(y, axis=0) / N
    d = tl.where(mask, y - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b

    tl.store(Y + row * stride_y + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            y = F.gelu(y)
            y = F.layer_norm(y, (y.shape[-1],), self.ln2_g, self.ln2_b)
            y = F.layer_norm(y, (y.shape[-1],), self.ln3_g, self.ln3_b)
            return y

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK <= 4096 else 16

        _fused_ln_gelu_ln_ln[(rows,)](
            x2, out,
            self.ln0_g, self.ln0_b,
            self.ln2_g, self.ln2_b,
            self.ln3_g, self.ln3_b,
            N, x2.stride(0), out.stride(0), 1e-5,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)
