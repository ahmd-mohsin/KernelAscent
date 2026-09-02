import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 306
M, D, DT = 4096, 1025, torch.float16


@triton.jit
def _fused_ln_softmax_kernel(
    X, G, B, B2, Y,
    N, stride_x, stride_y,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm (fp32 accumulation, round output to fp16 like PyTorch) ----
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd * g + b
    y = y.to(tl.float16).to(tl.float32)

    # ---- Softmax #1 (fp32 accumulation, fp16 output) ----
    y = tl.where(mask, y, float('-inf'))
    mx = tl.max(y, axis=0)
    e = tl.exp(y - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = (e / s).to(tl.float16).to(tl.float32)

    # ---- Add bias (exact fp16 rounding) + ReLU ----
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y + b2).to(tl.float16).to(tl.float32)
    y = tl.maximum(y, 0.0)

    # ---- Softmax #2 ----
    y = tl.where(mask, y, float('-inf'))
    mx2 = tl.max(y, axis=0)
    e2 = tl.exp(y - mx2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    out = e2 / s2

    tl.store(Y + row * stride_y + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.float16:
            # fallback: reference path
            y = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            y = torch.softmax(y, dim=-1)
            y = y + self.b2
            y = torch.relu(y)
            return torch.softmax(y, dim=-1)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_ln_softmax_kernel[(rows,)](
            x2, self.ln0_g, self.ln0_b, self.b2, out,
            N, x2.stride(0), out.stride(0),
            1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
