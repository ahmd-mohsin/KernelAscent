import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 640
M, D, DT = 512, 4097, torch.bfloat16


@triton.jit
def _fused_softmax_ln_gelu(
    X, Y, G, B,
    N, stride_x, stride_y,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # ---- softmax (fp32 math, round to bf16 like PyTorch output) ----
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    s = e / denom
    # round to bf16 (softmax output dtype) then back to fp32 for LN input
    s = s.to(tl.bfloat16).to(tl.float32)

    # ---- layer norm (fp32 math) ----
    n_f = N.to(tl.float32)
    mean = tl.sum(tl.where(mask, s, 0.0), axis=0) / n_f
    diff = tl.where(mask, s - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / n_f
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (s - mean) * rstd * g + b
    # round to bf16 (LN output dtype) then fp32 for gelu math
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- gelu (exact, erf) in fp32 ----
    out = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))

    tl.store(Y + row * stride_y + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = torch.softmax(x, dim=-1)
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            return F.gelu(x)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        Mrows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK <= 4096 else 16

        _fused_softmax_ln_gelu[(Mrows,)](
            x2, y, self.ln1_g, self.ln1_b,
            N, x2.stride(0), y.stride(0),
            1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
