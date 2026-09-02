import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 964
M, D, DT = 8192, 4096, torch.float16


@triton.jit
def _fused_softmax_ln_bias(
    X, OUT, G, B, B2,
    N, eps,
    stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax (fp32 accumulation, like PyTorch half softmax)
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    sm = e / denom
    # round to fp16 (softmax output dtype), then back to fp32 for layernorm
    sm16 = sm.to(tl.float16)
    y = sm16.to(tl.float32)

    # layernorm in fp32
    mean = tl.sum(tl.where(mask, y, 0.0), axis=0) / N
    diff = tl.where(mask, y - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    ln = (y - mean) * rstd * g + b
    ln16 = ln.to(tl.float16)

    # bias add in fp16 (matches PyTorch half add)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0)
    out = ln16 + b2

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = torch.softmax(x, dim=-1)
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            return x + self.b2

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_softmax_ln_bias[(rows,)](
            x2, out, self.ln1_g, self.ln1_b, self.b2,
            N, 1e-5,
            x2.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
