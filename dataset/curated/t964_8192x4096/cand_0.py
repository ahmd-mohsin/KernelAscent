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
    X, Y, G, B, B2,
    stride_xm, stride_ym,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x_ptr = X + row * stride_xm + cols
    x = tl.load(x_ptr, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax (fp32 accumulation, then round to fp16 to match PyTorch intermediate)
    row_max = tl.max(x, axis=0)
    ex = tl.exp(x - row_max)
    ex = tl.where(mask, ex, 0.0)
    denom = tl.sum(ex, axis=0)
    sm = ex / denom
    sm = sm.to(tl.float16).to(tl.float32)  # match fp16 intermediate of torch.softmax

    # layer norm in fp32
    mean = tl.sum(sm, axis=0) / N
    diff = tl.where(mask, sm - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (sm - mean) * rstd * g + b
    y16 = y.to(tl.float16)  # layer_norm output cast to fp16

    b2 = tl.load(B2 + cols, mask=mask, other=0.0)
    out = y16 + b2  # fp16 add, matching x + self.b2

    tl.store(Y + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = torch.softmax(x, dim=-1)
            y = F.layer_norm(y, (y.shape[-1],), self.ln1_g, self.ln1_b)
            return y + self.b2

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        Mrows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_softmax_ln_bias[(Mrows,)](
            x2, y, self.ln1_g, self.ln1_b, self.b2,
            x2.stride(0), y.stride(0),
            N, 1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
