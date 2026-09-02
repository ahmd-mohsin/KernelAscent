import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 997
M, D, DT = 4096, 4097, torch.float16


@triton.jit
def _fused_ln_softmax_kernel(
    X, OUT, G, B, B1, B3,
    N, stride_x, stride_o,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 accumulation, like PyTorch)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * g + b

    # cast to fp16 (layer_norm output), add b1 in fp16
    y16 = y.to(tl.float16)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0)
    z16 = y16 + b1

    # softmax in fp32 (like PyTorch internal), output fp16
    zf = tl.where(mask, z16.to(tl.float32), float("-inf"))
    zmax = tl.max(zf, axis=0)
    e = tl.exp(zf - zmax)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm16 = (e / s).to(tl.float16)

    # add b3 in fp16
    b3 = tl.load(B3 + cols, mask=mask, other=0.0)
    out = sm16 + b3
    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            y = y + self.b1
            y = torch.softmax(y, dim=-1)
            return y + self.b3

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        _fused_ln_softmax_kernel[(rows,)](
            x2, out, self.ln0_g, self.ln0_b, self.b1, self.b3,
            N, x2.stride(0), out.stride(0),
            1e-5,
            BLOCK=BLOCK,
            num_warps=16,
        )
        return out.view(orig_shape)
