import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 196
M, D, DT = 8192, 512, torch.float16


@triton.jit
def _fused_kernel(
    X, B0, G, B, B3, OUT,
    N, stride_x, stride_o,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    # load in fp16
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    b0 = tl.load(B0 + cols, mask=mask, other=0.0)

    # add in fp16 (match PyTorch half add), then round to fp16
    x = (x + b0).to(tl.float16)

    # layer norm computed in fp32
    xf = x.to(tl.float32)
    mean = tl.sum(tl.where(mask, xf, 0.0), axis=0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xf - mean) * rstd * g + b
    y = y.to(tl.float16)  # layer_norm output is fp16

    # softmax 1 (fp32 accumulation, fp16 output)
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, float('-inf'))
    m1 = tl.max(yf, axis=0)
    e1 = tl.exp(yf - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    p1 = (e1 / s1).to(tl.float16)

    # add b3 in fp16
    b3 = tl.load(B3 + cols, mask=mask, other=0.0)
    z = (p1 + b3).to(tl.float16)

    # softmax 2
    zf = z.to(tl.float32)
    zf = tl.where(mask, zf, float('-inf'))
    m2 = tl.max(zf, axis=0)
    e2 = tl.exp(zf - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    p2 = (e2 / s2).to(tl.float16)

    tl.store(OUT + row * stride_o + cols, p2, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x + self.b0
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            x = torch.softmax(x, dim=-1)
            x = x + self.b3
            return torch.softmax(x, dim=-1)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        Mrows = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_kernel[(Mrows,)](
            x2, self.b0, self.ln1_g, self.ln1_b, self.b3, out,
            N, x2.stride(0), out.stride(0),
            1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
