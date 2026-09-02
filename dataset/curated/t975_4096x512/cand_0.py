import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 975
M, D, DT = 4096, 512, torch.float16


@triton.jit
def _fused_kernel(
    X, B0, G1, Bt1, G2, Bt2, OUT,
    stride_x, stride_o,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    b0 = tl.load(B0 + cols, mask=mask, other=0.0)

    # add in fp16 to match PyTorch's fp16 elementwise add
    x = (x + b0).to(tl.float16)

    # LayerNorm 1 (fp32 accumulation, fp16 output)
    xf = x.to(tl.float32)
    mean = tl.sum(xf, axis=0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)
    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(Bt1 + cols, mask=mask, other=0.0).to(tl.float32)
    y = ((xf - mean) * rstd * g1 + b1).to(tl.float16)

    # LayerNorm 2
    yf = y.to(tl.float32)
    mean2 = tl.sum(yf, axis=0) / N
    diff2 = tl.where(mask, yf - mean2, 0.0)
    var2 = tl.sum(diff2 * diff2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)
    g2 = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(Bt2 + cols, mask=mask, other=0.0).to(tl.float32)
    z = ((yf - mean2) * rstd2 * g2 + b2).to(tl.float16)

    # Softmax (fp32 accumulation, fp16 output)
    zf = z.to(tl.float32)
    zf = tl.where(mask, zf, float("-inf"))
    zmax = tl.max(zf, axis=0)
    e = tl.exp(zf - zmax)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x + self.b0
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
            return torch.softmax(x, dim=-1)

        orig_shape = x.shape
        n = orig_shape[-1]
        x2 = x.contiguous().view(-1, n)
        m = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_kernel[(m,)](
            x2, self.b0, self.ln1_g, self.ln1_b, self.ln2_g, self.ln2_b, out,
            x2.stride(0), out.stride(0),
            N=n, BLOCK=BLOCK, EPS=1e-5,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
