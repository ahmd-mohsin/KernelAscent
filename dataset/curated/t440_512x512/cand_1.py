import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 440
M, D, DT = 512, 512, torch.bfloat16


@triton.jit
def _ln_softmax_scale_kernel(
    X, G, B, Out,
    stride_xm, stride_om,
    N: tl.constexpr,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 accumulation, matching PyTorch bf16 layer_norm)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd * g + b

    # round to bf16 like F.layer_norm output, then upcast for softmax
    y = y.to(tl.bfloat16).to(tl.float32)

    # Softmax (fp32 internally)
    y = tl.where(mask, y, float('-inf'))
    m = tl.max(y, axis=0)
    e = tl.exp(y - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s

    # round to bf16 like softmax output, then scale in fp32 (opmath), cast back
    p = p.to(tl.bfloat16).to(tl.float32)
    out = (p * SCALE).to(tl.bfloat16)

    tl.store(Out + row * stride_om + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        rows, N = x.shape[-2], x.shape[-1]
        x2d = x.view(-1, N)
        out = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(N)
        _ln_softmax_scale_kernel[(x2d.shape[0],)](
            x2d, self.ln1_g, self.ln1_b, out,
            x2d.stride(0), out.stride(0),
            N=N, EPS=1e-5, SCALE=1.1165, BLOCK=BLOCK,
            num_warps=4,
        )
        return out.view_as(x)
