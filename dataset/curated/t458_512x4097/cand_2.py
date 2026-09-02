import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 458
M, D, DT = 512, 4097, torch.bfloat16


@triton.jit
def _fused_ln_softmax_kernel(
    X, G, B, B3, Out,
    stride_x, stride_o,
    N: tl.constexpr,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    # ReLU (in bf16, same as reference), then upcast to fp32
    x = tl.maximum(x, 0.0).to(tl.float32)

    # LayerNorm in fp32
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd * g + b
    # round to bf16 as layer_norm output does
    y = y.to(tl.bfloat16).to(tl.float32)

    # + b3 (bf16 result)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y + b3).to(tl.bfloat16).to(tl.float32)

    # * scale (bf16 result)
    y = (y * SCALE).to(tl.bfloat16).to(tl.float32)

    # softmax in fp32
    y = tl.where(mask, y, float('-inf'))
    m = tl.max(y, axis=0)
    e = tl.exp(y - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Out + row * stride_o + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 512, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        rows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_ln_softmax_kernel[(rows,)](
            h, self.ln2_g, self.ln2_b, self.b3, out,
            h.stride(0), out.stride(0),
            N=N, EPS=1e-5, SCALE=1.1477, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
