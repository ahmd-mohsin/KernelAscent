import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 598
M, D, DT = 512, 4096, torch.float16


@triton.jit
def _fused_row_kernel(
    X, G, B, W, Y,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    base = row * D

    # load input, upcast to fp32
    x = tl.load(X + base + offs, mask=mask, other=0.0).to(tl.float32)

    # relu
    x = tl.maximum(x, 0.0)

    # layer norm (fp32 accumulation, like PyTorch's half layer_norm)
    mean = tl.sum(x, axis=0) / D
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    inv = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    x = (x - mean) * inv * g + b

    # round to fp16 (layer_norm output dtype), back to fp32
    x = x.to(tl.float16).to(tl.float32)

    # relu
    x = tl.maximum(x, 0.0)

    # softmax (fp32 internal, fp16 output)
    mval = tl.max(tl.where(mask, x, float('-inf')), axis=0)
    e = tl.where(mask, tl.exp(x - mval), 0.0)
    s = tl.sum(e, axis=0)
    x = e / s
    x = x.to(tl.float16).to(tl.float32)

    # rms norm in fp32
    ms = tl.sum(x * x, axis=0) / D
    x = x * (1.0 / tl.sqrt(ms + 1e-6))

    # cast to fp16, multiply by weight in fp16 (matches reference)
    w = tl.load(W + offs, mask=mask, other=0.0)
    y = x.to(tl.float16) * w

    tl.store(Y + base + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        _fused_row_kernel[(m,)](
            x2, self.ln1_g, self.ln1_b, self.rms4_w, y,
            D=d, BLOCK=BLOCK,
            num_warps=8,
        )
        return y.view(orig_shape)
