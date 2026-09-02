import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 596
M, D, DT = 4096, 1024, torch.float16


@triton.jit
def _fused_gelu_bias_relu_softmax(
    X, B, Y,
    stride_x, stride_y,
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    # exact GELU (erf-based), computed in fp32, rounded to fp16 to match op boundary
    g = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.float16)

    # bias add in fp16, then relu
    h = g + b.to(tl.float16)
    h = tl.maximum(h, 0.0)

    # softmax with fp32 accumulation
    hf = h.to(tl.float32)
    hf = tl.where(mask, hf, float('-inf'))
    m = tl.max(hf, axis=0)
    e = tl.exp(hf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * stride_y + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = F.gelu(x)
            x = x + self.b1
            x = torch.relu(x)
            return torch.softmax(x, dim=-1)

        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_gelu_bias_relu_softmax[(m,)](
            x, self.b1, y,
            x.stride(0), y.stride(0),
            n,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
