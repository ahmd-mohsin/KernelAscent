import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 346
M, D, DT = 8192, 1024, torch.float16


@triton.jit
def _fused_kernel(
    X, B, Y,
    stride_xm, stride_ym,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # GELU (exact erf variant), computed in fp32 then rounded to fp16
    # to match eager: x = F.gelu(x) produces an fp16 tensor
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * INV_SQRT2))
    g = g.to(tl.float16).to(tl.float32)

    # Softmax with fp32 accumulation (matches PyTorch half softmax)
    g_masked = tl.where(mask, g, float('-inf'))
    row_max = tl.max(g_masked, axis=0)
    e = tl.exp(g_masked - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    s = (e / denom).to(tl.float16)

    # ReLU (fp16)
    s = tl.maximum(s, tl.zeros_like(s))

    # Add bias: fp16 tensors, PyTorch computes in fp32 opmath then rounds
    b = tl.load(B + cols, mask=mask, other=0.0)
    out = (s.to(tl.float32) + b.to(tl.float32)).to(tl.float16)

    tl.store(Y + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = F.gelu(x)
            y = torch.softmax(y, dim=-1)
            y = torch.relu(y)
            return y + self.b3

        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        m, n = x2.shape
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_kernel[(m,)](
            x2, self.b3, y,
            x2.stride(0), y.stride(0),
            N=n, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
