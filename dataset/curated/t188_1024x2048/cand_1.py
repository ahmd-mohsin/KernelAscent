import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 188
M, D, DT = 1024, 2048, torch.float16


@triton.jit
def _fused_relu_bias_softmax(
    X, B, Y,
    stride_xm, stride_ym,
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf')).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    # relu + bias (compute relu in fp16 semantics: relu is exact, bias add in fp16)
    x = tl.maximum(x, 0.0)
    z = (x + b).to(tl.float16).to(tl.float32)
    z = tl.where(mask, z, float('-inf'))

    zmax = tl.max(z, axis=0)
    e = tl.exp(z - zmax)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * stride_ym + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = torch.relu(x)
            x = x + self.b1
            return torch.softmax(x, dim=-1)

        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_relu_bias_softmax[(m,)](
            x, self.b1, y,
            x.stride(0), y.stride(0),
            n,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
