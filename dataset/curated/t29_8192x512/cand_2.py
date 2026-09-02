import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 29
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _fused_gelu_bias_softmax(
    X, B, Y,
    stride_xm, stride_ym,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    # exact GELU in fp32, then round to bf16 to match F.gelu(x) on bf16 input
    xf = x.to(tl.float32)
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    g_bf = g.to(tl.bfloat16)

    # bias add in bf16 (matches x + b1 on bf16 tensors), then softmax in fp32
    s = (g_bf + b).to(tl.float32)
    s = tl.where(mask, s, float('-inf'))

    m = tl.max(s, axis=0)
    e = tl.exp(s - m)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Y + row * stride_ym + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        if not x.is_cuda:
            x = F.gelu(x)
            x = x + self.b1
            return torch.softmax(x, dim=-1)

        orig_shape = x.shape
        n = orig_shape[-1]
        x2 = x.view(-1, n)
        m = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(n)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_gelu_bias_softmax[(m,)](
            x2, self.b1, y,
            x2.stride(0), y.stride(0),
            n, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
