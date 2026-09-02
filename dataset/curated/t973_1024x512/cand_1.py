import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 973
M, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _fused_bias_scale_softmax(
    X, B0, B1, B2, OUT,
    stride_xm, stride_om,
    N, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    b0 = tl.load(B0 + cols, mask=mask, other=0.0)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0)

    # Replicate sequential bf16 additions with bf16 rounding at each step
    v = (x.to(tl.float32) + b0.to(tl.float32)).to(tl.bfloat16)
    v = (v.to(tl.float32) + b1.to(tl.float32)).to(tl.bfloat16)
    v = (v.to(tl.float32) + b2.to(tl.float32)).to(tl.bfloat16)
    v = (v.to(tl.float32) * scale).to(tl.bfloat16)

    # Softmax in fp32 (matches PyTorch's internal accumulation)
    vf = v.to(tl.float32)
    vf = tl.where(mask, vf, float('-inf'))
    row_max = tl.max(vf, axis=0)
    e = tl.math.exp(vf - row_max)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(OUT + row * stride_om + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            v = x + self.b0
            v = v + self.b1
            v = v + self.b2
            v = v * 1.3976
            return torch.softmax(v, dim=-1)

        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        m, n = x2.shape
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_bias_scale_softmax[(m,)](
            x2, self.b0, self.b1, self.b2, out,
            x2.stride(0), out.stride(0),
            n, 1.3976,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)
