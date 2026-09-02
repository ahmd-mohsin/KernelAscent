import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 578
M, D, DT = 2048, 1024, torch.bfloat16


@triton.jit
def _fused_bias_scale_softmax_bias(
    X, B0, B3, OUT,
    n_cols,
    stride_xm, stride_om,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    b0 = tl.load(B0 + cols, mask=mask, other=0.0)

    # x + b0 : PyTorch computes in fp32 opmath then rounds to bf16
    t = (x.to(tl.float32) + b0.to(tl.float32)).to(tl.bfloat16)
    # x * scale : fp32 opmath, round to bf16
    t = (t.to(tl.float32) * SCALE).to(tl.bfloat16)

    # softmax in fp32 (matches PyTorch's accscalar_t = float for bf16)
    tf = t.to(tl.float32)
    tf = tl.where(mask, tf, float("-inf"))
    row_max = tl.max(tf, axis=0)
    e = tl.exp(tf - row_max)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = (e / s).to(tl.bfloat16)

    # + b3 : fp32 opmath, round to bf16
    b3 = tl.load(B3 + cols, mask=mask, other=0.0)
    out = (y.to(tl.float32) + b3.to(tl.float32)).to(tl.bfloat16)

    tl.store(OUT + row * stride_om + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x + self.b0
            x = x * 1.4701
            x = torch.softmax(x, dim=-1)
            x = x + self.b3
            return x

        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        m, n = x2.shape
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(n)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _fused_bias_scale_softmax_bias[(m,)](
            x2, self.b0, self.b3, out,
            n,
            x2.stride(0), out.stride(0),
            SCALE=1.4701,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
