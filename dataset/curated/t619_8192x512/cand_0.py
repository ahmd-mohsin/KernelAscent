import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 619
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _fused_softmax_scale_bias(
    X, B, Y,
    stride_xm, stride_ym,
    N,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float("-inf")).to(tl.float32)

    row_max = tl.max(x, axis=0)
    x = x - row_max
    num = tl.exp(x)
    denom = tl.sum(num, axis=0)
    sm = num / denom

    # match PyTorch: softmax output cast to bf16, then scale (fp32 opmath -> bf16),
    # then bias add (fp32 opmath -> bf16)
    sm_bf = sm.to(tl.bfloat16).to(tl.float32)
    scaled = (sm_bf * SCALE).to(tl.bfloat16).to(tl.float32)

    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    out = (scaled + b).to(tl.bfloat16)

    tl.store(Y + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = torch.softmax(x, dim=-1)
            y = y * 1.0686
            return y + self.b2

        orig_shape = x.shape
        n = orig_shape[-1]
        x2 = x.contiguous().view(-1, n)
        m = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK_N = triton.next_power_of_2(n)
        num_warps = 4 if BLOCK_N <= 1024 else 8

        _fused_softmax_scale_bias[(m,)](
            x2, self.b2, out,
            x2.stride(0), out.stride(0),
            n,
            SCALE=1.0686,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
