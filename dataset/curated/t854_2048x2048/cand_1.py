import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 854
M, D, DT = 2048, 2048, torch.bfloat16


@triton.jit
def _softmax_bias_kernel(
    X, B, Y,
    stride_xm, stride_ym,
    N, BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf')).to(tl.float32)
    row_max = tl.max(x, axis=0)
    x = x - row_max
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    sm = num / den

    # match reference: cast softmax result to bf16, then add bf16 bias
    sm_bf = sm.to(tl.bfloat16)
    b = tl.load(B + cols, mask=mask, other=0.0)
    out = sm_bf + b

    tl.store(Y + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            return torch.softmax(x, dim=-1) + self.b1

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        Mrows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK_N >= 2048 else 4

        _softmax_bias_kernel[(Mrows,)](
            x2, self.b1, y,
            x2.stride(0), y.stride(0),
            N, BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
