import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 973
M, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _fused_bias_scale_softmax_kernel(
    X, B0, B1, B2, OUT,
    stride_xm, stride_om,
    N, SCALE,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(B0 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)

    # Replicate PyTorch bf16 elementwise semantics: compute in fp32 (opmath),
    # round to bf16 after each op.
    t = (x + b0).to(tl.bfloat16).to(tl.float32)
    t = (t + b1).to(tl.bfloat16).to(tl.float32)
    t = (t + b2).to(tl.bfloat16).to(tl.float32)
    t = (t * SCALE).to(tl.bfloat16).to(tl.float32)

    # Softmax in fp32 accumulation (matches PyTorch bf16 softmax internals)
    t = tl.where(mask, t, float('-inf'))
    row_max = tl.max(t, axis=0)
    e = tl.exp(t - row_max)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(OUT + row * stride_om + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x + self.b0
            x = x + self.b1
            x = x + self.b2
            x = x * 1.3976
            return torch.softmax(x, dim=-1)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        Mrows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK_N <= 1024 else 8

        _fused_bias_scale_softmax_kernel[(Mrows,)](
            x2, self.b0, self.b1, self.b2, out,
            x2.stride(0), out.stride(0),
            N, 1.3976,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
