import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 531
M, D, DT = 1024, 4096, torch.bfloat16


@triton.jit
def _scale_relu_softmax_kernel(
    X, Y,
    stride_xm, stride_ym,
    N: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf'))
    # multiply in fp32, then round to bf16 (matches PyTorch elementwise mul kernel)
    t = x.to(tl.float32) * SCALE
    t = t.to(tl.bfloat16).to(tl.float32)
    # relu
    t = tl.maximum(t, 0.0)
    t = tl.where(mask, t, float('-inf'))
    # softmax in fp32 accumulation (matches PyTorch bf16 softmax)
    m = tl.max(t, axis=0)
    e = tl.exp(t - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s
    tl.store(Y + row * stride_ym + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 matmul
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _scale_relu_softmax_kernel[(m,)](
            h, out,
            h.stride(0), out.stride(0),
            N=n, SCALE=1.3067, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
