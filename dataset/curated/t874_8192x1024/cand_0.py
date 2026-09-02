import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 874
M, D, DT = 8192, 1024, torch.float16


@triton.jit
def _bias_scale_softmax_kernel(
    Y, B, OUT,
    N, stride_ym, stride_om,
    SCALE,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    # load matmul result and bias in fp16 (match reference fp16 elementwise ops)
    y = tl.load(Y + row * stride_ym + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    s = tl.full((), SCALE, tl.float16)
    v = (y + b) * s  # fp16 add and mul, same as reference

    # softmax with fp32 accumulation (matches PyTorch half softmax)
    vf = v.to(tl.float32)
    vf = tl.where(mask, vf, float('-inf'))
    m = tl.max(vf, axis=0)
    e = tl.exp(vf - m)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = (e / denom).to(tl.float16)

    tl.store(OUT + row * stride_om + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS matmul (tensor cores), identical op to reference
        y = x @ self.W0
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK_N = triton.next_power_of_2(n)
        _bias_scale_softmax_kernel[(m,)](
            y, self.b1, out,
            n, y.stride(0), out.stride(0),
            1.4874,
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        return out
