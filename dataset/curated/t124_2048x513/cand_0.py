import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 124
M, D, DT = 2048, 513, torch.float16


@triton.jit
def _relu_bias_softmax_scale(X, B, Y, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * N + offs, mask=mask, other=0.0)   # fp16
    b = tl.load(B + offs, mask=mask, other=0.0)             # fp16

    # relu(relu(x)) == relu(x); bias add in fp16 (matches reference dtype semantics)
    v = tl.maximum(x, 0.0) + b

    # softmax with fp32 accumulation (matches PyTorch half softmax)
    vf = tl.where(mask, v.to(tl.float32), float('-inf'))
    m = tl.max(vf, axis=0)
    e = tl.exp(vf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = (e / s).to(tl.float16)

    # scale in fp16 (matches half tensor * python scalar)
    scale = tl.full((), 1.1092, dtype=tl.float16)
    out = p * scale

    tl.store(Y + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 2048, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS half GEMM (tensor cores on A100)
        h = x @ self.W0
        h = h.contiguous()
        rows, N = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _relu_bias_softmax_scale[(rows,)](
            h, self.b3, y, N,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
