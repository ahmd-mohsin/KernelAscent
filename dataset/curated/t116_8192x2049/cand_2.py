import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

SEED = 116
M, D, DT = 8192, 2049, torch.float16


@triton.jit
def _relu_double_softmax_kernel(X, Y, N, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    ptr = X + row * stride + offs

    x = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
    # ReLU (masked lanes -> -inf so they vanish in softmax)
    x = tl.where(mask, tl.maximum(x, 0.0), float('-inf'))

    # softmax #1 (fp32 accumulation, matching PyTorch half softmax)
    m1 = tl.max(x, axis=0)
    e1 = tl.exp(x - m1)
    s1 = tl.sum(e1, axis=0)
    y = e1 / s1

    # round to fp16 exactly like the reference (softmax output is fp16)
    y = y.to(tl.float16).to(tl.float32)
    y = tl.where(mask, y, float('-inf'))

    # softmax #2
    m2 = tl.max(y, axis=0)
    e2 = tl.exp(y - m2)
    s2 = tl.sum(e2, axis=0)
    out = (e2 / s2).to(tl.float16)

    tl.store(Y + row * stride + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 4096, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores)
        h = x @ self.W0
        h = h.contiguous()
        rows, cols = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(cols)
        _relu_double_softmax_kernel[(rows,)](
            h, out, cols, h.stride(0),
            BLOCK=BLOCK,
            num_warps=16 if BLOCK >= 4096 else 8,
        )
        return out
