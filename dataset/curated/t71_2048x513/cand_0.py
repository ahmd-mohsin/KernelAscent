import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 71
M, D, DT = 2048, 513, torch.float16


@triton.jit
def _fused_softmax_bias_softmax(X, B, Y, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)

    # first softmax (fp32 accumulation, like PyTorch half softmax)
    x = tl.load(X + row * stride_x + offs).to(tl.float32)
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    s = e / tl.sum(e, axis=0)

    # cast back to fp16 (matches reference intermediate dtype), add bias in fp16
    s16 = s.to(tl.float16)
    b = tl.load(B + offs)
    t16 = s16 + b

    # second softmax (fp32 accumulation)
    t = t16.to(tl.float32)
    t = t - tl.max(t, axis=0)
    e2 = tl.exp(t)
    y = e2 / tl.sum(e2, axis=0)

    tl.store(Y + row * stride_y + offs, y.to(tl.float16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 1024, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        m, n = h.shape  # n == 1024

        out = torch.empty_like(h)
        _fused_softmax_bias_softmax[(m,)](
            h, self.b2, out,
            h.stride(0), out.stride(0),
            BLOCK=1024,
            num_warps=8,
        )
        return out
