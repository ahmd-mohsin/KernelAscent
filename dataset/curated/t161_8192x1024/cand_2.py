import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 161
M, D, DT = 8192, 1024, torch.float16


@triton.jit
def _fused_epilogue(X, B1, B2, W, Out, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    ptr = X + row * N + offs

    # load fp16 row and add biases in fp16 (matches reference rounding order)
    x = tl.load(ptr)
    b1 = tl.load(B1 + offs)
    b2 = tl.load(B2 + offs)
    x = x + b1
    x = x + b2

    # softmax #1 (fp32 accumulate, fp16 output like PyTorch half softmax)
    xf = x.to(tl.float32)
    m = tl.max(xf, axis=0)
    e = tl.exp(xf - m)
    s = tl.sum(e, axis=0)
    y = (e / s).to(tl.float16)

    # softmax #2
    yf = y.to(tl.float32)
    m2 = tl.max(yf, axis=0)
    e2 = tl.exp(yf - m2)
    s2 = tl.sum(e2, axis=0)
    z = (e2 / s2).to(tl.float16)

    # RMSNorm in fp32, cast to fp16, then fp16 multiply with weight
    zf = z.to(tl.float32)
    ms = tl.sum(zf * zf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    nrm = (zf * inv).to(tl.float16)
    w = tl.load(W + offs)
    out = nrm * w

    tl.store(Out + row * N + offs, out)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 4096, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS fp16 tensor-core GEMM
        h = torch.matmul(x, self.W0)
        Mrows, N = h.shape
        out = torch.empty_like(h)
        _fused_epilogue[(Mrows,)](
            h, self.b1, self.b2, self.rms5_w, out,
            N=N, BLOCK=4096,
            num_warps=8, num_stages=1,
        )
        return out
