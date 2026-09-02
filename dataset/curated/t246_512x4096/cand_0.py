import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 246
M, D, DT = 512, 4096, torch.float16


@triton.jit
def _fused_softmax_rms_relu_softmax(
    X, W, B, Y,
    N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, N)
    ptr = X + row * N + offs

    x = tl.load(ptr).to(tl.float32)

    # ---- softmax #1 (fp32 accumulate, matching PyTorch half softmax) ----
    m1 = tl.max(x, axis=0)
    e1 = tl.exp(x - m1)
    s1 = tl.sum(e1, axis=0)
    p = e1 / s1
    p16 = p.to(tl.float16)          # PyTorch stores softmax result in fp16

    # ---- RMSNorm: recast to fp32, mean of squares, rsqrt ----
    xf = p16.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    y16 = (xf * inv).to(tl.float16)  # cast to fp16 like reference

    # ---- scale, bias, relu (fp16 arithmetic like reference) ----
    w = tl.load(W + offs)
    b = tl.load(B + offs)
    y16 = y16 * w
    y16 = y16 + b
    zero = tl.zeros_like(y16)
    y16 = tl.maximum(y16, zero)

    # ---- softmax #2 ----
    yf = y16.to(tl.float32)
    m2 = tl.max(yf, axis=0)
    e2 = tl.exp(yf - m2)
    s2 = tl.sum(e2, axis=0)
    out = (e2 / s2).to(tl.float16)

    tl.store(Y + row * N + offs, out)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS fp16 GEMM (tensor cores), identical to reference matmul
        h = x @ self.W0
        h = h.contiguous()

        rows, N = h.shape
        out = torch.empty_like(h)

        _fused_softmax_rms_relu_softmax[(rows,)](
            h, self.rms2_w, self.b3, out,
            N=N,
            num_warps=8,
        )
        return out
