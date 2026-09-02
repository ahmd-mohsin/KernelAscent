import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 30
M, D, DT = 8192, 2048, torch.float16


@triton.jit
def _fused_bias_softmax_scale_rms_kernel(
    X, B, W, OUT,
    N: tl.constexpr,
    SCALE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # load matmul output (fp16) and bias (fp16)
    x = tl.load(X + row * N + offs, mask=mask, other=0.0)
    b = tl.load(B + offs, mask=mask, other=0.0)

    # bias add (fp16 arithmetic done in fp32 then rounded == PyTorch half add)
    h = (x.to(tl.float32) + b.to(tl.float32)).to(tl.float16)

    # softmax with fp32 accumulation (matches PyTorch half softmax), output fp16
    f = h.to(tl.float32)
    f = tl.where(mask, f, float('-inf'))
    m = tl.max(f, axis=0)
    e = tl.exp(f - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p16 = (e / s).to(tl.float16)

    # scale by 1.0129 (half * scalar -> compute in fp32, round to fp16)
    y16 = (p16.to(tl.float32) * SCALE).to(tl.float16)

    # RMSNorm in fp32, cast to fp16, then multiply by weight
    xf = y16.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + EPS)
    n16 = (xf * r).to(tl.float16)

    w = tl.load(W + offs, mask=mask, other=0.0)
    out = (n16.to(tl.float32) * w.to(tl.float32)).to(tl.float16)
    tl.store(OUT + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100)
        y = torch.matmul(x, self.W0)
        y = y.contiguous()

        Mrows, N = y.shape
        out = torch.empty_like(y)

        BLOCK = triton.next_power_of_2(N)
        _fused_bias_softmax_scale_rms_kernel[(Mrows,)](
            y, self.b1, self.rms4_w, out,
            N=N,
            SCALE=1.0129,
            EPS=1e-6,
            BLOCK=BLOCK,
            num_warps=16,
        )
        return out
