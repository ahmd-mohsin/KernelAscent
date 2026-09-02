import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 840
M, D, DT = 1024, 4096, torch.bfloat16


@triton.jit
def _softmax_gelu_rms_kernel(
    X, W, Y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * N + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax (fp32 accumulation, matching PyTorch)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s
    # round to bf16 (softmax output dtype)
    p = p.to(tl.bfloat16).to(tl.float32)

    # exact GELU: 0.5*x*(1+erf(x/sqrt(2)))
    g = 0.5 * p * (1.0 + tl.math.erf(p * 0.7071067811865476))
    # round to bf16 (gelu output dtype)
    g = g.to(tl.bfloat16).to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(g * g, axis=0) / N
    inv = tl.math.rsqrt(ms + 1e-6)
    y = (g * inv).to(tl.bfloat16).to(tl.float32)

    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    out = (y * w).to(tl.bfloat16)

    tl.store(Y + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.W4 = nn.Parameter((torch.512 if False else torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS bf16 GEMM
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        _softmax_gelu_rms_kernel[(m,)](
            x, self.rms3_w, y,
            N=n, BLOCK=triton.next_power_of_2(n),
            num_warps=8,
        )
        out = y @ self.W4  # cuBLAS bf16 GEMM
        return torch.relu_(out)
