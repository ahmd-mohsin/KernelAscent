import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 782
M, D, DT = 512, 2048, torch.bfloat16


@triton.jit
def _fused_relu_gelu_rms_softmax(
    X, W, Out,
    N,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # relu
    x = tl.maximum(x, 0.0)

    # exact gelu (erf), computed in fp32 (matches PyTorch opmath), round to bf16
    x = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    x = x.to(tl.bfloat16).to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(x * x, axis=0) / N
    inv = tl.math.rsqrt(ms + EPS)
    x = x * inv
    x = x.to(tl.bfloat16).to(tl.float32)

    # scale by weight (fp32 opmath, round to bf16)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    x = x * w
    x = x.to(tl.bfloat16).to(tl.float32)

    # softmax in fp32
    x = tl.where(mask, x, float('-inf'))
    mx = tl.max(x, axis=0)
    e = tl.exp(x - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Out + row * N + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.W5 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        rows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_relu_gelu_rms_softmax[(rows,)](
            x, self.rms3_w, out, N,
            EPS=1e-6, BLOCK=BLOCK,
            num_warps=8,
        )
        return out @ self.W5
