import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 466
M, D, DT = 512, 513, torch.float16


@triton.jit
def _ln_scale_bias_kernel(
    X, G, B, B3, Out,
    N, eps, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)

    # layer_norm in fp32 (matches PyTorch mixed-precision LN for fp16), cast to fp16
    y16 = (d * rstd * g + b).to(tl.float16)
    # x * 1.2009 : opmath fp32, round back to fp16 (matches PyTorch half elementwise)
    t16 = (y16.to(tl.float32) * scale).to(tl.float16)
    # x + b3 : opmath fp32, round back to fp16
    b3 = tl.load(B3 + offs, mask=mask, other=0.0).to(tl.float32)
    out = (t16.to(tl.float32) + b3).to(tl.float16)

    tl.store(Out + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 4096, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = torch.matmul(x, self.W0)  # cuBLAS tensor-core GEMM
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _ln_scale_bias_kernel[(Mrows,)](
            h, self.ln1_g, self.ln1_b, self.b3, out,
            N, 1e-5, 1.2009,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
