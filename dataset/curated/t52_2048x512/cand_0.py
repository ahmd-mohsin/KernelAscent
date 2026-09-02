import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 52
M, D, DT = 2048, 512, torch.bfloat16


@triton.jit
def _fused_rms_relu_bias_rms(
    X, W1, B3, W4, OUT,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    x = tl.load(X + row * N + offs)                 # bf16
    xf = x.to(tl.float32)

    # RMSNorm 1 (mean over N, eps=1e-6)
    ms1 = tl.sum(xf * xf, axis=0) / N
    rs1 = 1.0 / tl.sqrt(ms1 + 1e-6)
    y = (xf * rs1).to(tl.bfloat16)                  # round like .to(x.dtype)

    w1 = tl.load(W1 + offs).to(tl.float32)
    y = (y.to(tl.float32) * w1).to(tl.bfloat16)     # bf16 * bf16 -> bf16 (fp32 math)

    # ReLU
    y = tl.maximum(y, tl.zeros_like(y))

    # + b3 (bf16 add, fp32 math then round)
    b3 = tl.load(B3 + offs).to(tl.float32)
    y = (y.to(tl.float32) + b3).to(tl.bfloat16)

    # RMSNorm 2
    yf = y.to(tl.float32)
    ms2 = tl.sum(yf * yf, axis=0) / N
    rs2 = 1.0 / tl.sqrt(ms2 + 1e-6)
    z = (yf * rs2).to(tl.bfloat16)

    w4 = tl.load(W4 + offs).to(tl.float32)
    z = (z.to(tl.float32) * w4).to(tl.bfloat16)

    tl.store(OUT + row * N + offs, z)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 1024, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        m, n = x.shape
        out = torch.empty_like(x)
        _fused_rms_relu_bias_rms[(m,)](
            x, self.rms1_w, self.b3, self.rms4_w, out,
            N=n, BLOCK=n, num_warps=8,
        )
        return out
