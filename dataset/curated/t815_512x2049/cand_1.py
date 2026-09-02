import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 815
M, D, DT = 512, 2049, torch.bfloat16


@triton.jit
def _rms_scale_relu_kernel(
    X, W, Y,
    stride_xm,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_xm + offs, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # rms in fp32
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)

    # normalized, rounded to bf16 (matches .to(x.dtype))
    y = (xf * inv).to(tl.bfloat16).to(tl.float32)

    # * rms1_w  (bf16 elementwise op computes in fp32, rounds to bf16)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    y = (y * w).to(tl.bfloat16).to(tl.float32)

    # * 1.3892 then * 1.0246, each rounded to bf16
    y = (y * 1.3892).to(tl.bfloat16).to(tl.float32)
    y = (y * 1.0246).to(tl.bfloat16).to(tl.float32)

    # relu
    y = tl.maximum(y, 0.0)

    tl.store(Y + row * N + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 512, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (optimal on A100 tensor cores)
        x = x @ self.W0

        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _rms_scale_relu_kernel[(m,)](
            x, self.rms1_w, y,
            x.stride(0),
            n,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return y
