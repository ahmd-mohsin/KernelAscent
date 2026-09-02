import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 457
M, D, DT = 1024, 2048, torch.bfloat16


@triton.jit
def _fused_scale_rms_gelu(X, W, Y, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    # x = x * 1.2772  (bf16 elementwise: fp32 math, round to bf16)
    x = (x * 1.2772).to(tl.bfloat16).to(tl.float32)

    # RMS norm in fp32, then cast to bf16
    ms = tl.sum(x * x, axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    xn = (x * r).to(tl.bfloat16).to(tl.float32)

    # multiply by weight (bf16 mul: fp32 math, round to bf16)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    z = (xn * w).to(tl.bfloat16).to(tl.float32)

    # exact (erf) GELU in fp32, round to bf16
    g = 0.5 * z * (1.0 + tl.math.erf(z * 0.7071067811865476))
    tl.store(Y + row * N + cols, g.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.W4 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1])

        # GEMM 1 (tensor cores via cuBLAS)
        h = torch.matmul(x2, self.W0)
        h = h.contiguous()

        rows, N = h.shape
        out_mid = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        _fused_scale_rms_gelu[(rows,)](
            h, self.rms2_w, out_mid, N,
            BLOCK=BLOCK,
            num_warps=8,
        )

        # GEMM 2
        y = torch.matmul(out_mid, self.W4)
        return y.reshape(*orig_shape[:-1], self.W4.shape[1])
