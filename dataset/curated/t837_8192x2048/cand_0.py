import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

SEED = 837
M, D, DT = 8192, 2048, torch.bfloat16


@triton.jit
def _double_rms_kernel(X, W1, W2, Y, N, eps,
                       BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    ptr = X + row * N + cols

    xf = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)

    # --- RMSNorm 1 ---
    ms = tl.sum(xf * xf, axis=0) / N
    rs = 1.0 / tl.sqrt(ms + eps)
    w1 = tl.load(W1 + cols, mask=mask, other=0.0).to(tl.float32)
    # match: (xf * rsqrt).to(bf16) * w1  (bf16 elementwise mul = fp32 math, bf16 round)
    t = (xf * rs).to(tl.bfloat16).to(tl.float32) * w1
    t = t.to(tl.bfloat16)

    # --- RMSNorm 2 ---
    xf2 = t.to(tl.float32)
    ms2 = tl.sum(xf2 * xf2, axis=0) / N
    rs2 = 1.0 / tl.sqrt(ms2 + eps)
    w2 = tl.load(W2 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xf2 * rs2).to(tl.bfloat16).to(tl.float32) * w2
    y = y.to(tl.bfloat16)

    tl.store(Y + row * N + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.W3 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _double_rms_kernel[(m,)](
            x, self.rms1_w, self.rms2_w, y, n, 1e-6,
            BLOCK=BLOCK, num_warps=8,
        )
        return y @ self.W3
