import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 119
M, D, DT = 2048, 1024, torch.bfloat16


@triton.jit
def _scale_rmsnorm_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, stride_x, stride_y,
    SCALE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0)
    # emulate: x = (x * 1.4953)  computed in fp32, stored to bf16
    xs = (x.to(tl.float32) * SCALE).to(tl.bfloat16)
    # _xf = x.float()
    xf = xs.to(tl.float32)

    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + EPS)

    # (_xf * rsqrt).to(bf16) * w  (bf16*bf16 computed in fp32, stored bf16)
    y_bf16 = (xf * inv).to(tl.bfloat16)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    out = (y_bf16.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    tl.store(Y_ptr + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 4096, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK_N >= 2048 else 4
        _scale_rmsnorm_kernel[(m,)](
            x, self.rms2_w, y,
            n, x.stride(0), y.stride(0),
            SCALE=1.4953, EPS=1e-6,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y
