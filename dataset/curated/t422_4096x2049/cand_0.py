import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 422
M, D, DT = 4096, 2049, torch.float16


@triton.jit
def _relu_scale_rmsnorm_kernel(
    X_ptr, W_ptr, Out_ptr,
    stride_xm,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0)  # fp16
    # relu in fp16 (exact)
    x = tl.maximum(x, 0.0)
    # multiply by scalar, round to fp16 to match reference fp16 tensor
    xs = (x.to(tl.float32) * 1.0221).to(tl.float16)
    # rmsnorm in fp32
    xf = xs.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    y = (xf * inv).to(tl.float16)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0)  # fp16
    out = y * w
    tl.store(Out_ptr + row * stride_xm + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 512, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores)
        m, n = h.shape
        out = torch.empty_like(h)
        _relu_scale_rmsnorm_kernel[(m,)](
            h, self.rms3_w, out,
            h.stride(0),
            N=n,
            BLOCK=triton.next_power_of_2(n),
            num_warps=4,
        )
        return out
