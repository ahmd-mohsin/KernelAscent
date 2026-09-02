import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 372
M, D, DT = 512, 2049, torch.float16


@triton.jit
def _fused_softmax2_rms_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # ---- softmax 1 (fp32 math, output rounded to fp16 like PyTorch) ----
    m1 = tl.max(x, 0)
    e1 = tl.exp(x - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, 0)
    x_h = (e1 / s1).to(tl.float16)

    # relu is a no-op on softmax output (all values > 0)

    # ---- softmax 2 (input is the fp16-rounded result) ----
    xf = x_h.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))
    m2 = tl.max(xf, 0)
    e2 = tl.exp(xf - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, 0)
    x_h = (e2 / s2).to(tl.float16)

    # ---- scale by 1.2112 (opmath fp32, rounded back to fp16) ----
    x_h = (x_h.to(tl.float32) * 1.2112).to(tl.float16)

    # ---- RMS norm in fp32, round to fp16, then multiply by weight ----
    xf = x_h.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), 0) / N
    xn_h = (xf * tl.rsqrt(ms + 1e-6)).to(tl.float16)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    y = (xn_h.to(tl.float32) * w.to(tl.float32)).to(tl.float16)

    tl.store(Y_ptr + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 1024, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100)
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        y = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 1024 else 4
        _fused_softmax2_rms_kernel[(Mrows,)](
            h, self.rms5_w, y,
            N, h.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
