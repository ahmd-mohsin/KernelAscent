import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 340
M, D, DT = 4096, 512, torch.float16


@triton.jit
def _softmax_scale_bias_kernel(
    X_ptr, B_ptr, Y_ptr,
    N, stride_x, stride_y,
    S1, S2,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)

    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    # Match PyTorch's per-op rounding to fp16 (half opmath = fp32 compute, fp16 store)
    y16 = y.to(tl.float16)
    y16 = (y16.to(tl.float32) * S1).to(tl.float16)
    y16 = (y16.to(tl.float32) * S2).to(tl.float16)

    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = (y16.to(tl.float32) + b).to(tl.float16)

    tl.store(Y_ptr + row * stride_y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 1024, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS fp16 GEMM (tensor cores on A100)
        h = x @ self.W0

        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)

        _softmax_scale_bias_kernel[(Mrows,)](
            h, self.b4, out,
            N, h.stride(0), out.stride(0),
            1.2811, 1.1377,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
