import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 178
M, D, DT = 512, 4096, torch.bfloat16


@triton.jit
def _fused_relu_rms2_bias_kernel(
    X_ptr, W2_ptr, W3_ptr, B4_ptr, Out_ptr,
    N, stride_x, stride_o,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    # relu
    x = tl.maximum(x, 0.0)

    # first RMSNorm
    ms1 = tl.sum(x * x, axis=0) / N
    r1 = 1.0 / tl.sqrt(ms1 + EPS)
    xn = (x * r1).to(tl.bfloat16).to(tl.float32)
    w2 = tl.load(W2_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x = (xn * w2).to(tl.bfloat16).to(tl.float32)

    # second RMSNorm
    ms2 = tl.sum(x * x, axis=0) / N
    r2 = 1.0 / tl.sqrt(ms2 + EPS)
    xn2 = (x * r2).to(tl.bfloat16).to(tl.float32)
    w3 = tl.load(W3_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x = (xn2 * w3).to(tl.bfloat16).to(tl.float32)

    # bias add
    b4 = tl.load(B4_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x + b4).to(tl.bfloat16)

    tl.store(Out_ptr + row * stride_o + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS matmul (bf16 tensor cores)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_relu_rms2_bias_kernel[(Mrows,)](
            h, self.rms2_w, self.rms3_w, self.b4, out,
            N, h.stride(0), out.stride(0),
            EPS=1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
