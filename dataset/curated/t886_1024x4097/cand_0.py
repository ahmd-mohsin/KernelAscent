import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

SEED = 886
M, D, DT = 1024, 4097, torch.bfloat16


@triton.jit
def _fused_bias_double_rms_kernel(
    X_ptr, B_ptr, W2_ptr, W3_ptr, Out_ptr,
    N, stride_xm, stride_om,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0)

    # x = x + b1 (bf16 add -> compute in f32, round to bf16)
    xf = x.to(tl.float32) + b.to(tl.float32)
    xf = xf.to(tl.bfloat16).to(tl.float32)

    # first RMSNorm
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)
    xn = (xf * inv).to(tl.bfloat16).to(tl.float32)
    w2 = tl.load(W2_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    xf = (xn * w2).to(tl.bfloat16).to(tl.float32)

    # second RMSNorm
    ms2 = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / N
    inv2 = 1.0 / tl.sqrt(ms2 + eps)
    xn2 = (xf * inv2).to(tl.bfloat16).to(tl.float32)
    w3 = tl.load(W3_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = (xn2 * w3).to(tl.bfloat16)

    tl.store(Out_ptr + row * stride_om + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 512, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # cuBLAS GEMM (bf16)
        Mrows, N = y.shape
        out = torch.empty_like(y)
        BLOCK_N = triton.next_power_of_2(N)
        _fused_bias_double_rms_kernel[(Mrows,)](
            y, self.b1, self.rms2_w, self.rms3_w, out,
            N, y.stride(0), out.stride(0),
            1e-6,
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        return out
