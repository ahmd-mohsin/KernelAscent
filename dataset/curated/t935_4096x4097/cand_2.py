import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 935
M, D, DT = 4096, 4097, torch.float16


@triton.jit
def _gelu_f32(x):
    # exact (erf-based) GELU in fp32
    return 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))


@triton.jit
def _fused_gelu_rms_gelu2_kernel(
    X_ptr, W_ptr, Out_ptr,
    N, stride_row,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_row + cols, mask=mask, other=0.0)

    # gelu #1: compute in fp32, round back to fp16 (matches PyTorch half gelu)
    x = _gelu_f32(x.to(tl.float32)).to(tl.float16)

    # RMSNorm in fp32
    xf = x.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / N
    inv = tl.math.rsqrt(ms + eps)
    xn = (xf * inv).to(tl.float16)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    # fp16 * fp16 elementwise -> computed in fp32, cast back (PyTorch opmath)
    y = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.float16)

    # gelu #2
    y = _gelu_f32(y.to(tl.float32)).to(tl.float16)
    # gelu #3
    y = _gelu_f32(y.to(tl.float32)).to(tl.float16)

    tl.store(Out_ptr + row * stride_row + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 4096, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS GEMM (fp16, tensor cores on A100)
        y = torch.matmul(x, self.W0)
        y = y.contiguous()

        Mrows, N = y.shape
        out = torch.empty_like(y)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_gelu_rms_gelu2_kernel[(Mrows,)](
            y, self.rms2_w, out,
            N, y.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
