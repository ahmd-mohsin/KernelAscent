import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 905
M, D, DT = 2048, 2048, torch.bfloat16


@triton.jit
def _softmax_rms_kernel(
    X_ptr, W_ptr, Y_ptr,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # Load row (bf16) and upcast to fp32 for softmax math (matches PyTorch opmath)
    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # Softmax in fp32, output rounded to bf16 (as torch.softmax on bf16 does)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p_bf16 = (e / s).to(tl.bfloat16)

    # RMSNorm: upcast to fp32 (matches _xf = x.float())
    pf = p_bf16.to(tl.float32)
    ms = tl.sum(pf * pf, axis=0) / N
    inv = 1.0 / tl.math.sqrt(ms + 1e-6)

    # (_xf * rsqrt).to(bf16), then bf16 * bf16 weight (computed in fp32, stored bf16)
    n_bf16 = (pf * inv).to(tl.bfloat16)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (n_bf16.to(tl.float32) * w).to(tl.bfloat16)

    tl.store(Y_ptr + row * stride_y + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100 for bf16)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mrows, N = h.shape
        y = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        _softmax_rms_kernel[(Mrows,)](
            h, self.rms2_w, y,
            h.stride(0), y.stride(0),
            N=N,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
