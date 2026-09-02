import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 242
M, D, DT = 2048, 512, torch.bfloat16


@triton.jit
def _fused_post_kernel(
    X_ptr, W2_ptr, G_ptr, B_ptr, Out_ptr,
    N, stride_xm, stride_om,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0)

    # x = x * 1.303 (bf16 rounding as in reference)
    xf = x.to(tl.float32) * 1.303
    x_bf = xf.to(tl.bfloat16)

    # RMSNorm in float32
    _xf = x_bf.to(tl.float32)
    ms = tl.sum(_xf * _xf, axis=0) / N
    rinv = 1.0 / tl.sqrt(ms + 1e-6)
    normed_bf = (_xf * rinv).to(tl.bfloat16)

    w2 = tl.load(W2_ptr + cols, mask=mask, other=0.0)
    # bf16 * bf16 -> bf16 (computed in fp32, rounded)
    y_bf = (normed_bf.to(tl.float32) * w2.to(tl.float32)).to(tl.bfloat16)

    # LayerNorm in float32
    yf = y_bf.to(tl.float32)
    mean = tl.sum(yf, axis=0) / N
    diff = tl.where(mask, yf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv_std = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = (yf - mean) * inv_std * g + b

    tl.store(Out_ptr + row * stride_om + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # cuBLAS bf16 tensor-core matmul
        Mrows, N = y.shape
        out = torch.empty_like(y)
        BLOCK_N = triton.next_power_of_2(N)
        _fused_post_kernel[(Mrows,)](
            y, self.rms2_w, self.ln3_g, self.ln3_b, out,
            N, y.stride(0), out.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
