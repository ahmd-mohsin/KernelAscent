import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 178
M, D, DT = 512, 4096, torch.bfloat16


@triton.jit
def _fused_relu_double_rmsnorm_bias(
    X_ptr, W2_ptr, W3_ptr, B4_ptr, Out_ptr,
    N, stride_row,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_row + cols, mask=mask, other=0.0)
    # relu
    x = tl.maximum(x, 0.0)

    # first RMSNorm (compute in fp32, round to bf16, then multiply by weight in bf16 semantics)
    xf = x.to(tl.float32)
    ms1 = tl.sum(xf * xf, axis=0) / N
    inv1 = 1.0 / tl.sqrt(ms1 + eps)
    y1 = (xf * inv1).to(tl.bfloat16)
    w2 = tl.load(W2_ptr + cols, mask=mask, other=0.0)
    y1 = (y1.to(tl.float32) * w2.to(tl.float32)).to(tl.bfloat16)

    # second RMSNorm
    yf = y1.to(tl.float32)
    ms2 = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0) / N
    inv2 = 1.0 / tl.sqrt(ms2 + eps)
    y2 = (yf * inv2).to(tl.bfloat16)
    w3 = tl.load(W3_ptr + cols, mask=mask, other=0.0)
    y2 = (y2.to(tl.float32) * w3.to(tl.float32)).to(tl.bfloat16)

    # bias add (bf16 add semantics: fp32 compute, round)
    b4 = tl.load(B4_ptr + cols, mask=mask, other=0.0)
    out = (y2.to(tl.float32) + b4.to(tl.float32)).to(tl.bfloat16)

    tl.store(Out_ptr + row * stride_row + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS bf16 matmul (tensor cores)
        x = x.contiguous()
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_relu_double_rmsnorm_bias[(Mrows,)](
            x, self.rms2_w, self.rms3_w, self.b4, out,
            N, x.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
