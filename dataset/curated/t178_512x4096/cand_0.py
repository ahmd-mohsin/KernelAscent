import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 178
M, D, DT = 512, 4096, torch.bfloat16


@triton.jit
def _fused_relu_rms2_kernel(
    X, W2, W3, B4, OUT,
    stride_x, stride_o,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    # relu (in bf16, same as torch)
    x = tl.maximum(x, 0.0)

    xf = x.to(tl.float32)
    # RMSNorm 1
    ms1 = tl.sum(xf * xf, axis=0) / N
    rstd1 = 1.0 / tl.sqrt(ms1 + eps)
    y1 = (xf * rstd1).to(tl.bfloat16)  # round to bf16 like torch .to(dtype)

    w2 = tl.load(W2 + cols, mask=mask, other=0.0)
    # bf16 * bf16 computed in fp32 then rounded to bf16 (matches torch elementwise)
    y1w = (y1.to(tl.float32) * w2.to(tl.float32)).to(tl.bfloat16)

    # RMSNorm 2
    xf2 = y1w.to(tl.float32)
    ms2 = tl.sum(tl.where(mask, xf2 * xf2, 0.0), axis=0) / N
    rstd2 = 1.0 / tl.sqrt(ms2 + eps)
    y2 = (xf2 * rstd2).to(tl.bfloat16)

    w3 = tl.load(W3 + cols, mask=mask, other=0.0)
    y2w = (y2.to(tl.float32) * w3.to(tl.float32)).to(tl.bfloat16)

    b4 = tl.load(B4 + cols, mask=mask, other=0.0)
    out = (y2w.to(tl.float32) + b4.to(tl.float32)).to(tl.bfloat16)

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 matmul (tensor cores)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_relu_rms2_kernel[(Mrows,)](
            h, self.rms2_w, self.rms3_w, self.b4, out,
            h.stride(0), out.stride(0),
            N, 1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
