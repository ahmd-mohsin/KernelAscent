import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 886
M, D, DT = 1024, 4097, torch.bfloat16


@triton.jit
def _fused_bias_double_rms(
    X_ptr, B_ptr, W2_ptr, W3_ptr, Y_ptr,
    N, stride_x, stride_y,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # bias add (float compute, round to bf16, like PyTorch opmath)
    xb = (x + b).to(tl.bfloat16)
    xf = xb.to(tl.float32)

    # first RMSNorm
    ms1 = tl.sum(xf * xf, axis=0) / N
    r1 = tl.math.rsqrt(ms1 + EPS)
    y1 = (xf * r1).to(tl.bfloat16)

    w2 = tl.load(W2_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    z = (y1.to(tl.float32) * w2).to(tl.bfloat16)
    zf = z.to(tl.float32)

    # second RMSNorm
    ms2 = tl.sum(zf * zf, axis=0) / N
    r2 = tl.math.rsqrt(ms2 + EPS)
    y2 = (zf * r2).to(tl.bfloat16)

    w3 = tl.load(W3_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = (y2.to(tl.float32) * w3).to(tl.bfloat16)

    tl.store(Y_ptr + row * stride_y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 512, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS (identical op to reference)
        h = x @ self.W0
        h = h.contiguous()
        M_, N_ = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N_)
        _fused_bias_double_rms[(M_,)](
            h, self.b1, self.rms2_w, self.rms3_w, y,
            N_, h.stride(0), y.stride(0),
            EPS=1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
