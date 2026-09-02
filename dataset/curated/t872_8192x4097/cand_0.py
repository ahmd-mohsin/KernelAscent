import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 872
M, D, DT = 8192, 4097, torch.bfloat16


@triton.jit
def _ln_kernel(X, G, B, Y, N, stride, EPS, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * g + b
    tl.store(Y + row * stride + cols, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _rms_relu_kernel(X, W, Y, N, stride, EPS, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    # x = x * 1.25 (done in bf16 in reference: fp32 mul then round to bf16)
    x = (x * 1.25).to(tl.bfloat16).to(tl.float32)

    ms = tl.sum(x * x, axis=0) / N
    rrms = 1.0 / tl.sqrt(ms + 1e-6)
    xn = (x * rrms).to(tl.bfloat16).to(tl.float32)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xn * w).to(tl.bfloat16)
    y = tl.maximum(y, 0.0)
    tl.store(Y + row * stride + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.W1 = nn.Parameter((torch.randn(4097, 2048, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]

        # LayerNorm (fp32 accumulation, matching F.layer_norm on bf16)
        y_ln = torch.empty_like(x2)
        BLOCK_LN = triton.next_power_of_2(d)
        _ln_kernel[(m,)](
            x2, self.ln0_g, self.ln0_b, y_ln,
            d, x2.stride(0), 1e-5,
            BLOCK=BLOCK_LN, num_warps=8,
        )

        # Matmul via cuBLAS (bf16 tensor cores)
        h = y_ln @ self.W1
        h = h.contiguous()
        n = h.shape[-1]

        # Fused: *1.25 -> RMSNorm(fp32) -> *weight -> ReLU
        out = torch.empty_like(h)
        BLOCK_RMS = triton.next_power_of_2(n)
        _rms_relu_kernel[(m,)](
            h, self.rms3_w, out,
            n, h.stride(0), 1e-6,
            BLOCK=BLOCK_RMS, num_warps=8,
        )

        return out.view(*orig_shape[:-1], n)
