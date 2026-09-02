import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 950
M, D, DT = 1024, 1024, torch.bfloat16


@triton.jit
def _fused_rms_rms_gelu(
    X_ptr, W1_ptr, W2_ptr, Y_ptr,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm 1 (compute in fp32, round to bf16, multiply by weight in fp32, round)
    ms1 = tl.sum(x * x, axis=0) / N
    r1 = tl.math.rsqrt(ms1 + eps)
    x1 = (x * r1).to(tl.bfloat16)
    w1 = tl.load(W1_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x1 = (x1.to(tl.float32) * w1).to(tl.bfloat16)

    # RMSNorm 2
    x2f = x1.to(tl.float32)
    ms2 = tl.sum(tl.where(mask, x2f * x2f, 0.0), axis=0) / N
    r2 = tl.math.rsqrt(ms2 + eps)
    x2 = (x2f * r2).to(tl.bfloat16)
    w2 = tl.load(W2_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x2 = (x2.to(tl.float32) * w2).to(tl.bfloat16)

    # Exact (erf-based) GELU, computed in fp32 like PyTorch's opmath, rounded to bf16
    xf = x2.to(tl.float32)
    y = xf * 0.5 * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    tl.store(Y_ptr + row * N + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 2048, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS bf16 GEMM (tensor cores)

        if not x.is_cuda:
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            return F.gelu(x)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2d = x.contiguous().view(-1, N)
        Mrows = x2d.shape[0]
        y = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_rms_rms_gelu[(Mrows,)](
            x2d, self.rms1_w, self.rms2_w, y,
            N, 1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
