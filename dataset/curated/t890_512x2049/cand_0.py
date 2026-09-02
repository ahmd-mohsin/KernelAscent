import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 890
M, D, DT = 512, 2049, torch.bfloat16


@triton.jit
def _rmsnorm_softmax_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, stride_x, stride_y,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # RMS norm (mean of squares over the row) in fp32
    ms = tl.sum(x * x, axis=0) / N
    r = 1.0 / tl.sqrt(ms + EPS)

    w = tl.load(W_ptr + offs, mask=mask, other=0.0)  # bf16

    # Match reference: (fp32 normalized) -> bf16, then bf16 multiply with weight
    y_bf16 = (x * r).to(tl.bfloat16) * w
    yf = y_bf16.to(tl.float32)

    # Softmax in fp32
    yf = tl.where(mask, yf, float('-inf'))
    m = tl.max(yf, axis=0)
    e = tl.exp(yf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)

    tl.store(Y_ptr + row * stride_y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 2048, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (already optimal on A100 tensor cores)
        x = x @ self.W0

        if not x.is_cuda:
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            return torch.softmax(x, dim=-1)

        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _rmsnorm_softmax_kernel[(Mrows,)](
            x, self.rms1_w, y,
            N, x.stride(0), y.stride(0),
            EPS=1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
