import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 272
M, D, DT = 4096, 2049, torch.bfloat16


@triton.jit
def _fused_rms_gelu_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    N, stride_row,
    SCALE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x_bf = tl.load(X_ptr + row * stride_row + cols, mask=mask, other=0.0)
    # scale in fp32, round to bf16 (matches PyTorch elementwise opmath behavior)
    x = x_bf.to(tl.float32) * SCALE
    x = x.to(tl.bfloat16)

    # RMS norm: computed on the bf16 values upcast to fp32
    xf = x.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / N
    inv = 1.0 / tl.sqrt(ms + EPS)
    xn = (xf * inv).to(tl.bfloat16)

    # multiply by rms weight (fp32 opmath, cast bf16)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    xw = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    # exact GELU in fp32, cast bf16
    g = xw.to(tl.float32)
    g = 0.5 * g * (1.0 + tl.math.erf(g * 0.7071067811865476))
    g = g.to(tl.bfloat16)

    # add bias (fp32 opmath, cast bf16)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0)
    out = (g.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)

    tl.store(Out_ptr + row * stride_row + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 2048, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS (tensor cores)
        y = torch.matmul(x, self.W0)
        y = y.contiguous()
        rows, N = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(N)
        _fused_rms_gelu_kernel[(rows,)](
            y, self.rms2_w, self.b4, out,
            N, y.stride(0),
            SCALE=1.4684, EPS=1e-6, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
