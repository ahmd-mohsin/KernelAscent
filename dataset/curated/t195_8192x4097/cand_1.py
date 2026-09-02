import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 195
M, D, DT = 8192, 4097, torch.float16


@triton.jit
def _softmax_ln_scale_kernel(
    X_ptr, G_ptr, B_ptr, Y_ptr,
    stride_x, stride_y,
    N: tl.constexpr,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax (float32 accumulation, like PyTorch on fp16 input)
    x_max = tl.max(x, axis=0)
    e = tl.exp(x - x_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    p = e / denom

    # emulate rounding of softmax output to fp16 before layernorm
    p16 = p.to(tl.float16)
    pf = p16.to(tl.float32)

    # layer norm (float32 accumulation)
    mean = tl.sum(pf, axis=0) / N
    diff = tl.where(mask, pf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    y = (pf - mean) * rstd * g + b

    # round to fp16 (layernorm output), then scale in float opmath, round to fp16
    y16 = y.to(tl.float16)
    out = (y16.to(tl.float32) * SCALE).to(tl.float16)

    tl.store(Y_ptr + row * stride_y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 2048, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 tensor-core GEMM
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _softmax_ln_scale_kernel[(Mrows,)](
            h, self.ln2_g, self.ln2_b, out,
            h.stride(0), out.stride(0),
            N=N, EPS=1e-5, SCALE=1.0319,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
