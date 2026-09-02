import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 671
M, D, DT = 8192, 1024, torch.bfloat16


@triton.jit
def _fused_softmax_gelu2_rms(
    Y_ptr, B_ptr, W_ptr, Out_ptr,
    N, stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # load matmul result row (bf16) and bias, add at fp32 opmath, round to bf16 (match torch)
    y = tl.load(Y_ptr + row * stride + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x = (y + b).to(tl.bfloat16).to(tl.float32)

    # softmax (fp32 accumulation like torch's bf16 softmax), output rounded to bf16
    x_for_max = tl.where(mask, x, float('-inf'))
    m = tl.max(x_for_max, 0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    x = (e / s).to(tl.bfloat16).to(tl.float32)

    # exact gelu (erf), applied twice, rounding to bf16 after each (match torch elementwise)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    x = (x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))).to(tl.bfloat16).to(tl.float32)
    x = (x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))).to(tl.bfloat16).to(tl.float32)

    # rmsnorm in fp32, cast to bf16, then multiply by bf16 weight (fp32 opmath -> bf16)
    xm = tl.where(mask, x, 0.0)
    ms = tl.sum(xm * xm, 0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    xn = (x * r).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = (xn * w).to(tl.bfloat16)

    tl.store(Out_ptr + row * stride + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 4096, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # heavy matmul -> cuBLAS tensor cores
        y = torch.matmul(x, self.W0)
        y = y.contiguous()
        M_, N_ = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(N_)
        _fused_softmax_gelu2_rms[(M_,)](
            y, self.b1, self.rms5_w, out,
            N_, y.stride(0),
            BLOCK=BLOCK,
            num_warps=16,
        )
        return out
