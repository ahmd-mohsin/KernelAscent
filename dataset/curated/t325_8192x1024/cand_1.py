import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 325
M, D, DT = 8192, 1024, torch.float16


@triton.jit
def _fused_scale_gelu_softmax(
    X, Out,
    N, stride_x, stride_o,
    S1: tl.constexpr, S2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)

    # scale 1: fp16 tensor * python scalar -> opmath fp32, result cast to fp16
    xf = x.to(tl.float32) * S1
    xf = xf.to(tl.float16).to(tl.float32)
    # scale 2
    xf = xf * S2
    xf = xf.to(tl.float16).to(tl.float32)

    # exact (erf) GELU computed in fp32, cast back to fp16 (matches CUDA half gelu)
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)

    # softmax over the row in fp32 accumulation
    g = tl.where(mask, g, float("-inf"))
    m = tl.max(g, axis=0)
    e = tl.exp(g - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Out + row * stride_o + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 1024, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100)
        y = x @ self.W0
        y = y.contiguous()
        M_, N_ = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(N_)
        _fused_scale_gelu_softmax[(M_,)](
            y, out,
            N_, y.stride(0), out.stride(0),
            1.3389, 1.4601,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
