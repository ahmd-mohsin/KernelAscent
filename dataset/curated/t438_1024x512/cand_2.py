import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 438
M, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _fused_rms_bias_scale_relu(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, D)
    base = row * D

    # load row (bf16 -> fp32), matching _xf = x.float()
    x = tl.load(X_ptr + base + offs).to(tl.float32)

    # rms in fp32
    ms = tl.sum(x * x, axis=0) / D
    r = tl.math.rsqrt(ms + EPS)

    # (xf * rsqrt).to(bf16)
    xn = (x * r).to(tl.bfloat16)

    # * rms1_w : bf16 op computed in fp32, rounded to bf16
    w = tl.load(W_ptr + offs).to(tl.float32)
    v = (xn.to(tl.float32) * w).to(tl.bfloat16)

    # + b2 : bf16 op computed in fp32, rounded to bf16
    b = tl.load(B_ptr + offs).to(tl.float32)
    v = (v.to(tl.float32) + b).to(tl.bfloat16)

    # * 1.1065 then * 1.1693 with bf16 rounding between, as in reference
    v = (v.to(tl.float32) * S1).to(tl.bfloat16)
    v = (v.to(tl.float32) * S2).to(tl.bfloat16)

    # relu (exact in bf16)
    zero = tl.zeros_like(v)
    v = tl.maximum(v, zero)

    tl.store(Y_ptr + base + offs, v)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS bf16 matmul (tensor cores)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        m, d = h.shape
        y = torch.empty_like(h)
        _fused_rms_bias_scale_relu[(m,)](
            h, self.rms1_w, self.b2, y,
            D=d, EPS=1e-6, S1=1.1065, S2=1.1693,
            num_warps=4,
        )
        return y
