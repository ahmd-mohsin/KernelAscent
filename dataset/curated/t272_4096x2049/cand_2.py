import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 272
M, D, DT = 4096, 2049, torch.bfloat16


@triton.jit
def _fused_scale_rms_gelu_bias(
    X, W, B, Out,
    N, stride,
    eps, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    # load matmul output (bf16), promote to fp32
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)

    # x = x * 1.4684  (computed in fp32, stored/rounded to bf16 as in reference)
    x = (x * scale).to(tl.bfloat16).to(tl.float32)

    # RMS norm in fp32
    ms = tl.sum(x * x, axis=0) / N
    rs = 1.0 / tl.sqrt(ms + eps)
    y = (x * rs).to(tl.bfloat16).to(tl.float32)  # .to(x.dtype) in reference

    # * rms2_w (bf16 op -> opmath fp32, round back to bf16)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y * w).to(tl.bfloat16).to(tl.float32)

    # exact GELU (erf), fp32 opmath, round to bf16
    g = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # + b4, fp32 opmath, round to bf16
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    out = (g + b).to(tl.bfloat16)

    tl.store(Out + row * stride + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 2048, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS bf16 matmul with fp32 accumulate (same as reference)
        h = x @ self.W0
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_scale_rms_gelu_bias[(m,)](
            h, self.rms2_w, self.b4, out,
            n, h.stride(0),
            1e-6, 1.4684,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
