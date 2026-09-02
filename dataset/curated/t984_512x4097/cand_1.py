import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 984
M, D, DT = 512, 4097, torch.bfloat16


@triton.jit
def _fused_scale_softmax_bias(
    X, B3, B4, Out,
    stride_x, stride_o,
    N,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=float('-inf'))
    # scale computed in fp32, rounded to bf16 (matches bf16 elementwise mul), then back to fp32
    xf = x.to(tl.float32) * SCALE
    xf = xf.to(tl.bfloat16).to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))

    m = tl.max(xf, axis=0)
    e = tl.math.exp(xf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = (e / s).to(tl.bfloat16)

    b3 = tl.load(B3 + offs, mask=mask, other=0)
    b4 = tl.load(B4 + offs, mask=mask, other=0)
    # bf16 adds (fp32 compute rounded back to bf16, matching PyTorch elementwise semantics)
    y = y + b3
    y = y + b4

    tl.store(Out + row * stride_o + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 2048, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (fastest path on A100 tensor cores)
        h = x @ self.W0
        if not h.is_contiguous():
            h = h.contiguous()

        rows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_scale_softmax_bias[(rows,)](
            h, self.b3, self.b4, out,
            h.stride(0), out.stride(0),
            N,
            SCALE=1.2993,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
