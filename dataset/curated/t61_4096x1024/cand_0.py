import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 61
M, D, DT = 4096, 1024, torch.bfloat16


@triton.jit
def _fused_scale_bias_softmax(
    H_ptr, B_ptr, OUT_ptr,
    N, stride_h, stride_o,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    h = tl.load(H_ptr + row * stride_h + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # match eager bf16 rounding: mul in fp32 -> round bf16, add in fp32 -> round bf16
    v = (h * SCALE).to(tl.bfloat16).to(tl.float32)
    v = (v + b).to(tl.bfloat16).to(tl.float32)
    v = tl.where(mask, v, float('-inf'))

    m = tl.max(v, axis=0)
    e = tl.exp(v - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(OUT_ptr + row * stride_o + offs, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.W2 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM (cuBLAS/tensor cores) + in-place ReLU (exact, no rounding issues)
        h = torch.mm(x, self.W0)
        h.relu_()
        h = torch.mm(h, self.W2)

        rows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_scale_bias_softmax[(rows,)](
            h, self.b4, out,
            N, h.stride(0), out.stride(0),
            SCALE=1.1673,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
