import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 928
M, D, DT = 2048, 4096, torch.float16


@triton.jit
def _fused_act_softmax(X, B, Out, N, stride_x, stride_o, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # x = x * 1.0502  (fp16 binary op: compute in fp32, round to fp16)
    x = (x * 1.0502).to(tl.float16).to(tl.float32)

    # relu(relu(x)) == relu(x), exact in any precision
    x = tl.maximum(x, 0.0)

    # exact (erf) GELU, computed in fp32 then rounded to fp16 (matches PyTorch half kernel)
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)

    # bias add (fp16 op with fp32 opmath, rounded to fp16)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (g + b).to(tl.float16).to(tl.float32)

    # softmax in fp32 (matches PyTorch half softmax which accumulates in float)
    y = tl.where(mask, y, float('-inf'))
    m = tl.max(y, axis=0)
    e = tl.exp(y - m)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(Out + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 2048, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b5 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores (already optimal on A100)
        h = x @ self.W0
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_act_softmax[(Mrows,)](
            h, self.b5, out,
            N, h.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out
