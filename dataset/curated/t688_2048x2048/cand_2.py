import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 688
M, D, DT = 2048, 2048, torch.float16


@triton.jit
def _fused_act_softmax(X, Y, N, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)

    # scale (round to fp16 like reference elementwise op)
    v = (x.to(tl.float32) * 1.0642).to(tl.float16)
    # relu
    v = tl.maximum(v, 0.0)
    # exact (erf) gelu, computed in fp32 then rounded to fp16 (matches CUDA half gelu)
    vf = v.to(tl.float32)
    g = 0.5 * vf * (1.0 + tl.math.erf(vf * 0.7071067811865476))
    g16 = g.to(tl.float16)

    # softmax in fp32 (matches PyTorch half softmax accumulation)
    gf = tl.where(mask, g16.to(tl.float32), float('-inf'))
    m = tl.max(gf, axis=0)
    e = tl.exp(gf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 tensor-core GEMM
        Mrows, N = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_act_softmax[(Mrows,)](
            h, y, N, h.stride(0), y.stride(0),
            BLOCK=BLOCK, num_warps=4,
        )
        return y
