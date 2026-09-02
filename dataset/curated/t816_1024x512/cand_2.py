import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 816
M, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _fused_scale_gelu_softmax(
    X_ptr, Y_ptr,
    N, stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0)

    # scale (round to bf16 to mirror eager op-by-op behavior)
    x = (x.to(tl.float32) * 1.482).to(tl.bfloat16)

    # exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    xf = x.to(tl.float32)
    g = xf * 0.5 * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # softmax (masked lanes -> -inf)
    g = tl.where(mask, g, float('-inf'))
    m = tl.max(g, axis=0)
    e = tl.exp(g - m)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)

    tl.store(Y_ptr + row * stride_y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mrows, N = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_scale_gelu_softmax[(Mrows,)](
            h, y, N, h.stride(0), y.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y
