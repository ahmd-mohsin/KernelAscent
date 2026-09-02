import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 928
M, D, DT = 2048, 4096, torch.float16


@triton.jit
def _fused_epilogue_softmax(
    X_ptr, B_ptr, Out_ptr,
    N, stride_row,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)

    # x * 1.0502  (fp32 opmath, rounded back to fp16 like PyTorch)
    x = (x * 1.0502).to(tl.float16).to(tl.float32)

    # relu (applied twice == once)
    x = tl.maximum(x, 0.0)

    # exact GELU (erf-based, fp32 opmath, rounded to fp16)
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)

    # + bias (fp32 opmath, rounded to fp16)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (g + b).to(tl.float16).to(tl.float32)

    # softmax in fp32
    y = tl.where(mask, y, float('-inf'))
    m = tl.max(y, axis=0)
    e = tl.exp(y - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(Out_ptr + row * stride_row + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 2048, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b5 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS GEMM (tensor cores on A100)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()

        Mrows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_epilogue_softmax[(Mrows,)](
            h, self.b5, out,
            N, h.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
