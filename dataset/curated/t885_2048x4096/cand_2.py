import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 885
M, D, DT = 2048, 4096, torch.float16


@triton.jit
def _fused_double_softmax_kernel(
    X_ptr, B_ptr, Out_ptr,
    N, stride_x, stride_o,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # First softmax (fp32 accumulation, like PyTorch's half softmax)
    m1 = tl.max(x, axis=0)
    e1 = tl.exp(x - m1)
    s1 = tl.sum(tl.where(mask, e1, 0.0), axis=0)
    y = (e1 / s1).to(tl.float16)  # round to fp16 like reference output of softmax

    # x * 1.3127  (fp16 tensor * python float -> fp32 opmath, round to fp16)
    t = (y.to(tl.float32) * SCALE).to(tl.float16)

    # + b3  (fp16 + fp16 -> fp32 opmath, round to fp16)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    u = (t.to(tl.float32) + b).to(tl.float16)

    # Second softmax
    u32 = tl.where(mask, u.to(tl.float32), float('-inf'))
    m2 = tl.max(u32, axis=0)
    e2 = tl.exp(u32 - m2)
    s2 = tl.sum(tl.where(mask, e2, 0.0), axis=0)
    out = (e2 / s2).to(tl.float16)

    tl.store(Out_ptr + row * stride_o + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 2048, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS fp16 GEMM (same as reference)
        h = torch.matmul(x, self.W0)
        if not h.is_contiguous():
            h = h.contiguous()

        rows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_double_softmax_kernel[(rows,)](
            h, self.b3, out,
            N, h.stride(0), out.stride(0),
            SCALE=1.3127,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
