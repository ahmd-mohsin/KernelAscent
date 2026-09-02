import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 482
M, D, DT = 8192, 2048, torch.float16


@triton.jit
def _fused_epilogue_kernel(
    X_ptr, B_ptr, Out_ptr,
    N, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)

    # scale in fp16 (mimic reference fp16 elementwise ops)
    v = (x.to(tl.float32) * 1.024).to(tl.float16)
    v = (v.to(tl.float32) + b.to(tl.float32)).to(tl.float16)

    # softmax in fp32 accumulation (matches PyTorch fp16 softmax internals)
    f = v.to(tl.float32)
    f = tl.where(mask, f, float('-inf'))
    row_max = tl.max(f, axis=0)
    e = tl.exp(f - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    p = (e / denom).to(tl.float16)  # round softmax output to fp16

    # exact (erf-based) GELU computed in fp32
    pf = p.to(tl.float32)
    g = 0.5 * pf * (1.0 + tl.math.erf(pf * 0.7071067811865476))

    tl.store(Out_ptr + row * stride_o + offs, g.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS (tensor cores)
        y = x @ self.W0

        if not y.is_cuda:
            y = y * 1.024
            y = y + self.b2
            y = torch.softmax(y, dim=-1)
            return F.gelu(y)

        y = y.contiguous()
        m, n = y.shape
        out = torch.empty_like(y)

        BLOCK = triton.next_power_of_2(n)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _fused_epilogue_kernel[(m,)](
            y, self.b2, out,
            n, y.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
