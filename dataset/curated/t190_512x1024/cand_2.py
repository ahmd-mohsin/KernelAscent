import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 190
M, D, DT = 512, 1024, torch.bfloat16


@triton.jit
def _fused_scale_rms_bias(
    Y_ptr, W_ptr, B_ptr, O_ptr,
    N, stride_row,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # load matmul output (bf16)
    y = tl.load(Y_ptr + row * stride_row + offs, mask=mask, other=0.0)

    # x = x * 1.0427 (bf16 arithmetic: fp32 compute, round to bf16)
    xb = (y.to(tl.float32) * 1.0427).to(tl.bfloat16)

    # RMSNorm in fp32
    xf = xb.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    r = tl.math.rsqrt(ms + 1e-6)
    xn = (xf * r).to(tl.bfloat16)

    # * rms2_w (bf16 mul)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0)
    t = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    # + b3 (bf16 add)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    o = (t.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)

    tl.store(O_ptr + row * stride_row + offs, o, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 4096, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS tensor cores
        y = x @ self.W0
        y = y.contiguous()

        rows = y.shape[0]
        N = y.shape[1]
        out = torch.empty_like(y)

        BLOCK = triton.next_power_of_2(N)
        _fused_scale_rms_bias[(rows,)](
            y, self.rms2_w, self.b3, out,
            N, y.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
