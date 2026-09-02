import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 984
M, D, DT = 512, 4097, torch.bfloat16


@triton.jit
def _fused_softmax_bias(
    x_ptr, b3_ptr, b4_ptr, out_ptr,
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # load matmul output (bf16), scale in fp32, round back to bf16 (match eager)
    x = tl.load(x_ptr + row * N + offs, mask=mask, other=float('-inf')).to(tl.float32)
    x = x * 1.2993
    x = x.to(tl.bfloat16).to(tl.float32)
    x = tl.where(mask, x, float('-inf'))

    # softmax in fp32
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = (e / s).to(tl.bfloat16)

    # + b3 (fp32 opmath, round to bf16), then + b4 (fp32 opmath, round to bf16)
    b3 = tl.load(b3_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(b4_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (y.to(tl.float32) + b3).to(tl.bfloat16)
    y = (y.to(tl.float32) + b4).to(tl.bfloat16)

    tl.store(out_ptr + row * N + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 2048, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # heavy lifting via cuBLAS (tensor cores)
        h = torch.matmul(x, self.W0)
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_softmax_bias[(Mrows,)](
            h, self.b3, self.b4, out,
            N,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
