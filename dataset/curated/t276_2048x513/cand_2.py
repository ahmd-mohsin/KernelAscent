import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 276
M, D, DT = 2048, 513, torch.bfloat16


@triton.jit
def _fused_gelu_bias_softmax(
    Y_ptr, B_ptr, OUT_ptr,
    N, stride_row,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    y = tl.load(Y_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)

    # x = x * 1.159  (rounded to bf16 as in reference)
    z = y * 1.159
    z = z.to(tl.bfloat16).to(tl.float32)

    # exact GELU: 0.5 * z * (1 + erf(z / sqrt(2)))
    g = 0.5 * z * (1.0 + tl.math.erf(z * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # + bias
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s = (g + b).to(tl.bfloat16).to(tl.float32)

    # numerically stable softmax over the row (fp32 accumulation)
    s = tl.where(mask, s, float('-inf'))
    m = tl.max(s, axis=0)
    e = tl.exp(s - m)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(OUT_ptr + row * stride_row + offs, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 4096, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.W1 = nn.Parameter((torch.randn(4096, 2048, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # Two GEMMs via cuBLAS (tensor cores, fp32 accumulation) — matches reference order.
        y = (x @ self.W0) @ self.W1
        y = y.contiguous()

        M_rows, N = y.shape
        out = torch.empty_like(y)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_gelu_bias_softmax[(M_rows,)](
            y, self.b4, out,
            N, y.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
