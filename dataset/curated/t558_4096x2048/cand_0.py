import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 558
M, D, DT = 4096, 2048, torch.float16


@triton.jit
def _fused_scale_bias_softmax(
    Y_ptr, B_ptr, O_ptr,
    N, stride_y, stride_o,
    S1, S2,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    y = tl.load(Y_ptr + row * stride_y + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)

    # replicate fp16 rounding of each elementwise op in the reference:
    # t = (y * 1.4099).half(); t = (t + b).half(); t = (t * 1.4586).half()
    t = (y.to(tl.float32) * S1).to(tl.float16)
    t = (t.to(tl.float32) + b.to(tl.float32)).to(tl.float16)
    t = (t.to(tl.float32) * S2).to(tl.float16)

    # softmax in fp32 (matches PyTorch fp16 softmax which accumulates in float)
    f = tl.where(mask, t.to(tl.float32), float('-inf'))
    m = tl.max(f, axis=0)
    e = tl.exp(f - m)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(O_ptr + row * stride_o + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores
        y = x @ self.W0
        y = y.contiguous()

        Mrows, N = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(N)

        _fused_scale_bias_softmax[(Mrows,)](
            y, self.b2, out,
            N, y.stride(0), out.stride(0),
            1.4099, 1.4586,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
