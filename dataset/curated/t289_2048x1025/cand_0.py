import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 289
M, D, DT = 2048, 1025, torch.float16


@triton.jit
def _fused_gelu_bias_softmax(
    y_ptr, b_ptr, out_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    ptr = y_ptr + row * N + offs

    # load matmul output (fp16) and compute exact GELU in fp32 (matches PyTorch opmath)
    x = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g16 = g.to(tl.float16)

    # bias add in fp16 (matches reference: fp16 + fp16)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    v = g16 + b

    # softmax computed in fp32 (matches PyTorch half softmax accumulation)
    vf = v.to(tl.float32)
    vf = tl.where(mask, vf, float("-inf"))
    m = tl.max(vf, axis=0)
    e = tl.exp(vf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(out_ptr + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 4096, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (fp16 tensor cores) — identical to reference matmul
        y = x @ self.W0
        y = y.contiguous()

        Mrows, N = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(N)

        _fused_gelu_bias_softmax[(Mrows,)](
            y, self.b2, out,
            N=N, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
