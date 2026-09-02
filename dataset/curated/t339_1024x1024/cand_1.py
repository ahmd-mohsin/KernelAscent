import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 339
M, D, DT = 1024, 1024, torch.float16


@triton.jit
def _fused_gelu_bias_gelu_scale_softmax(
    X_ptr, B_ptr, Out_ptr,
    N,
    stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # GELU (exact, erf-based) computed in fp32, rounded back to fp16 (matches PyTorch opmath)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    x = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.float16).to(tl.float32)

    # + bias (fp32 compute -> fp16 round == correctly-rounded fp16 add)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x = (x + b).to(tl.float16).to(tl.float32)

    # second GELU
    x = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.float16).to(tl.float32)

    # scale
    x = (x * 1.437).to(tl.float16).to(tl.float32)

    # softmax over the row in fp32 (matches PyTorch half softmax which accumulates in float)
    x = tl.where(mask, x, float('-inf'))
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = (e / denom).to(tl.float16)

    tl.store(Out_ptr + row * stride_o + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (fp16 in, fp32 accumulate) - identical to reference matmul
        y = torch.matmul(x, self.W0)
        y = y.contiguous()
        m, n = y.shape
        out = torch.empty_like(y)

        BLOCK = triton.next_power_of_2(n)
        _fused_gelu_bias_gelu_scale_softmax[(m,)](
            y, self.b2, out,
            n,
            y.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
