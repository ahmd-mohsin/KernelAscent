import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 550
M, D, DT = 4096, 4097, torch.bfloat16


@triton.jit
def _softmax_bias_kernel(
    X, B, Y,
    stride_xm,
    stride_ym,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_xm + offs, mask=mask, other=float('-inf')).to(tl.float32)

    row_max = tl.max(x, axis=0)
    x = x - row_max
    num = tl.exp(x)
    denom = tl.sum(tl.where(mask, num, 0.0), axis=0)
    sm = num / denom

    # round softmax to bf16 (matches PyTorch output dtype), then add bias
    # (PyTorch bf16 add uses fp32 opmath, then rounds)
    sm_bf16 = sm.to(tl.bfloat16)
    b = tl.load(B + offs, mask=mask, other=0.0)
    out = (sm_bf16.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)

    tl.store(Y + row * stride_ym + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 512, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 GEMM (tensor cores)
        m, n = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _softmax_bias_kernel[(m,)](
            h, self.b2, y,
            h.stride(0), y.stride(0),
            N=n, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
