import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 106
M, D, DT = 4096, 512, torch.bfloat16


@triton.jit
def _softmax_kernel(
    X, Y,
    stride_xm, stride_ym,
    N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf')).to(tl.float32)
    row_max = tl.max(x, axis=0)
    x = x - row_max
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    y = num / den
    tl.store(Y + row * stride_ym + cols, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _bias_add_kernel(
    X, B, Y,
    stride_xm, stride_ym,
    N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + row * stride_xm + cols, mask=mask)
    b = tl.load(B + cols, mask=mask)
    tl.store(Y + row * stride_ym + cols, x + b, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.W1 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, d = x.shape

        # Fused bias add (elementwise, one pass) -> same math as x + b0
        xb = torch.empty_like(x)
        BLOCK_D = triton.next_power_of_2(d)
        _bias_add_kernel[(m,)](
            x, self.b0, xb,
            x.stride(0), xb.stride(0),
            d,
            BLOCK_N=BLOCK_D,
            num_warps=4,
        )

        # Tensor-core GEMM via cuBLAS (fastest for this shape on A100)
        logits = torch.matmul(xb, self.W1)

        # Fused single-pass row softmax in fp32 accumulation, bf16 output
        n = logits.shape[1]
        out = torch.empty_like(logits)
        BLOCK_N = triton.next_power_of_2(n)
        _softmax_kernel[(m,)](
            logits, out,
            logits.stride(0), out.stride(0),
            n,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out


def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(M, D, generator=g).to(DT)]
