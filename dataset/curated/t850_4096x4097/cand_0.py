import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 850
M, D, DT = 4096, 4097, torch.bfloat16


@triton.jit
def _relu_scale_softmax_kernel(
    X_ptr, Y_ptr,
    N, stride_x, stride_y,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # relu (bf16 relu is exact, so fp32 relu is identical)
    xf = tl.maximum(xf, 0.0)

    # x * 1.2179 : PyTorch computes in fp32 then rounds to bf16
    y = xf * SCALE
    y_bf = y.to(tl.bfloat16)
    y2 = y_bf.to(tl.float32)

    # softmax in fp32 (matches PyTorch bf16 softmax which accumulates in fp32)
    y_masked = tl.where(mask, y2, float('-inf'))
    row_max = tl.max(y_masked, axis=0)
    e = tl.exp(y_masked - row_max)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y_ptr + row * stride_y + offs, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 512, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (bf16 with fp32 accumulation, same as reference)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()

        Mrows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        grid = (Mrows,)
        _relu_scale_softmax_kernel[grid](
            h, out,
            N, h.stride(0), out.stride(0),
            SCALE=1.2179,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
