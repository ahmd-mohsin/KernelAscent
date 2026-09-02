import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 208
M, D, DT = 2048, 1025, torch.bfloat16


@triton.jit
def _scale_relu_softmax_kernel(
    X_ptr, Y_ptr,
    N, stride_x, stride_y,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0)
    # scale in fp32, round back to bf16 to match reference elementwise mul,
    # then upcast for softmax (matches PyTorch softmax fp32 accumulation)
    x32 = x.to(tl.float32) * SCALE
    x32 = x32.to(tl.bfloat16).to(tl.float32)
    # relu (relu twice == relu once)
    x32 = tl.maximum(x32, 0.0)

    x32 = tl.where(mask, x32, float('-inf'))
    m = tl.max(x32, axis=0)
    e = tl.exp(x32 - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s
    # relu(softmax) == softmax since softmax > 0

    tl.store(Y_ptr + row * stride_y + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 4096, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS bf16 GEMM (tensor cores)
        h = x @ self.W0
        h = h.contiguous()
        rows, cols = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(cols)
        num_warps = 8 if BLOCK >= 2048 else 4
        _scale_relu_softmax_kernel[(rows,)](
            h, out,
            cols, h.stride(0), out.stride(0),
            SCALE=1.0675,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out


def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(M, D, generator=g).to(DT)]
