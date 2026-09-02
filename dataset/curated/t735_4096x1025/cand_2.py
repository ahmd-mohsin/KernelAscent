import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 735
M, D, DT = 4096, 1025, torch.bfloat16


@triton.jit
def _bias_scale_softmax_kernel(
    X_ptr, B_ptr, Y_ptr,
    stride_x, stride_y,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)

    x = tl.load(X_ptr + row * stride_x + offs).to(tl.float32)
    b = tl.load(B_ptr + offs).to(tl.float32)

    # match PyTorch bf16 semantics: fp32 opmath, round to bf16 after each op
    y = (x + b).to(tl.bfloat16).to(tl.float32)
    y = (y * SCALE).to(tl.bfloat16).to(tl.float32)

    # softmax in fp32 (matches torch's fp32 accumulation for bf16 softmax)
    m = tl.max(y, axis=0)
    e = tl.exp(y - m)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y_ptr + row * stride_y + offs, out.to(tl.bfloat16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 2048, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS bf16 matmul (tensor cores)
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape

        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)  # 2048, no mask needed

        _bias_scale_softmax_kernel[(Mrows,)](
            h, self.b1, out,
            h.stride(0), out.stride(0),
            SCALE=1.1961,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
