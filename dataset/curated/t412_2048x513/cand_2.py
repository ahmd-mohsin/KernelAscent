import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 412
M, D, DT = 2048, 513, torch.float16


@triton.jit
def _bias_relu_softmax_kernel(
    X_ptr, B_ptr, Y_ptr,
    stride_x, stride_y,
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # load matmul output (fp16) and bias (fp16)
    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)

    # bias add + relu in fp16 (matches reference elementwise fp16 math)
    v = x + b
    v = tl.maximum(v, v * 0)

    # softmax in fp32 (matches PyTorch's internal fp32 accumulation for half)
    vf = v.to(tl.float32)
    vf = tl.where(mask, vf, float('-inf'))
    row_max = tl.max(vf, axis=0)
    e = tl.exp(vf - row_max)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Y_ptr + row * stride_y + offs, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 1024, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # fallback (reference path)
            y = x @ self.W0
            y = y + self.b1
            y = torch.relu(y)
            return torch.softmax(y, dim=-1)

        # cuBLAS matmul (tensor cores on A100)
        h = torch.matmul(x, self.W0)

        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 1024 else 4

        _bias_relu_softmax_kernel[(Mrows,)](
            h, self.b1, out,
            h.stride(0), out.stride(0),
            N,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
