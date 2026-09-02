import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 742
M, D, DT = 4096, 2049, torch.float16


@triton.jit
def _bias_relu_softmax_kernel(
    Y_ptr, B_ptr, O_ptr,
    N,
    stride_ym, stride_om,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # load matmul output (fp16) and bias (fp16)
    y = tl.load(Y_ptr + row * stride_ym + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)

    # bias add in fp16 (matches x + b1 in half precision), then relu
    v16 = y + b
    vf = v16.to(tl.float32)
    vf = tl.maximum(vf, 0.0)

    # softmax in fp32 (matches PyTorch half softmax accumulate type)
    vf = tl.where(mask, vf, float('-inf'))
    m = tl.max(vf, axis=0)
    e = tl.exp(vf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(O_ptr + row * stride_om + offs, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 2048, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS TensorCore matmul (same as reference x @ W0)
        y = torch.matmul(x, self.W0)
        y = y.contiguous()

        Mrows, N = y.shape
        out = torch.empty_like(y)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _bias_relu_softmax_kernel[(Mrows,)](
            y, self.b1, out,
            N,
            y.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
