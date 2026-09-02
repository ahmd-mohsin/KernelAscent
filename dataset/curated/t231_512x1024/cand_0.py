import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 231
M, D, DT = 512, 1024, torch.bfloat16


@triton.jit
def _rms_softmax_relu_kernel(
    X_ptr, W_ptr, Out_ptr,
    N, stride_x, stride_o,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMSNorm (mean over N elements)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)

    # cast normalized value back to bf16, multiply by bf16 weight (bf16 math), like reference
    xn = (xf * inv).to(tl.bfloat16)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0)
    z = (xn * w).to(tl.float32)

    # softmax in fp32
    z = tl.where(mask, z, float('-inf'))
    zmax = tl.max(z, axis=0)
    e = tl.exp(z - zmax)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    # relu is a no-op on softmax output (>= 0), kept implicitly
    tl.store(Out_ptr + row * stride_o + offs, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 2048, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS bf16 matmul (tensor cores)
        x = x.contiguous()
        m, n = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _rms_softmax_relu_kernel[(m,)](
            x, self.rms1_w, out,
            n, x.stride(0), out.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
