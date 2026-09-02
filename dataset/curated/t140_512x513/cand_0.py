import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 140
M, D, DT = 512, 513, torch.float16


@triton.jit
def _gelu2_bias_softmax_kernel(
    Y_ptr, B_ptr, Out_ptr,
    N,
    stride_ym, stride_om,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    y = tl.load(Y_ptr + row * stride_ym + offs, mask=mask, other=0.0)

    # GELU #1 (exact erf variant, computed in fp32, cast back to fp16 like PyTorch)
    yf = y.to(tl.float32)
    g = 0.5 * yf * (1.0 + tl.math.erf(yf * 0.7071067811865476))
    g16 = g.to(tl.float16)

    # GELU #2
    gf = g16.to(tl.float32)
    g2 = 0.5 * gf * (1.0 + tl.math.erf(gf * 0.7071067811865476))
    g2_16 = g2.to(tl.float16)

    # bias add in fp16 (matches x + b3 on half tensors)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    z16 = g2_16 + b

    # softmax in fp32 (matches PyTorch half softmax which upcasts to float)
    z = z16.to(tl.float32)
    z = tl.where(mask, z, float('-inf'))
    m = tl.max(z, axis=0)
    e = tl.exp(z - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(Out_ptr + row * stride_om + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 1024, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = x @ self.W0
            y = F.gelu(y)
            y = F.gelu(y)
            y = y + self.b3
            return torch.softmax(y, dim=-1)

        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1])

        # cuBLAS matmul (same as reference)
        y = x2 @ self.W0

        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK_N = triton.next_power_of_2(n)

        _gelu2_bias_softmax_kernel[(m,)](
            y, self.b3, out,
            n,
            y.stride(0), out.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=8 if BLOCK_N >= 1024 else 4,
        )

        return out.reshape(*orig_shape[:-1], n)
