import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 876
M, D, DT = 512, 2048, torch.float16


@triton.jit
def _softmax_gelu_kernel(
    X, Y,
    N, stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # numerically-stable softmax (fp32 accumulation, matching PyTorch half softmax)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s

    # exact (erf) GELU in fp32, matching F.gelu's opmath on half
    g = 0.5 * p * (1.0 + tl.math.erf(p * 0.7071067811865476))

    tl.store(Y + row * stride_y + cols, g.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores
        z = torch.matmul(x, self.W0)
        if not z.is_cuda:
            z = torch.softmax(z, dim=-1)
            return F.gelu(z)

        z = z.contiguous()
        rows, cols = z.shape[0], z.shape[-1]
        z2d = z.view(-1, cols)
        out = torch.empty_like(z2d)

        BLOCK = triton.next_power_of_2(cols)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _softmax_gelu_kernel[(z2d.shape[0],)](
            z2d, out,
            cols, z2d.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(z.shape)
