import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 173
M, D, DT = 2048, 4096, torch.bfloat16


@triton.jit
def _gelu_softmax_kernel(X, Y, N, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # exact (erf-based) GELU, computed in fp32 like PyTorch's opmath
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    # round to bf16 to match the intermediate tensor dtype in the reference
    g = g.to(tl.bfloat16).to(tl.float32)

    g = tl.where(mask, g, float("-inf"))
    m = tl.max(g, axis=0)
    e = tl.exp(g - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * stride_y + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.W2 = nn.Parameter((torch.512 if False else torch.randn(512, 1024, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)

    def forward(self, x):
        # fused bias add via addmm (single cuBLAS epilogue)
        h = torch.addmm(self.b1, x, self.W0)
        z = h @ self.W2
        z = z.contiguous()
        m, n = z.shape
        out = torch.empty_like(z)
        BLOCK = triton.next_power_of_2(n)
        _gelu_softmax_kernel[(m,)](
            z, out, n, z.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=8,
        )
        return out
